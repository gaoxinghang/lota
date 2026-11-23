"""验证微压缩只缩短已读的旧工具结果。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from fake_model import ToolCallingModel
from lota import workspace
from lota.agent import build_agent


class MicroCompactTest(unittest.TestCase):
  """通过离线 Agent 验证原文保留、工具配对和压缩时机。"""

  def setUp(self):
    """将工具输入和转存文件放在临时工作区。"""
    temporary = self.enterContext(TemporaryDirectory())
    self.workdir = Path(temporary).resolve()
    self.output_dir = self.workdir / ".task_outputs" / "tool-results"
    self.enterContext(patch.object(workspace, "WORKDIR", self.workdir))
    self.enterContext(patch("builtins.print"))


  def read_batch(self, outputs):
    """为测试文本创建文件并构造一批读取调用。"""
    calls = []
    for index, output in enumerate(outputs):
      path = self.workdir / f"input-{index}.txt"
      path.write_text(output, encoding="utf-8")
      calls.append({
        "name": "read_file", "args": {"path": path.name}, "id": f"read-{index}",
      })
    return AIMessage(content="", tool_calls=calls)


  def run_turn(self, history, *replies):
    """执行一轮并返回模型收到的消息和 Agent 最终历史。"""
    model = ToolCallingModel(responses=[*replies, AIMessage(content="完成")])
    result = build_agent(model).invoke({"messages": history})
    return model.calls, result["messages"]


  def results(self, messages):
    """按原顺序取出工具结果。"""
    return [message for message in messages if isinstance(message, ToolMessage)]


  def test_old_results_unchanged_below_limit(self):
    """历史未超限，即使有可压缩的旧结果也不写盘。"""
    outputs = ["甲" * 5_000] * 5
    _, history = self.run_turn(
      [HumanMessage(content="读取文件")], self.read_batch(outputs),
    )
    calls, _ = self.run_turn([*history, HumanMessage(content="继续")])

    self.assertEqual([m.content for m in self.results(calls[0])], outputs)
    self.assertFalse(self.output_dir.exists())


  def test_archives_oldest_long_result_until_target(self):
    """跳过短结果，从最早的长结果压缩，降到目标后停止。"""
    outputs = ["短" * 120, "甲" * 25_000, *(["乙" * 8_000] * 2)]
    outputs += ["近" * 4_000] * 3
    _, history = self.run_turn(
      [HumanMessage(content="读取文件")], self.read_batch(outputs),
    )
    calls, after = self.run_turn([*history, HumanMessage(content="继续")])

    saved, = self.output_dir.glob("*.txt")
    self.assertEqual(saved.read_text(encoding="utf-8"), outputs[1])
    result = self.results(calls[0])
    self.assertEqual(result[0].content, outputs[0])
    self.assertIn(str(saved.relative_to(self.workdir)), result[1].content)
    self.assertEqual([m.content for m in result[2:]], outputs[2:])
    self.assertEqual(result[1].id, self.results(history)[1].id)
    self.assertEqual(result[1].name, "read_file")
    self.assertEqual(result[1].tool_call_id, "read-1")
    self.assertEqual(self.results(after), result)
    self.assertEqual(len(calls), 1)


  def test_latest_batch_is_not_compacted_before_model_reads_it(self):
    """新结果先完整交给模型，之后也保留最近三条已读结果。"""
    outputs = ["甲" * 20_000] * 4
    calls, history = self.run_turn(
      [HumanMessage(content="读取文件")], self.read_batch(outputs),
    )
    self.assertEqual([m.content for m in self.results(calls[-1])], outputs)
    self.assertFalse(self.output_dir.exists())

    calls, _ = self.run_turn([*history, HumanMessage(content="继续")])
    saved, = self.output_dir.glob("*.txt")
    results = self.results(calls[0])
    self.assertIn(str(saved.relative_to(self.workdir)), results[0].content)
    self.assertEqual([m.content for m in results[1:]], outputs[1:])


  def test_snip_runs_before_micro_compact(self):
    """第二步已经裁掉的工具结果，不再被第三步单独转存。"""
    outputs = ["甲" * 55_000, "乙", "丙", "丁"]
    _, history = self.run_turn([
      HumanMessage(content="开始"), AIMessage(content="好"),
      HumanMessage(content="读取文件"),
    ], self.read_batch(outputs))
    for _ in range(20):
      history += [HumanMessage(content="继续"), AIMessage(content="好")]
    calls, _ = self.run_turn([*history, HumanMessage(content="结束")])

    self.assertEqual(self.results(calls[0]), [])
    transcript, = (self.workdir / ".transcripts").glob("*.jsonl")
    self.assertIn(outputs[0], transcript.read_text(encoding="utf-8"))
    self.assertFalse(self.output_dir.exists())


  def test_reuses_first_step_archive_without_rewriting_preview(self):
    """第一步已经保存原文时，只替换预览，不再次转存。"""
    outputs = ["甲" * 210_000, *(["乙" * 20_000] * 3)]
    _, history = self.run_turn(
      [HumanMessage(content="读取文件")], self.read_batch(outputs),
    )
    saved, = self.output_dir.glob("*.txt")
    path = str(saved.relative_to(self.workdir))
    self.assertIn("Preview:", self.results(history)[0].content)

    calls, after = self.run_turn([*history, HumanMessage(content="继续")])
    compacted = self.results(calls[0])[0].content
    self.assertIn(path, compacted)
    self.assertNotIn("Preview:", compacted)
    self.assertEqual(saved.read_text(encoding="utf-8"), outputs[0])
    self.assertEqual(list(self.output_dir.glob("*.txt")), [saved])

    calls, _ = self.run_turn([*after, HumanMessage(content="再次继续")])
    self.assertEqual(self.results(calls[0])[0].content, compacted)
    self.assertEqual(list(self.output_dir.glob("*.txt")), [saved])
