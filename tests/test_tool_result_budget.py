"""验证最新工具结果的预算与文件转存，不请求模型 API。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from fake_model import ToolCallingModel
from lota import workspace
from lota.agent import build_agent


class ToolResultBudgetTest(unittest.TestCase):
  """通过真实 Agent 验证工具结果在送给模型前完成转存。"""

  def setUp(self):
    """使用临时工作区，避免写入仓库。"""
    temporary = self.enterContext(TemporaryDirectory())
    self.workdir = Path(temporary).resolve()
    self.output_dir = self.workdir / ".task_outputs" / "tool-results"
    self.enterContext(patch.object(workspace, "WORKDIR", self.workdir))
    self.enterContext(patch("builtins.print"))


  def read_batch(self, outputs: list[str], prefix: str = "read") -> AIMessage:
    """为每个输出创建临时文件，并构造对应的一批文件读取调用。"""
    calls = []
    for index, output in enumerate(outputs):
      call_id = f"{prefix}-{index}"
      path = self.workdir / f"{call_id}.txt"
      path.write_text(output, encoding="utf-8")
      calls.append({
        "name": "read_file", "args": {"path": path.name}, "id": call_id,
      })
    return AIMessage(content="", tool_calls=calls)


  def test_archives_largest_result_and_preserves_tool_message(self):
    """超预算时优先归档最大的结果，模型与历史都保留调用配对。"""
    outputs = ["甲" * 45_000, "乙" * 85_000, "丙" * 90_000]
    model = ToolCallingModel(responses=[
      self.read_batch(outputs), AIMessage(content="done"),
    ])
    old_output = "旧" * 210_000
    result = build_agent(model).invoke({"messages": [
      HumanMessage(content="旧请求"),
      AIMessage(content="", tool_calls=[{
        "name": "read_file", "args": {"path": "old.txt"}, "id": "old",
      }]),
      ToolMessage(content=old_output, tool_call_id="old"),
      AIMessage(content="previous answer"),
      HumanMessage(content="读取三份文件"),
    ]})

    self.assertEqual(len(model.calls), 2)
    model_results = [m for m in model.calls[-1] if isinstance(m, ToolMessage)]
    saved, = self.output_dir.iterdir()
    self.assertEqual(saved.read_text(encoding="utf-8"), outputs[-1])
    preview = (
      f"<persisted-output>\nFull output: {saved.relative_to(self.workdir)}\n"
      f"Preview:\n{outputs[-1][:2000]}\n</persisted-output>"
    )
    self.assertEqual([m.tool_call_id for m in model_results], [
      "old", "read-0", "read-1", "read-2",
    ])
    self.assertEqual([m.content for m in model_results], [
      old_output, *outputs[:2], preview,
    ])
    self.assertEqual(model_results[-1].name, "read_file")
    self.assertEqual(
      [m for m in result["messages"] if isinstance(m, ToolMessage)],
      model_results,
    )


  def test_batch_at_budget_is_unchanged(self):
    """单条虽大，但整批没有超过预算时不转存。"""
    outputs = ["甲" * 150_000, "乙" * 50_000]
    model = ToolCallingModel(responses=[
      self.read_batch(outputs), AIMessage(content="done"),
    ])
    build_agent(model).invoke({"messages": [HumanMessage(content="读取文件")]})

    results = [m.content for m in model.calls[-1] if isinstance(m, ToolMessage)]
    self.assertEqual(results, outputs)
    self.assertFalse(self.output_dir.exists())


  def test_over_budget_batch_keeps_small_results(self):
    """超预算但没有单条大结果时，不裁剪或转存任何结果。"""
    outputs = ["甲" * 30_000] * 7
    model = ToolCallingModel(responses=[
      self.read_batch(outputs), AIMessage(content="done"),
    ])
    build_agent(model).invoke({"messages": [HumanMessage(content="读取文件")]})

    results = [m.content for m in model.calls[-1] if isinstance(m, ToolMessage)]
    self.assertEqual(results, outputs)
    self.assertFalse(self.output_dir.exists())


  def test_subagent_applies_its_own_batch_budget(self):
    """子 Agent 也转存大结果，父 Agent 仍只收到子任务结论。"""
    output = "子" * 200_001
    model = ToolCallingModel(responses=[
      AIMessage(content="", tool_calls=[{
        "name": "run_subagent", "args": {"prompt": "读取子任务文件"},
        "id": "delegate",
      }]),
      self.read_batch([output], prefix="child"),
      AIMessage(content="child done"), AIMessage(content="parent done"),
    ])
    build_agent(model).invoke({"messages": [HumanMessage(content="委派任务")]})

    saved, = self.output_dir.iterdir()
    self.assertEqual(saved.read_text(encoding="utf-8"), output)
    child_result = model.calls[2][-1]
    self.assertEqual(child_result.tool_call_id, "child-0")
    self.assertIn(str(saved.relative_to(self.workdir)), child_result.content)
    parent_result = model.calls[3][-1]
    self.assertEqual(parent_result.tool_call_id, "delegate")
    self.assertEqual(parent_result.content, "child done")
