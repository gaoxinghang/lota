"""使用离线模型验证消息归档及裁剪边界。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import (
  AIMessage, HumanMessage, ToolMessage, messages_from_dict,
)

from fake_model import ToolCallingModel
from lota import workspace
from lota.agent import build_agent


class SnipCompactTest(unittest.TestCase):
  """验证传给模型的历史与 Agent 返回的历史都已裁剪。"""

  def setUp(self):
    """将归档和工具输出限制在测试的临时工作区。"""
    temporary = self.enterContext(TemporaryDirectory())
    self.workdir = Path(temporary).resolve()
    self.transcripts = self.workdir / ".transcripts"
    self.enterContext(patch.object(workspace, "WORKDIR", self.workdir))
    self.enterContext(patch("builtins.print"))


  def text_history(self, count):
    """创建 count 条带编号的消息，最后一条为用户输入。"""
    return [
      (HumanMessage if index % 2 == 0 or index == count - 1 else AIMessage)(
        content=f"消息 {index}"
      )
      for index in range(count)
    ]


  def batch(self, prefix, size):
    """创建含 size 次无副作用清单操作的模型回复。"""
    return AIMessage(content="", tool_calls=[
      {"name": "todo_write", "args": {"todos": []}, "id": f"{prefix}-{i}"}
      for i in range(size)
    ])


  def completed_batch(self, prefix, size):
    """为历史构造一批带对应结果的工具调用。"""
    return [self.batch(prefix, size), *[
      ToolMessage(content="No todos.", tool_call_id=f"{prefix}-{i}")
      for i in range(size)
    ]]


  def run_agent(self, history, *replies):
    """执行一轮，返回测试模型及 Agent 的新历史。"""
    model = ToolCallingModel(responses=[*replies, AIMessage(content="完成")])
    result = build_agent(model).invoke({"messages": history})
    return model, result["messages"]


  def test_under_threshold_keeps_history_without_archive(self):
    """49 条时不创建归档，下一次模型输入不变。"""
    history = self.text_history(49)
    model, _ = self.run_agent(history)

    self.assertEqual(model.calls[0][1:], history)
    self.assertFalse(self.transcripts.exists())


  def test_at_threshold_archives_and_replaces_state(self):
    """50 条时归档可还原，只保留头尾和一条归档提示。"""
    history = self.text_history(50)
    model, compacted = self.run_agent(history)

    saved, = self.transcripts.glob("*.jsonl")
    lines = saved.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    self.assertEqual(messages_from_dict(records), history)
    visible = model.calls[0][1:]
    self.assertEqual(len(visible), 30)
    self.assertEqual(visible[:3], history[:3])
    self.assertEqual(visible[4:], history[-26:])
    self.assertIn(str(saved.relative_to(self.workdir)), visible[3].text)
    self.assertEqual(compacted[:-1], visible)

    next_turn, _ = self.run_agent([*compacted, HumanMessage(content="继续")])
    self.assertEqual(next_turn.calls[0][1:-1], compacted)
    self.assertEqual(list(self.transcripts.glob("*.jsonl")), [saved])


  def test_both_boundaries_keep_whole_tool_batches(self):
    """头尾切点都在工具结果中时，保留完整的调用和结果。"""
    history = [
      HumanMessage(content="开始"), *self.completed_batch("head", 4),
      *self.text_history(16), *self.completed_batch("tail", 5),
      *self.text_history(22),
    ]
    model, _ = self.run_agent(history)

    visible = model.calls[0][1:]
    self.assertEqual(visible[:6], history[:6])
    self.assertEqual(visible[7:], history[22:])


  def test_current_request_survives_two_compactions(self):
    """当前请求进入中段后，两次裁剪都把原文留在归档提示里。"""
    request = "当前请求：继续检查这批文件"
    history = [*self.text_history(20), HumanMessage(content=request)]
    model, compacted = self.run_agent(
      history, self.batch("first", 28), self.batch("second", 25),
    )

    self.assertEqual(len(model.calls), 3)
    self.assertEqual(len(list(self.transcripts.glob("*.jsonl"))), 2)
    for visible in model.calls[1:]:
      self.assertEqual(sum(request in message.text for message in visible), 1)
      self.assertIsInstance(visible[-1], ToolMessage)
    self.assertEqual(compacted[:-1], model.calls[-1][1:])


  def test_archives_preview_after_tool_result_budget(self):
    """达到阈值时，归档保留第一步生成的预览和路径。"""
    original = "甲" * 200_001
    (self.workdir / "large.txt").write_text(original, encoding="utf-8")
    read = AIMessage(content="", tool_calls=[{
      "name": "read_file", "args": {"path": "large.txt"}, "id": "read-large",
    }])
    model, _ = self.run_agent(self.text_history(48), read)

    transcript, = self.transcripts.glob("*.jsonl")
    outputs = self.workdir / ".task_outputs" / "tool-results"
    saved, = outputs.glob("*.txt")
    self.assertEqual(saved.read_text(encoding="utf-8"), original)
    lines = transcript.read_text(encoding="utf-8").splitlines()
    archived = json.loads(lines[-1])["data"]
    self.assertEqual(archived["tool_call_id"], "read-large")
    self.assertIn(str(saved.relative_to(self.workdir)), archived["content"])
    self.assertEqual(archived["content"], model.calls[-1][-1].content)


  def test_subagent_keeps_history_without_exposing_it_to_parent(self):
    """子任务达到阈值不归档，父 Agent 仍只收到最终文本。"""
    delegate = AIMessage(content="", tool_calls=[{
      "name": "run_subagent", "args": {"prompt": "子任务"}, "id": "delegate",
    }])
    model, _ = self.run_agent(
      [HumanMessage(content="父任务")], delegate,
      *(self.batch(f"child-{i}", 8) for i in range(6)),
      AIMessage(content="子任务完成"),
    )

    self.assertFalse(self.transcripts.exists())
    visible = model.calls[-2][1:]
    self.assertGreaterEqual(len(visible), 50)
    parent_result = model.calls[-1][-1]
    self.assertEqual(parent_result.tool_call_id, "delegate")
    self.assertEqual(parent_result.content, "子任务完成")
