"""使用离线模型验证第四步的归档、摘要和节点顺序。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import (
  AIMessage, HumanMessage, SystemMessage, ToolMessage, messages_from_dict,
)

from fake_model import ToolCallingModel
from lota import workspace
from lota.agent import build_agent


class CompactHistoryTest(unittest.TestCase):
  """检查第四步只在前三步处理之后仍超限时生成摘要。"""

  def setUp(self):
    """使用临时工作区，不在仓库留下归档。"""
    temporary = self.enterContext(TemporaryDirectory())
    self.workdir = Path(temporary).resolve()
    self.transcripts = self.workdir / ".transcripts"
    self.enterContext(patch.object(workspace, "WORKDIR", self.workdir))
    self.enterContext(patch("builtins.print"))


  def test_fourth_step_has_its_own_node(self):
    """确认微压缩、摘要和正常模型调用是连续的独立节点。"""
    model = ToolCallingModel(responses=[AIMessage(content="完成")])
    graph = build_agent(model).get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}
    self.assertIn((
      "micro_compact.before_model", "compact_history.before_model",
    ), edges)
    self.assertIn(("compact_history.before_model", "model"), edges)


  def test_summarizes_oversized_history_and_keeps_current_request(self):
    """完整归档只落盘一次，模型继续工作时只收到一条摘要消息。"""
    request = "继续修改测试，不要提交"
    history = [
      HumanMessage(content="较早的目标"),
      AIMessage(content="甲" * 90_000),
      HumanMessage(content=request),
    ]
    model = ToolCallingModel(responses=[
      AIMessage(content="已定位问题，测试尚未运行。"),
      AIMessage(content="完成"),
    ])
    result = build_agent(model).invoke({"messages": history})

    self.assertEqual(len(model.calls), 2)
    system, summary_input = model.calls[0]
    self.assertIsInstance(system, SystemMessage)
    self.assertIn("不要执行", system.text)
    self.assertIn("较早的目标", summary_input.text)
    self.assertIn(request, summary_input.text)
    self.assertLessEqual(len(summary_input.text), 80_000)

    transcript, = self.transcripts.glob("*.jsonl")
    lines = transcript.read_text(encoding="utf-8").splitlines()
    saved = messages_from_dict([json.loads(line) for line in lines])
    self.assertEqual(saved, history)
    _, marker = model.calls[1]
    self.assertIn(f"当前用户请求：\n{request}", marker.text)
    self.assertIn("已定位问题，测试尚未运行。", marker.text)
    self.assertIn(str(transcript.relative_to(self.workdir)), marker.text)
    self.assertEqual(result["messages"], [marker, model.responses[-1]])


  def test_micro_compaction_can_avoid_summary(self):
    """第三步降到预算内后，第四步不归档、不调用模型生成摘要。"""
    ids = [f"read-{i}" for i in range(4)]
    history = [
      HumanMessage(content="读取文件"),
      AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {"path": "file.txt"}, "id": call_id}
        for call_id in ids
      ]),
      *[
        ToolMessage(content=output, tool_call_id=call_id)
        for output, call_id in zip(["甲" * 55_000, "乙", "丙", "丁"], ids)
      ],
      AIMessage(content="读完"), HumanMessage(content="继续"),
    ]
    model = ToolCallingModel(responses=[AIMessage(content="完成")])
    build_agent(model).invoke({"messages": history})

    self.assertEqual(len(model.calls), 1)
    self.assertFalse(self.transcripts.exists())
    self.assertIn("旧工具结果已保存", model.calls[0][3].text)


  def test_second_summary_reuses_request_from_previous_summary(self):
    """同一轮二次摘要时，当前请求不变，也不会失去工具批次归档。"""
    request = "读取这个文件"
    output = "文" * 60_000
    (self.workdir / "file.txt").write_text(output, encoding="utf-8")
    model = ToolCallingModel(responses=[
      AIMessage(content="待读取文件"),
      AIMessage(content="", tool_calls=[{
        "name": "read_file", "args": {"path": "file.txt"}, "id": "read-file",
      }]),
      AIMessage(content="已读取文件"),
      AIMessage(content="完成"),
    ])
    result = build_agent(model).invoke({"messages": [
      HumanMessage(content="先前的任务"), AIMessage(content="甲" * 55_000),
      HumanMessage(content=request),
    ]})

    self.assertEqual(len(model.calls), 4)
    marker = result["messages"][0]
    self.assertIn(f"当前用户请求：\n{request}", marker.text)
    self.assertIn("已读取文件", marker.text)
    self.assertEqual(len(result["messages"]), 2)
    files = list(self.transcripts.glob("*.jsonl"))
    self.assertEqual(len(files), 2)
    self.assertTrue(any(output in p.read_text(encoding="utf-8") for p in files))


  def test_child_uses_its_own_summary(self):
    """子 Agent 可独立归档，父 Agent 只收到任务结论。"""
    (self.workdir / "file.txt").write_text("子" * 55_000, encoding="utf-8")
    model = ToolCallingModel(responses=[
      AIMessage(content="", tool_calls=[{
        "name": "run_subagent", "args": {"prompt": "读取文件"}, "id": "child",
      }]),
      AIMessage(content="", tool_calls=[{
        "name": "read_file", "args": {"path": "file.txt"}, "id": "read",
      }]),
      AIMessage(content="已读取文件"), AIMessage(content="子任务完成"),
      AIMessage(content="父任务完成"),
    ])
    build_agent(model).invoke({"messages": [HumanMessage(content="父任务")]})

    self.assertEqual(len(model.calls), 5)
    self.assertIn("当前用户请求：\n读取文件", model.calls[3][-1].text)
    self.assertEqual(model.calls[4][-1].text, "子任务完成")
    self.assertEqual(len(list(self.transcripts.glob("*.jsonl"))), 1)


  def test_empty_summary_keeps_messages(self):
    """摘要没有正文时，不用空摘要覆盖原有历史。"""
    history = [HumanMessage(content="开始"), AIMessage(content="甲" * 55_000)]
    model = ToolCallingModel(responses=[
      AIMessage(content=""), AIMessage(content="完成"),
    ])
    result = build_agent(model).invoke({"messages": history})

    self.assertEqual(model.calls[-1][1:], history)
    self.assertEqual(result["messages"][:-1], history)
