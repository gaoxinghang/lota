"""使用离线模型验证轮末记忆提取、落盘与失败处理。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from fake_model import ToolCallingModel
from lota import workspace
from lota.agent import build_agent
from lota.middleware.memory import MemoryStore

QUERY = "以后 Python 都用两个空格缩进。"
CANDIDATE = {
  "name": "python_indent",
  "description": "Python 缩进偏好",
  "type": "user",
  "scope": "persistent",
  "body": "Python 代码使用两个空格缩进。",
  "evidence": QUERY,
}


class MemoryExtractionTest(unittest.TestCase):
  """在临时工作区执行父 Agent，不调用真实模型。"""

  def setUp(self):
    """为本测试隔离工作区与记忆目录。"""
    self.workdir = Path(self.enterContext(TemporaryDirectory())).resolve()
    self.enterContext(patch.object(workspace, "WORKDIR", self.workdir))
    self.store = MemoryStore(self.workdir)


  def test_persistent_preference_is_recalled_by_a_new_agent(self):
    """轮末保存记录，新 Agent 能从磁盘召回而不依赖旧历史。"""
    model = ToolCallingModel(
      responses=[AIMessage(content="好的。")],
      extraction_reply=json.dumps([CANDIDATE]),
    )
    result = build_agent(model).invoke({"messages": [
      HumanMessage(content=QUERY),
    ]})
    self.assertEqual(len(model.extraction_calls), 1)
    self.assertEqual(result["messages"][-1].text, "好的。")
    self.assertNotIn("turn_query", result)
    self.assertEqual(
      self.store.read("python_indent.md").body, CANDIDATE["body"],
    )

    new_model = ToolCallingModel(responses=[
      AIMessage(content="[0]"), AIMessage(content="两个空格。"),
    ])
    build_agent(new_model).invoke({"messages": [
      HumanMessage(content="我喜欢什么缩进？"),
    ]})
    self.assertIn(CANDIDATE["body"], new_model.calls[1][0].text)


  def test_extraction_excludes_history_summary_and_tools(self):
    """长历史被压缩后，仅发本轮原始输入和最终回复给提取模型。"""
    model = ToolCallingModel(responses=[
      AIMessage(content="历史摘要"),
      AIMessage(content="", tool_calls=[{
        "name": "run_subagent", "args": {"prompt": "检查另一个任务"},
        "id": "sub",
      }]),
      AIMessage(content="子任务输出"),
      AIMessage(content="本轮最终答复"),
    ])
    with patch("builtins.print"):
      build_agent(model).invoke({"messages": [
        HumanMessage(content="旧问题"),
        AIMessage(content="甲" * 60_000),
        HumanMessage(content=QUERY),
      ]})
    self.assertEqual(len(model.extraction_calls), 1)
    self.assertEqual(json.loads(model.extraction_calls[0][-1].text), {
      "user": QUERY, "assistant": "本轮最终答复",
    })


  def test_temporary_and_sensitive_input_are_not_extracted(self):
    """已知临时要求或常见凭据不发送给辅助模型，也不落盘。"""
    for query in ("这次用两个空格。", "密码是 example123", "不要记住这条"):
      with self.subTest(query=query):
        model = ToolCallingModel(
          responses=[AIMessage(content="好的。")],
          extraction_reply=json.dumps([CANDIDATE | {"evidence": query}]),
        )
        build_agent(model).invoke({"messages": [HumanMessage(content=query)]})
        self.assertEqual(model.extraction_calls, [])
    self.assertFalse((self.workdir / ".memory").exists())


  def test_invalid_candidates_are_not_saved(self):
    """候选需有用户原文支持，且符合持久范围和字段规则。"""
    cases = [
      {"scope": "current_task"},
      {"evidence": "只在助手回复里"},
      {"name": "invalid name"},
      {"name": "memory"},
      {"body": "API_KEY=example123"},
    ]
    model = ToolCallingModel(
      responses=[AIMessage(content="只在助手回复里")],
      extraction_reply=json.dumps([CANDIDATE | fields for fields in cases]),
    )
    build_agent(model).invoke({"messages": [HumanMessage(content=QUERY)]})
    self.assertEqual(len(model.extraction_calls), 1)
    self.assertEqual(self.store.catalog(), [])


  def test_duplicate_records_do_not_create_more_files(self):
    """跨轮重复和同轮重复的正文都只保留一份。"""
    first = ToolCallingModel(
      responses=[AIMessage(content="好的。")],
      extraction_reply=json.dumps([CANDIDATE, CANDIDATE | {"name": "copy"}]),
    )
    build_agent(first).invoke({"messages": [HumanMessage(content=QUERY)]})
    second = ToolCallingModel(
      responses=[AIMessage(content="[]"), AIMessage(content="知道了。")],
      extraction_reply=json.dumps([CANDIDATE | {
        "name": "other", "description": "另一个简介",
      }]),
    )
    build_agent(second).invoke({"messages": [HumanMessage(content=QUERY)]})
    self.assertEqual([filename for filename, _ in self.store.catalog()], [
      "python_indent.md",
    ])


  def test_bad_json_or_storage_error_does_not_hide_reply(self):
    """记忆处理失败时照常返回父 Agent 的答复。"""
    for extraction_reply in ("不是 JSON", json.dumps([CANDIDATE])):
      with self.subTest(extraction_reply=extraction_reply):
        model = ToolCallingModel(
          responses=[AIMessage(content="照常回答")],
          extraction_reply=extraction_reply,
        )
        with (
          patch.object(MemoryStore, "save", side_effect=OSError),
          self.assertLogs("lota.middleware.memory", level="WARNING"),
        ):
          result = build_agent(model).invoke({"messages": [
            HumanMessage(content=QUERY),
          ]})
        self.assertEqual(result["messages"][-1].text, "照常回答")
        self.assertEqual(self.store.catalog(), [])
