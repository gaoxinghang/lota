"""使用离线模型验证 s09 召回和 CLI 接入。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from fake_model import ToolCallingModel
from lota import workspace
from lota.agent import build_agent
from lota.main import main
from lota.middleware.memory import MemoryRecord, MemoryStore, recall_memories


class MemoryRecallTest(unittest.TestCase):
  """只在临时工作区保存记忆，不使用模型 API。"""

  def setUp(self):
    """为每项测试创建独立的工作区。"""
    self.workdir = Path(self.enterContext(TemporaryDirectory())).resolve()
    self.store = MemoryStore(self.workdir)


  def save(self, name: str, description: str, body: str) -> None:
    """保存一条用户偏好。

    Args:
        name: 记忆名称。
        description: 目录中的简介。
        body: 仅在召回后读取的正文。
    """
    self.store.save(MemoryRecord(
      name=name, description=description, type="user", body=body,
    ))


  def test_no_index_skips_selector(self):
    """空工作区不调用模型，也不创建记忆目录。"""
    model = ToolCallingModel(responses=[])
    self.assertEqual(recall_memories(model, self.store, "缩进"), "")
    self.assertEqual(model.calls, [])
    self.assertFalse((self.workdir / ".memory").exists())


  def test_selector_only_sees_catalog_and_reads_selected_file(self):
    """按索引选择，不向筛选模型发送任何正文。"""
    self.save("A_indent", "Python 缩进偏好", "正文：使用两个空格")
    self.save("B_editor", "代码编辑器", "正文：使用终端编辑器")
    model = ToolCallingModel(responses=[AIMessage(content="[0, 0]")])
    with patch.object(self.store, "read", wraps=self.store.read) as read:
      result = recall_memories(model, self.store, "我喜欢什么缩进？")

    read.assert_called_once_with("a_indent.md")
    self.assertEqual(json.loads(result), [
      {"source": "a_indent.md", "content": "正文：使用两个空格"},
    ])
    selector_prompt = model.calls[0][-1].text
    self.assertIn("0: A_indent - Python 缩进偏好", selector_prompt)
    self.assertIn("1: B_editor - 代码编辑器", selector_prompt)
    self.assertNotIn("正文", selector_prompt)


  def test_model_failure_or_invalid_indices_use_keywords(self):
    """解析失败、无效编号或模型不可用时匹配当前中文请求。"""
    self.save("indent", "项目的缩进偏好", "使用两个空格")
    self.save("editor", "编辑器偏好", "使用终端编辑器")
    for output in ("不是 JSON", "[true]", "[2]"):
      with self.subTest(output=output):
        model = ToolCallingModel(responses=[AIMessage(content=output)])
        result = recall_memories(model, self.store, "请告诉我缩进怎么写")
        self.assertEqual(json.loads(result)[0]["source"], "indent.md")

    model = ToolCallingModel(responses=[])
    with patch.object(type(model), "invoke", side_effect=TimeoutError):
      result = recall_memories(model, self.store, "请告诉我缩进怎么写")
    self.assertEqual(json.loads(result)[0]["source"], "indent.md")


  def test_empty_selection_does_not_use_fallback(self):
    """模型明确返回空数组，就不读取正文。"""
    self.save("indent", "缩进偏好", "使用两个空格")
    model = ToolCallingModel(responses=[AIMessage(content="[]")])
    with patch.object(self.store, "read") as read:
      self.assertEqual(recall_memories(model, self.store, "缩进"), "")
    read.assert_not_called()


  def test_body_budget(self):
    """按选择顺序加载，正文合计不超过两万字符。"""
    for i in range(6):
      self.save(f"record_{i}", f"简介 {i}", "甲" * 4_100)
    model = ToolCallingModel(responses=[
      AIMessage(content="[5, 4, 3, 2, 1, 0]"),
    ])
    result = json.loads(recall_memories(model, self.store, "查找记录"))
    self.assertEqual(
      [item["source"] for item in result],
      [f"record_{i}.md" for i in range(5, 0, -1)],
    )
    self.assertEqual(sum(len(item["content"]) for item in result), 20_000)


  def test_at_most_five_records_are_loaded(self):
    """即使正文很短，最多也只加载五条记忆。"""
    for i in range(6):
      self.save(f"record_{i}", f"简介 {i}", "短正文")
    model = ToolCallingModel(responses=[
      AIMessage(content="[5, 4, 3, 2, 1, 0]"),
    ])
    with patch.object(self.store, "read", wraps=self.store.read) as read:
      result = json.loads(recall_memories(model, self.store, "查找记录"))
    self.assertEqual(len(result), 5)
    self.assertEqual(read.call_count, 5)


  def test_missing_record_is_skipped(self):
    """索引指向已删除文件时，其余选中记录仍可召回。"""
    self.save("a", "第一条", "已删除")
    self.save("b", "第二条", "保留")
    (self.workdir / ".memory" / "a.md").unlink()
    model = ToolCallingModel(responses=[AIMessage(content="[0, 1]")])
    result = json.loads(recall_memories(model, self.store, "查找记录"))
    self.assertEqual(result, [{"source": "b.md", "content": "保留"}])


  def test_agent_recalls_before_compaction_without_exposing_state(self):
    """直接调用 Agent 时先召回，摘要和返回状态都不带记忆。"""
    self.save("indent", "缩进偏好", "只记在背景：使用两个空格")
    model = ToolCallingModel(responses=[
      AIMessage(content="[0]"),
      AIMessage(content="历史摘要"),
      AIMessage(content="本轮完成"),
    ])
    with patch.object(workspace, "WORKDIR", self.workdir):
      agent = build_agent(model)
      result = agent.invoke({"messages": [
        HumanMessage(content="旧问题"),
        AIMessage(content="甲" * 60_000),
        HumanMessage(content="看看缩进"),
      ]})

    self.assertEqual(len(model.calls), 3)
    self.assertIn("当前请求：\n看看缩进", model.calls[0][-1].text)
    self.assertNotIn("旧问题", model.calls[0][-1].text)
    self.assertIn("事实摘要", model.calls[1][0].text)
    self.assertNotIn("只记在背景", "\n".join(m.text for m in model.calls[1]))
    self.assertIn("只记在背景", model.calls[2][0].text)
    self.assertNotIn("recalled_memories", result)
    self.assertNotIn("只记在背景", "\n".join(m.text for m in result["messages"]))


  def test_cli_recalls_once_per_turn_and_keeps_it_out_of_history(self):
    """本轮父 Agent 每次都看到背景，子 Agent 和下轮看不到。"""
    self.save("indent", "缩进偏好", "只记在背景：使用两个空格")
    model = ToolCallingModel(responses=[
      AIMessage(content="[0]"),
      AIMessage(content="", tool_calls=[{
        "name": "run_subagent", "args": {"prompt": "委派任务"}, "id": "sub",
      }]),
      AIMessage(content="子任务完成"),
      AIMessage(content="", tool_calls=[{
        "name": "glob", "args": {"pattern": "*.missing"}, "id": "glob",
      }]),
      AIMessage(content="首轮完成"),
      AIMessage(content="[]"),
      AIMessage(content="次轮完成"),
    ])
    with (
      patch.object(workspace, "WORKDIR", self.workdir),
      patch("lota.main.build_model", return_value=model),
      patch("builtins.input", side_effect=["看看缩进", "另一题", "q"]),
      patch("builtins.print"),
      patch("lota.main.speak"),
    ):
      main()

    self.assertEqual(len(model.calls), 7)
    for selector in (0, 5):
      self.assertIn("只返回编号", model.calls[selector][0].text)
    for parent in (1, 3, 4):
      system = model.calls[parent][0]
      self.assertIsInstance(system, SystemMessage)
      self.assertIn("可用工具", system.text)
      self.assertIn("只记在背景", system.text)
      self.assertIn("以当前请求为准", system.text)
    self.assertIn("你是子 Agent", model.calls[2][0].text)
    for step in (2, 6):
      self.assertNotIn("只记在背景", "\n".join(m.text for m in model.calls[step]))
    self.assertNotIn("看看缩进", model.calls[5][-1].text)
    self.assertEqual(
      [m.text for m in model.calls[6][-2:]], ["首轮完成", "另一题"],
    )
