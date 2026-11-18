"""使用离线模型验证清单更新、提醒和跨轮状态。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from fake_model import ToolCallingModel
from lota import workspace
from lota.agent import build_agent
from lota.middleware.todo_reminder import REMINDER
from lota.tools.todo import TodoManager


def tool_batch(round_id: str, size: int = 1) -> AIMessage:
  """生成一批在工作区查找文件的工具请求。

  Args:
      round_id: 批次 ID，用于生成各工具调用的 ID。
      size: 批次中的工具数。

  Returns:
      携带工具请求的模型消息。
  """
  return AIMessage(content="", tool_calls=[
    {"name": "glob", "args": {"pattern": "*.missing"}, "id": f"{round_id}-{i}"}
    for i in range(size)
  ])


class TodoTest(unittest.TestCase):
  """验证清单和按批生成的模型提醒。"""

  def setUp(self):
    """使用隔离工作区，不打印工具日志或进入审批。"""
    temporary = self.enterContext(TemporaryDirectory())
    self.enterContext(patch.object(
      workspace, "WORKDIR", Path(temporary).resolve()
    ))
    self.enterContext(patch("builtins.print"))
    self.enterContext(patch(
      "builtins.input", side_effect=AssertionError("Unexpected approval")
    ))


  def test_update_replaces_and_renders(self):
    """全量替换后按状态显示进度，空列表清空。"""
    manager = TodoManager()
    write_todos = tool(manager.todo_write)
    self.assertEqual(write_todos.invoke({"todos": [
      {"content": " read ", "status": "completed"},
      {"content": "edit", "status": "in_progress"},
      {"content": "test", "status": "pending"},
    ]}), "[x] read\n[>] edit\n[ ] test\n\n(1/3 completed)")
    self.assertEqual(write_todos.invoke({"todos": [
      {"content": "test", "status": "completed"},
    ]}), "[x] test\n\n(1/1 completed)")
    self.assertEqual(write_todos.invoke({"todos": []}), "No todos.")


  def test_invalid_update_keeps_old_list(self):
    """无效更新不会覆盖旧清单。"""
    manager = TodoManager()
    write_todos = tool(manager.todo_write)
    old = write_todos.invoke({
      "todos": [{"content": "keep", "status": "pending"}],
    })
    cases = [
      [{"content": " ", "status": "pending"}],
      [{"content": "read", "status": "unknown"}],
      [{"content": "read", "status": "pending"}] * 21,
      [{"content": "read", "status": "in_progress"}] * 2,
      "[]",
    ]
    for todos in cases:
      with self.subTest(todos=todos), self.assertRaises(ValueError):
        write_todos.invoke({"todos": todos})
      self.assertEqual(manager.render(), old)


  def test_reminder_counts_batches_and_does_not_enter_history(self):
    """第三、六批才提醒，且不改结果 ID 或历史。"""
    model = ToolCallingModel(responses=[
      tool_batch(str(i), size=2) for i in range(1, 7)
    ] + [AIMessage(content="done")])
    result = build_agent(model).invoke({
      "messages": [HumanMessage(content="inspect files")],
    })
    for index, call in enumerate(model.calls):
      reminders = [item for item in call if REMINDER in item.text]
      self.assertEqual(len(reminders), int(index in {3, 6}))
      if reminders:
        self.assertIs(reminders[0], call[-1])
        self.assertIsInstance(call[-1], ToolMessage)
        self.assertEqual(call[-1].tool_call_id, f"{index}-1")
        self.assertEqual(call[-1].text, f"(no matches)\n\n{REMINDER}")
    self.assertTrue(all(
      REMINDER not in item.text for item in result["messages"]
    ))


  def test_success_resets_count_even_with_other_calls_in_batch(self):
    """混合批次中的成功更新可以重置计数。"""
    update = AIMessage(content="", tool_calls=[
      {
        "name": "todo_write", "id": "plan", "args": {
          "todos": [{"content": "inspect files", "status": "pending"}],
        },
      },
      *tool_batch("3").tool_calls,
    ])
    model = ToolCallingModel(responses=[
      tool_batch("1"), tool_batch("2"), update,
      tool_batch("4"), tool_batch("5"), tool_batch("6"),
      AIMessage(content="done"),
    ])
    result = build_agent(model).invoke({
      "messages": [HumanMessage(content="inspect files")],
    })
    plan = next(
      item for item in result["messages"]
      if isinstance(item, ToolMessage) and item.tool_call_id == "plan"
    )
    self.assertEqual(plan.text, "[ ] inspect files\n\n(0/1 completed)")
    self.assertEqual(
      [i for i, call in enumerate(model.calls) if REMINDER in call[-1].text],
      [6],
    )


  def test_failed_update_does_not_reset_count(self):
    """校验失败不重置计数，错误与提醒可并存。"""
    model = ToolCallingModel(responses=[
      tool_batch("1"), tool_batch("2"),
      AIMessage(content="", tool_calls=[{
        "name": "todo_write", "id": "invalid", "args": {
          "todos": [{"content": "read", "status": "in_progress"}] * 2,
        },
      }]),
      AIMessage(content="done"),
    ])
    build_agent(model).invoke({"messages": [HumanMessage(content="inspect")]})
    error = model.calls[-1][-1]
    self.assertEqual(error.status, "error")
    self.assertEqual(error.tool_call_id, "invalid")
    self.assertTrue(error.text.startswith("Error: Only one todo"))
    self.assertTrue(error.text.endswith(REMINDER))


  def test_new_turn_resets_count_but_keeps_list(self):
    """输入新请求后重新计数，原 Agent 的清单仍在内存中。"""
    model = ToolCallingModel(responses=[
      AIMessage(content="", tool_calls=[{
        "name": "todo_write", "id": "plan", "args": {
          "todos": [{"content": "inspect files", "status": "in_progress"}],
        },
      }]),
      tool_batch("1"), tool_batch("2"), AIMessage(content="done"),
      tool_batch("3"), tool_batch("4"), tool_batch("5"),
      AIMessage(content="done"),
    ])
    manager = TodoManager()
    with patch("lota.tools.todo.TodoManager", return_value=manager):
      agent = build_agent(model)
    first = agent.invoke({"messages": [HumanMessage(content="first")]})
    agent.invoke({
      "messages": [*first["messages"], HumanMessage(content="next")],
    })
    self.assertEqual(manager.render(), "[>] inspect files\n\n(0/1 completed)")
    self.assertEqual(
      [i for i, call in enumerate(model.calls) if REMINDER in call[-1].text],
      [7],
    )
