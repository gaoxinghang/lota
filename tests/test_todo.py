"""验证 TODO 清单更新和跨轮状态。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from fake_model import ToolCallingModel
from lota import workspace
from lota.agent import build_agent
from lota.tools.todo import TodoManager


class TodoTest(unittest.TestCase):
  """验证清单校验和会话内状态。"""

  def setUp(self):
    """隔离工作区，不打印工具日志或进入审批。"""
    workdir = Path(self.enterContext(TemporaryDirectory())).resolve()
    self.enterContext(patch.object(workspace, "WORKDIR", workdir))
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


  def test_invalid_update_returns_error(self):
    """清单更新失败时将错误回传给模型。"""
    model = ToolCallingModel(responses=[
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
    self.assertEqual(
      error.text, "Error: Only one todo can be in_progress at a time"
    )


  def test_new_turn_keeps_list(self):
    """输入新请求后，原 Agent 的清单仍在内存中。"""
    model = ToolCallingModel(responses=[
      AIMessage(content="", tool_calls=[{
        "name": "todo_write", "id": "plan", "args": {
          "todos": [{"content": "inspect files", "status": "in_progress"}],
        },
      }]),
      AIMessage(content="done"),
      AIMessage(content="next reply"),
    ])
    manager = TodoManager()
    with patch("lota.tools.todo.TodoManager", return_value=manager):
      agent = build_agent(model)
    first = agent.invoke({"messages": [HumanMessage(content="first")]})
    agent.invoke({
      "messages": [*first["messages"], HumanMessage(content="next")],
    })
    self.assertEqual(manager.render(), "[>] inspect files\n\n(0/1 completed)")
