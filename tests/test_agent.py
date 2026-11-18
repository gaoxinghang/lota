"""使用真实 Agent 和预设模型回复验证分发，不请求外部 API。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import (
  AIMessage, HumanMessage, SystemMessage, ToolMessage,
)

from fake_model import ToolCallingModel
from lota import workspace
from lota.agent import build_agent
from lota.main import main


class AgentTest(unittest.TestCase):
  """验证停止条件、串行执行、错误回传和跨轮历史。"""

  def setUp(self):
    """创建隔离工作区，禁止意外请求用户输入。"""
    temporary = self.enterContext(TemporaryDirectory())
    self.workspace = Path(temporary).resolve()
    self.enterContext(patch.object(workspace, "WORKDIR", self.workspace))
    self.enterContext(patch("builtins.print"))
    self.enterContext(patch(
      "builtins.input", side_effect=AssertionError("Unexpected approval")
    ))


  def run_agent(self, model, history):
    """运行真实 Agent 并返回完整历史。

    Args:
        model: 使用预设回复的测试模型。
        history: 传给 Agent 的消息列表。

    Returns:
        包含模型回复和工具结果的消息列表。
    """
    result = build_agent(model).invoke({"messages": history})
    return result["messages"]


  def test_registered_tools(self):
    """六个工具通过同一列表交给 Agent。"""
    model = ToolCallingModel(responses=[AIMessage(content="done")])
    with patch("lota.agent.create_agent") as create_agent:
      build_agent(model)
    options = create_agent.call_args.kwargs
    names = [item.name for item in options["tools"]]
    self.assertEqual(
      names,
      ["bash", "read_file", "write_file", "edit_file", "glob", "todo_write"],
    )
    for name in names:
      self.assertIn(f"`{name}`", options["system_prompt"])


  def test_no_tool_calls(self):
    """没有工具调用时直接结束，系统提示仅发给模型，不进入历史。"""
    model = ToolCallingModel(responses=[AIMessage(content="done")])
    history = [HumanMessage(content="hi")]
    messages = self.run_agent(model, history)
    self.assertEqual(messages[:-1], history)
    self.assertEqual(messages[-1].content, "done")
    self.assertEqual(len(model.calls), 1)
    system, *conversation = model.calls[0]
    self.assertIsInstance(system, SystemMessage)
    self.assertIn("复杂任务", system.text)
    self.assertEqual(conversation, history)


  def test_dispatch_order_and_result_ids(self):
    """同批写入和读取按顺序执行，并用调用 ID 配对结果。"""
    request = AIMessage(content="", tool_calls=[
      {
        "name": "write_file",
        "args": {"path": "demo.txt", "content": "hello"},
        "id": "write-1",
      },
      {
        "name": "read_file", "args": {"path": "demo.txt"}, "id": "read-1",
      },
    ])
    model = ToolCallingModel(responses=[request, AIMessage(content="done")])
    messages = self.run_agent(model, [HumanMessage(content="write then read")])
    results = [item for item in messages if isinstance(item, ToolMessage)]
    self.assertEqual(
      [(item.name, item.tool_call_id, item.content) for item in results],
      [
        ("write_file", "write-1", "Wrote 5 characters to demo.txt"),
        ("read_file", "read-1", "hello"),
      ],
    )
    self.assertEqual(len(model.calls), 2)
    self.assertEqual(model.calls[-1][0], model.calls[0][0])
    self.assertEqual(model.calls[-1][1:], messages[:-1])


  def test_multiple_tool_rounds(self):
    """模型可连续请求多轮工具，直到返回无工具调用的回复。"""
    model = ToolCallingModel(responses=[
      AIMessage(content="", tool_calls=[{
        "name": "write_file",
        "args": {"path": "demo.txt", "content": "hello"},
        "id": "write-1",
      }]),
      AIMessage(content="", tool_calls=[{
        "name": "read_file", "args": {"path": "demo.txt"}, "id": "read-1",
      }]),
      AIMessage(content="done"),
    ])
    messages = self.run_agent(model, [HumanMessage(content="write then read")])
    self.assertEqual(len(model.calls), 3)
    self.assertEqual(model.calls[-1][1:], messages[:-1])
    self.assertEqual(messages[-2].content, "hello")
    self.assertEqual(messages[-1].content, "done")


  def test_bash_os_error_returns_to_model(self):
    """Shell 操作系统错误回传模型，不终止 Agent。"""
    request = AIMessage(content="", tool_calls=[
      {"name": "bash", "args": {"command": "echo hello"}, "id": "bash-1"},
    ])
    model = ToolCallingModel(responses=[request, AIMessage(content="done")])
    with patch(
      "lota.tools.shell.subprocess.run", side_effect=OSError("failed")
    ):
      messages = self.run_agent(model, [])
    result = messages[1]
    self.assertIsInstance(result, ToolMessage)
    self.assertEqual(result.content, "Error: failed")
    self.assertEqual(result.tool_call_id, "bash-1")
    self.assertEqual(result.name, "bash")
    self.assertEqual(len(model.calls), 2)


  def test_tool_errors_return_to_model(self):
    """未知工具、参数错误和文件错误不打断同批后续调用。"""
    request = AIMessage(content="", tool_calls=[
      {"name": "unknown", "args": {}, "id": "unknown-1"},
      {"name": "write_file", "args": {}, "id": "invalid-1"},
      {"name": "read_file", "args": {"path": "missing.txt"}, "id": "read-1"},
      {"name": "glob", "args": {"pattern": "*.missing"}, "id": "glob-1"},
    ])
    model = ToolCallingModel(responses=[
      request, AIMessage(content="try again"),
    ])
    messages = self.run_agent(model, [])
    results = [item for item in messages if isinstance(item, ToolMessage)]
    self.assertEqual(
      [item.tool_call_id for item in results],
      ["unknown-1", "invalid-1", "read-1", "glob-1"],
    )
    self.assertEqual(results[0].content, "Error: Unknown tool: unknown")
    self.assertTrue(all(
      item.content.startswith("Error:") for item in results[:3]
    ))
    self.assertEqual(results[-1].content, "(no matches)")
    self.assertEqual(len(model.calls), 2)


  def test_tool_value_error_returns_to_model(self):
    """工具内部的业务参数错误仍以原来的 Error 格式回传。"""
    request = AIMessage(content="", tool_calls=[
      {
        "name": "read_file", "args": {"path": "demo.txt", "limit": 0},
        "id": "read-1",
      },
    ])
    model = ToolCallingModel(responses=[request, AIMessage(content="done")])
    messages = self.run_agent(model, [])
    self.assertEqual(
      messages[1].content, "Error: limit must be a positive integer"
    )


  def test_unhandled_error_propagates(self):
    """不扩大原有异常捕获范围，意外异常仍向调用方抛出。"""
    request = AIMessage(content="", tool_calls=[
      {"name": "bash", "args": {"command": "echo hello"}, "id": "bash-1"},
    ])
    model = ToolCallingModel(responses=[request])
    with patch(
      "lota.tools.shell.subprocess.run", side_effect=RuntimeError("bug")
    ):
      with self.assertRaisesRegex(RuntimeError, "bug"):
        self.run_agent(model, [])


  def test_main_keeps_history(self):
    """命令行跨轮保留完整历史，空输入不调用模型。"""
    model = ToolCallingModel(responses=[
      AIMessage(content="first reply"), AIMessage(content="second reply"),
    ])
    with (
      patch("lota.main.build_model", return_value=model),
      patch("builtins.input", side_effect=["", "first", "second", "q"]),
    ):
      main()
    self.assertEqual(len(model.calls), 2)
    self.assertIsInstance(model.calls[0][0], SystemMessage)
    self.assertEqual(
      [item.content for item in model.calls[1][1:]],
      ["first", "first reply", "second"],
    )


  def test_main_denied_approval_continues(self):
    """审批拒绝本次调用后，仍执行后续调用并返回主输入。"""
    request = AIMessage(content="", tool_calls=[
      {"name": "bash", "args": {"command": "rm x"}, "id": "denied"},
      {"name": "glob", "args": {"pattern": "*.missing"}, "id": "allowed"},
    ])
    model = ToolCallingModel(responses=[request, AIMessage(content="done")])
    with (
      patch("lota.main.build_model", return_value=model),
      patch("builtins.input", side_effect=["delete x", "n", "q"]),
      patch("lota.tools.shell.subprocess.run") as run,
    ):
      main()
    run.assert_not_called()
    results = [
      item for item in model.calls[-1] if isinstance(item, ToolMessage)
    ]
    self.assertEqual(
      [(item.tool_call_id, item.content) for item in results],
      [("denied", "Permission denied."), ("allowed", "(no matches)")],
    )
