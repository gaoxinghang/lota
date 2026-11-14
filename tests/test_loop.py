"""使用模拟模型验证工具分发，不请求外部 API。"""

import unittest
from unittest.mock import Mock, call, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lota.loop import TOOL_MAP, agent_loop


class AgentLoopTest(unittest.TestCase):
  """验证循环停止条件、调用顺序与错误回传。"""

  def setUp(self):
    """隐藏测试中的终端预览。"""
    self.enterContext(patch("builtins.print"))


  def test_registered_tools(self):
    """五个工具从注册表自动映射到同名处理器。"""
    self.assertEqual(
      set(TOOL_MAP),
      {"bash", "read_file", "write_file", "edit_file", "glob"},
    )
    for name, handler in TOOL_MAP.items():
      self.assertEqual(name, handler.name)


  def test_no_tool_calls(self):
    """模型未请求工具时直接返回，并保留消息历史。"""
    reply = AIMessage(content="done")
    model = Mock()
    model.invoke.return_value = reply
    history = [HumanMessage(content="hello")]
    self.assertIs(agent_loop(model, history), reply)
    self.assertEqual(history[-1], reply)
    model.invoke.assert_called_once_with(history)


  def test_dispatch_order_and_result_ids(self):
    """按请求顺序执行非 Bash 工具，并用调用 ID 配对结果。"""
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
    final = AIMessage(content="done")
    handlers = Mock()
    handlers.writer.invoke.return_value = "written"
    handlers.reader.invoke.return_value = "hello"
    model = Mock()
    model.invoke.side_effect = [request, final]
    history = [HumanMessage(content="write then read")]

    with patch.dict(
      TOOL_MAP, write_file=handlers.writer, read_file=handlers.reader
    ):
      self.assertIs(agent_loop(model, history), final)

    self.assertEqual(handlers.mock_calls, [
      call.writer.invoke({"path": "demo.txt", "content": "hello"}),
      call.reader.invoke({"path": "demo.txt"}),
    ])
    results = [item for item in history if isinstance(item, ToolMessage)]
    self.assertEqual(
      [(item.name, item.tool_call_id, item.content) for item in results],
      [("write_file", "write-1", "written"), ("read_file", "read-1", "hello")],
    )
    self.assertEqual(model.invoke.call_count, 2)


  def test_tool_errors_return_to_model(self):
    """未知工具、参数错误与文件错误不打断后续工具调用。"""
    reader = Mock()
    reader.invoke.side_effect = FileNotFoundError("missing.txt")
    request = AIMessage(content="", tool_calls=[
      {"name": "unknown", "args": {}, "id": "unknown-1"},
      {"name": "write_file", "args": {}, "id": "invalid-1"},
      {"name": "read_file", "args": {"path": "missing.txt"}, "id": "read-1"},
    ])
    model = Mock()
    model.invoke.side_effect = [request, AIMessage(content="try again")]
    history = []

    with patch.dict(TOOL_MAP, read_file=reader):
      agent_loop(model, history)

    results = [item for item in history if isinstance(item, ToolMessage)]
    self.assertEqual(
      [item.tool_call_id for item in results],
      ["unknown-1", "invalid-1", "read-1"],
    )
    self.assertTrue(all(item.content.startswith("Error:") for item in results))
    reader.invoke.assert_called_once_with({"path": "missing.txt"})
    self.assertEqual(model.invoke.call_count, 2)
