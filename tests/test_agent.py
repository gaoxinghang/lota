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
from lota.middleware.permission import permission_middleware


class AgentTest(unittest.TestCase):
  """验证停止条件、串行执行、错误回传和跨轮历史。"""

  def setUp(self):
    """创建隔离工作区，禁止意外请求用户输入。"""
    temporary = self.enterContext(TemporaryDirectory())
    self.workspace = Path(temporary).resolve()
    self.enterContext(patch.object(workspace, "WORKDIR", self.workspace))
    self.print_log = self.enterContext(patch("builtins.print"))
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
    """子 Agent 只注册基础工具和权限中间件。"""
    model = ToolCallingModel(responses=[AIMessage(content="done")])
    with patch("lota.agent.create_agent") as create_agent:
      build_agent(model)
    child_call, parent_call = create_agent.call_args_list
    child_tools = [item.name for item in child_call.kwargs["tools"]]
    parent_tools = [item.name for item in parent_call.kwargs["tools"]]
    self.assertEqual(
      child_tools, ["bash", "read_file", "write_file", "edit_file", "glob"],
    )
    self.assertEqual(child_call.kwargs["middleware"], [permission_middleware])
    self.assertEqual(parent_tools, [*child_tools, "todo_write", "run_subagent"])
    for name in parent_tools:
      self.assertIn(f"`{name}`", parent_call.kwargs["system_prompt"])


  def write_review_skill(self) -> str:
    """写入供集成测试使用的代码审查技能并返回全文。"""
    manifest = self.workspace / "skills" / "review" / "SKILL.md"
    manifest.parent.mkdir(parents=True)
    content = "---\nname: review\ndescription: 审查代码\n---\n\n# 审查正文\n"
    manifest.write_text(content, encoding="utf-8")
    return content


  def test_skill_catalog_and_results(self):
    """首轮只发目录，工具加载正文，路径形状的名称按未知技能处理。"""
    content = self.write_review_skill()
    request = AIMessage(content="", tool_calls=[
      {
        "name": "load_skill",
        "args": {"name": "review"},
        "id": "skill-1",
      },
      {
        "name": "load_skill",
        "args": {"name": "skills/review/SKILL.md"},
        "id": "skill-2",
      },
    ])
    model = ToolCallingModel(responses=[request, AIMessage(content="done")])
    messages = self.run_agent(model, [HumanMessage(content="审查")])

    system = model.calls[0][0]
    self.assertIn("- review: 审查代码", system.text)
    self.assertIn("`load_skill`", system.text)
    self.assertNotIn("# 审查正文", system.text)

    loaded, unknown = [
      item for item in messages if isinstance(item, ToolMessage)
    ]
    self.assertEqual((loaded.tool_call_id, loaded.content), ("skill-1", content))
    self.assertIn(loaded, model.calls[1])
    self.assertEqual((unknown.tool_call_id, unknown.status), ("skill-2", "error"))
    self.assertIn("Unknown skill", unknown.text)
    self.assertIn("Available: review", unknown.text)
    self.assertNotIn("# 审查正文", unknown.text)


  def test_subagent_does_not_receive_skill_catalog(self):
    """只有父 Agent 收到技能目录，子 Agent 只收到委派消息。"""
    self.write_review_skill()
    model = ToolCallingModel(responses=[
      AIMessage(content="", tool_calls=[{
        "name": "run_subagent",
        "args": {"prompt": "审查代码"},
        "id": "sub-1",
      }]),
      AIMessage(content="子任务结论"),
      AIMessage(content="done"),
    ])
    messages = self.run_agent(model, [HumanMessage(content="委派审查")])

    system, prompt = model.calls[1]
    self.assertIn("- review: 审查代码", model.calls[0][0].text)
    self.assertNotIn("- review: 审查代码", system.text)
    self.assertNotIn("load_skill", system.text)
    self.assertIsInstance(prompt, HumanMessage)
    self.assertEqual(prompt.content, "审查代码")
    results = [item for item in messages if isinstance(item, ToolMessage)]
    self.assertEqual(
      [(item.name, item.tool_call_id, item.content) for item in results],
      [("run_subagent", "sub-1", "子任务结论")],
    )


  def test_subagent_isolates_messages_and_checks_permission(self):
    """子 Agent 可写入工作区，越界读取仍审批，只向父 Agent 返回结论。"""
    outside = Path(self.enterContext(TemporaryDirectory())).resolve()
    target = outside / "demo.txt"
    target.write_text("outside", encoding="utf-8")
    model = ToolCallingModel(responses=[
      AIMessage(content="", tool_calls=[{
        "name": "run_subagent",
        "args": {"prompt": "调查文件"},
        "id": "sub-1",
      }]),
      AIMessage(content="", tool_calls=[
        {
          "name": "write_file",
          "args": {"path": "sub.txt", "content": "子任务"},
          "id": "write-1",
        },
        {
          "name": "read_file",
          "args": {"path": str(target)},
          "id": "read-1",
        },
      ]),
      AIMessage(content="无法读取"),
      AIMessage(content="父 Agent 结束"),
    ])
    with patch("builtins.input", return_value="n") as approve:
      messages = self.run_agent(model, [HumanMessage(content="父任务")])

    approve.assert_called_once()
    self.assertIn("read_file", approve.call_args.args[0])
    self.assertIn(str(target), approve.call_args.args[0])
    self.assertEqual(
      (self.workspace / "sub.txt").read_text(encoding="utf-8"), "子任务"
    )
    child_prompt, = model.calls[1][1:]
    self.assertIsInstance(child_prompt, HumanMessage)
    self.assertEqual(child_prompt.content, "调查文件")
    self.assertIsInstance(model.calls[2][-1], ToolMessage)
    self.assertEqual(model.calls[2][-1].content, "Permission denied.")
    results = [item for item in messages if isinstance(item, ToolMessage)]
    self.assertEqual(
      [(item.name, item.tool_call_id, item.content) for item in results],
      [("run_subagent", "sub-1", "无法读取")],
    )
    self.assertEqual(messages[-1].content, "父 Agent 结束")

    output = [entry.args[0] for entry in self.print_log.call_args_list]
    self.assertEqual(output, [
      "\033[33m> run_subagent\033[0m",
      "\033[32m   Args: {'prompt': '调查文件'}\033[0m",
      "\n[permission] Access outside workspace",
      "\033[90m无法读取\033[0m",
    ])


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
    self.assertIn("涉及多个步骤的任务，使用 `todo_write`。", system.text)
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
      AIMessage(content="", tool_calls=[{
        "name": "glob",
        "args": {"pattern": "*.txt"},
        "id": "glob-1",
      }]),
      AIMessage(content="done"),
    ])
    messages = self.run_agent(model, [HumanMessage(content="write then read")])
    self.assertEqual(len(model.calls), 4)
    self.assertEqual(model.calls[-1][1:], messages[:-1])
    self.assertEqual(messages[-2].content, "demo.txt")
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
      patch("lota.main.speak"),
    ):
      main()
    self.assertEqual(len(model.calls), 2)
    self.assertIsInstance(model.calls[0][0], SystemMessage)
    self.assertEqual(
      [item.content for item in model.calls[1][1:]],
      ["first", "first reply", "second"],
    )


  def test_main_voice_reads_only_final_replies_and_recovers(self):
    """只播报最终回复，播报失败后继续对话。"""
    model = ToolCallingModel(responses=[
      AIMessage(content="", tool_calls=[{
        "name": "glob", "args": {"pattern": "*.missing"}, "id": "lookup",
      }]),
      AIMessage(content="第一次回复"),
      AIMessage(content="第二次回复"),
    ])
    with (
      patch("lota.main.build_model", return_value=model),
      patch("builtins.input", side_effect=["first", "second", "q"]),
      patch("lota.main.speak", side_effect=[RuntimeError("离线"), None])
      as speak,
    ):
      main()

    self.assertEqual(
      [entry.args for entry in speak.call_args_list],
      [("第一次回复",), ("第二次回复",)],
    )
    output = [entry.args[0] for entry in self.print_log.call_args_list]
    self.assertIn("\n第一次回复\n", output)
    self.assertIn("[voice] 播报失败：离线", output)
    self.assertIn("\n第二次回复\n", output)


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
      patch("lota.main.speak"),
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
