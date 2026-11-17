"""验证权限闸门和实际工具分发，不调用模型 API 或真实 Shell。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage

from fake_model import ToolCallingModel
from lota import permission, tools
from lota.agent import build_agent


class PermissionTest(unittest.TestCase):
  """覆盖硬拒绝、命令规则、单次审批与越界文件操作。"""

  def setUp(self):
    """创建临时工作区，默认禁止测试意外请求真实输入。"""
    temporary = TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name).resolve()
    self.workspace = root / "workspace"
    self.workspace.mkdir()
    self.outside = root / "outside"
    self.outside.mkdir()
    (self.workspace / "link").symlink_to(
      self.outside, target_is_directory=True
    )
    self.enterContext(patch.object(tools, "WORKDIR", self.workspace))
    self.enterContext(patch("builtins.print"))
    self.input_mock = self.enterContext(patch(
      "builtins.input", side_effect=AssertionError("Unexpected approval")
    ))


  def run_tool(self, name: str, args: dict) -> ToolMessage:
    """通过真实 Agent 循环执行一次工具，并验证结果配对。

    Args:
        name: 工具名称。
        args: 调用参数。

    Returns:
        回传模型的工具结果消息。
    """
    request = AIMessage(content="", tool_calls=[{
      "name": name, "args": args, "id": "call-1",
    }])
    final = AIMessage(content="done")
    model = ToolCallingModel(responses=[request, final])
    state = build_agent(model).invoke({"messages": []})
    self.assertEqual(state["messages"][-1].content, final.content)
    self.assertEqual(len(model.calls), 2)
    result = state["messages"][1]
    self.assertIsInstance(result, ToolMessage)
    self.assertEqual(result.tool_call_id, "call-1")
    self.assertEqual(result.name, name)
    return result


  def test_hard_deny_never_prompts_or_executes(self):
    """硬拒绝优先于审批，所有拒绝项都不能启动子进程。"""
    with patch("lota.tools.subprocess.run") as run:
      for command in permission.DENY_LIST:
        with self.subTest(command=command):
          result = self.run_tool("bash", {"command": command})
          self.assertEqual(result.content, "Permission denied.")
      run.assert_not_called()
    self.input_mock.assert_not_called()


  def test_command_rules(self):
    """匹配命令位置、大小写和原章子串规则，不误判相似单词。"""
    destructive = [
      "rm file.txt", "/usr/bin/rm file.txt", "command rm file.txt",
      "DEL file.txt", "echo ready; rm file.txt",
      "echo ready && del file.txt", "echo ready || RM file.txt",
      "echo ready | del file.txt", "echo ready & rm file.txt",
      "(del file.txt)", "rm; echo ready", "echo ready\nRM file.txt",
      "echo value > /etc/example", "chmod 777 file.txt",
    ]
    ordinary = [
      "ls", "model list", "delimiter file.txt", "echo del file.txt",
      "echo; delimiter file.txt",
    ]
    for command in destructive + ordinary:
      with self.subTest(command=command):
        reason = permission.check_rules("bash", {"command": command})
        self.assertEqual(bool(reason), command in destructive)


  def test_approval_inputs(self):
    """只有明确的 y 或 yes 才批准，其他输入默认拒绝。"""
    for choice in ["y", "yes", " YES ", "", "n", "no", "ok"]:
      with self.subTest(choice=choice), patch(
        "builtins.input", return_value=choice
      ):
        self.assertEqual(
          permission.ask_user("bash", {"command": "rm x"}, "test"),
          choice.strip().lower() in {"y", "yes"},
        )


  def test_interrupted_approval_denies(self):
    """模拟输入抛出 EOFError 或 KeyboardInterrupt 时拒绝执行。"""
    with patch("lota.tools.subprocess.run") as run:
      for error in [EOFError, KeyboardInterrupt]:
        with self.subTest(error=error), patch(
          "builtins.input", side_effect=error
        ):
          result = self.run_tool("bash", {"command": "rm file.txt"})
          self.assertEqual(result.content, "Permission denied.")
      run.assert_not_called()


  def test_shell_approval_is_per_call(self):
    """每次危险命令都审批，上次批准不影响下一次拒绝。"""
    with patch("builtins.input", side_effect=["yes", "n"]) as prompt:
      with patch("lota.tools.subprocess.run") as run:
        run.return_value = Mock(stdout="removed", stderr="")
        args = {"command": "rm file.txt"}
        self.assertEqual(self.run_tool("bash", args).content, "removed")
        self.assertEqual(
          self.run_tool("bash", args).content, "Permission denied."
        )
        run.assert_called_once()
      self.assertEqual(prompt.call_count, 2)


  def test_batch_approval_is_serial(self):
    """同批审批按请求顺序执行，批准不复用于后续调用。"""
    request = AIMessage(content="", tool_calls=[
      {"name": "bash", "args": {"command": "rm first"}, "id": "first"},
      {"name": "bash", "args": {"command": "rm second"}, "id": "second"},
    ])
    model = ToolCallingModel(responses=[request, AIMessage(content="done")])
    with (
      patch("builtins.input", side_effect=["y", "n"]) as prompt,
      patch("lota.tools.subprocess.run") as run,
    ):
      run.return_value = Mock(stdout="removed", stderr="")
      state = build_agent(model).invoke({"messages": []})
    self.assertEqual(prompt.call_count, 2)
    run.assert_called_once()
    self.assertEqual(run.call_args.args[0], "rm first")
    results = [
      item for item in state["messages"] if isinstance(item, ToolMessage)
    ]
    self.assertEqual(
      [(item.tool_call_id, item.content) for item in results],
      [("first", "removed"), ("second", "Permission denied.")],
    )


  def test_workspace_operations_do_not_prompt(self):
    """普通 Shell 和工作区内读写、内部链接、Glob 直接放行。"""
    with patch("lota.tools.subprocess.run") as run:
      run.return_value = Mock(stdout="hello", stderr="")
      self.assertEqual(
        self.run_tool("bash", {"command": "echo hello"}).content, "hello"
      )
    self.run_tool("write_file", {"path": "demo.txt", "content": "hello"})
    self.run_tool("edit_file", {
      "path": "demo.txt", "old_text": "hello", "new_text": "world",
    })
    (self.workspace / "internal.txt").symlink_to(
      self.workspace / "demo.txt"
    )
    for path in ["demo.txt", str(self.workspace / "demo.txt"), "internal.txt"]:
      self.assertEqual(
        self.run_tool("read_file", {"path": path}).content, "world"
      )
    self.run_tool("glob", {"pattern": "*.txt"})
    self.input_mock.assert_not_called()


  def test_outside_file_approval(self):
    """越界读写和编辑只有批准才执行，覆盖绝对路径和外部链接。"""
    target = self.outside / "demo.txt"
    for path in ["../outside/demo.txt", str(target), "link/demo.txt"]:
      calls = [
        ("read_file", {"path": path}),
        ("write_file", {"path": path, "content": "changed"}),
        ("edit_file", {
          "path": path, "old_text": "unchanged", "new_text": "changed",
        }),
      ]
      for name, args in calls:
        for approved in [False, True]:
          with self.subTest(path=path, tool=name, approved=approved):
            target.write_text("unchanged", encoding="utf-8")
            handler = getattr(tools, name)
            with (
              patch(
                "builtins.input", return_value="y" if approved else "n"
              ) as prompt,
              patch.object(
                handler, "func", autospec=True, side_effect=handler.func
              ) as execute,
            ):
              result = self.run_tool(name, args)
              prompt.assert_called_once()
              self.assertEqual(execute.call_count, int(approved))
            if not approved:
              self.assertEqual(result.content, "Permission denied.")
            elif name == "read_file":
              self.assertEqual(result.content, "unchanged")
            else:
              self.assertFalse(result.content.startswith("Error:"))
            expected = "unchanged"
            if approved and name != "read_file":
              expected = "changed"
            self.assertEqual(target.read_text(encoding="utf-8"), expected)


  def test_outside_glob_approval(self):
    """越界 Glob 先审批，拒绝不搜索，批准后返回外部匹配。"""
    target = self.outside / "demo.txt"
    target.touch()
    cases = [
      ("../outside/*.txt", "../outside/demo.txt"),
      (str(self.outside / "*.txt"), str(target)),
      ("link/*.txt", "link/demo.txt"),
      ("../outside/*.missing", "(no matches)"),
    ]
    for pattern, expected in cases:
      for approved in [False, True]:
        with self.subTest(pattern=pattern, approved=approved):
          with (
            patch(
              "builtins.input", return_value="y" if approved else "n"
            ) as prompt,
            patch("lota.tools.glob_paths", wraps=tools.glob_paths) as search,
          ):
            result = self.run_tool("glob", {"pattern": pattern})
            prompt.assert_called_once()
            self.assertEqual(search.call_count, int(approved))
          self.assertEqual(
            result.content, expected if approved else "Permission denied."
          )


  def test_workspace_glob_checks_pattern_only(self):
    """工作区内模式直接通过，通配符匹配到的外部链接不会额外审批。"""
    target = self.outside / "demo.txt"
    target.touch()
    (self.workspace / "external.txt").symlink_to(target)
    for pattern, expected in [
      ("*.txt", "external.txt"),
      (str(self.workspace / "*.txt"), str(self.workspace / "external.txt")),
      ("l*/*.txt", "link/demo.txt"),
    ]:
      with self.subTest(pattern=pattern):
        result = self.run_tool("glob", {"pattern": pattern})
        self.assertEqual(result.content, expected)
    self.input_mock.assert_not_called()


  def test_denied_write_does_not_create_directories(self):
    """拒绝新建外部文件时，连父目录也不能创建。"""
    with patch("builtins.input", return_value="n"):
      result = self.run_tool("write_file", {
        "path": "../outside/nested/demo.txt", "content": "hello",
      })
    self.assertEqual(result.content, "Permission denied.")
    self.assertFalse((self.outside / "nested").exists())


  def test_invalid_args_fail_before_approval(self):
    """缺失或错误类型参数先回传校验错误，不进入审批。"""
    calls = [
      ("bash", {}),
      ("bash", {"command": None}),
      ("read_file", {"path": 123}),
      ("glob", {}),
      ("glob", {"pattern": None}),
      ("write_file", {"path": "../outside/demo.txt"}),
      ("edit_file", {"path": "../outside/demo.txt", "old_text": "hello"}),
    ]
    with patch("lota.tools.subprocess.run") as run:
      for name, args in calls:
        with self.subTest(tool=name, args=args):
          self.assertTrue(
            self.run_tool(name, args).content.startswith("Error:")
          )
      run.assert_not_called()
    self.input_mock.assert_not_called()


  def test_denial_keeps_later_tool_results(self):
    """拒绝一项调用不影响同批后续调用及各自的结果 ID。"""
    request = AIMessage(content="", tool_calls=[
      {"name": "bash", "args": {"command": "sudo ls"}, "id": "denied"},
      {"name": "glob", "args": {"pattern": "*.missing"}, "id": "allowed"},
    ])
    model = ToolCallingModel(responses=[request, AIMessage(content="done")])
    state = build_agent(model).invoke({"messages": []})
    results = [
      item for item in state["messages"] if isinstance(item, ToolMessage)
    ]
    self.assertEqual(
      [(item.tool_call_id, item.content) for item in results],
      [("denied", "Permission denied."), ("allowed", "(no matches)")],
    )
    self.input_mock.assert_not_called()
