"""工具执行前的三道权限闸门：硬拒绝、规则匹配、用户审批。"""

import re
from collections.abc import Callable

from langchain.agents.middleware import ToolCallRequest, wrap_tool_call
from langchain_core.messages import ToolMessage

from lota import workspace

DENY_LIST = [
  "rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda",
]
DESTRUCTIVE_COMMAND_WORD = re.compile(
  r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)


def check_deny_list(command: str) -> str | None:
  """匹配教学版命令子串拒绝列表，不解析完整 Shell 语法。

  Args:
      command: 要执行的 Shell 命令。

  Returns:
      命中时返回拒绝原因，否则返回 None。
  """
  for pattern in DENY_LIST:
    if pattern in command:
      return f"Blocked: '{pattern}' is on the deny list"
  return None


def check_rules(tool_name: str, args: dict) -> str | None:
  """判断文件路径、Glob 模式越界或潜在破坏性命令是否需要审批。

  Args:
      tool_name: 已注册的工具名称。
      args: 已通过工具 Schema 校验的调用参数。

  Returns:
      需要审批时返回原因，否则返回 None。

  Raises:
      OSError: 文件路径无法解析。
      ValueError: 文件路径无效。
  """
  path_args = {
    "read_file": "path",
    "write_file": "path",
    "edit_file": "path",
    "glob": "pattern",
  }
  if tool_name in path_args:
    path = (workspace.WORKDIR / args[path_args[tool_name]]).resolve()
    if not path.is_relative_to(workspace.WORKDIR):
      return "Access outside workspace"

  if tool_name == "bash":
    command = args["command"]
    if DESTRUCTIVE_COMMAND_WORD.search(command) or any(
      pattern in command for pattern in ["rm ", "> /etc/", "chmod 777"]
    ):
      return "Potentially destructive command"
  return None


def ask_user(reason: str, action: str) -> bool:
  """展示审批原因和待执行操作，仅明确输入 y 或 yes 时批准。

  Args:
      reason: 需要审批的原因。
      action: 待审批的工具名与参数。

  Returns:
      用户批准时返回 True；其他输入或捕获到输入中断时返回 False。
  """
  print(f"\n[permission] {reason}")
  try:
    choice = input(f"   {action}\n   Allow? [y/N] ").strip().lower()
  except (EOFError, KeyboardInterrupt):
    print()
    return False
  return choice in {"y", "yes"}


def check_permission(tool_name: str, args: dict) -> bool:
  """按硬拒绝、规则匹配和单次审批的顺序决定是否执行工具。

  Args:
      tool_name: 已注册的工具名称。
      args: 已通过工具 Schema 校验的调用参数。

  Returns:
      可以执行时返回 True，否则返回 False；审批不跨调用复用。

  Raises:
      OSError: 文件路径无法解析。
      ValueError: 文件路径无效。
  """
  if tool_name == "bash":
    reason = check_deny_list(args["command"])
    if reason:
      print(f"\n[blocked] {reason}")
      return False

  reason = check_rules(tool_name, args)
  if reason:
    return ask_user(reason, f"{tool_name} {args!r}")
  return True


@wrap_tool_call
def permission_middleware(
  request: ToolCallRequest,
  handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
  """校验工具参数并审批，将拒绝或错误回传给模型。

  Args:
      request: Agent 提供的工具调用及对应工具。
      handler: 框架提供的同步工具执行函数。

  Returns:
      执行结果、拒绝信息或可回传的错误，均与原调用 ID 配对。
  """
  call = request.tool_call
  try:
    if request.tool is None:
      raise ValueError(f"Unknown tool: {call['name']}")
    request.tool.get_input_schema().model_validate(call["args"])
    if check_permission(call["name"], call["args"]):
      return handler(request)
    output = "Permission denied."
  except (OSError, ValueError) as exc:
    output = f"Error: {exc}"

  return ToolMessage(
    content=output,
    tool_call_id=call["id"],
    name=call["name"],
    status="error",
  )
