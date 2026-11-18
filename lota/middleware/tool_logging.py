"""统一打印工具调用参数和结果预览。"""

from collections.abc import Callable

from langchain.agents.middleware import ToolCallRequest, wrap_tool_call
from langchain_core.messages import ToolMessage


@wrap_tool_call
def tool_logging(
  request: ToolCallRequest,
  handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
  """打印工具名、完整参数和结果的前 200 个字符。

  Args:
      request: Agent 提供的工具调用。
      handler: 继续处理工具调用的同步函数。

  Returns:
      未修改的工具结果。
  """
  call = request.tool_call
  print(f"\033[33m> {call['name']}\033[0m")
  print(f"\033[32m   Args: {call['args']!r}\033[0m")
  result = handler(request)
  print(f"\033[90m{result.text[:200]}\033[0m")
  return result
