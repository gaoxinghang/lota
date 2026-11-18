"""按工具批次向模型附加 TODO 更新提醒。"""

from collections.abc import Callable

from langchain.agents.middleware import (
  ModelRequest, ModelResponse, wrap_model_call,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

REMINDER = "<reminder>请用 todo_write 更新 TODO 清单。</reminder>"


@wrap_model_call
def todo_reminder(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
  """连续三批未更新清单时，向当次模型调用附上提醒。

  Args:
      request: 包含完整工具结果的模型请求。
      handler: 执行模型调用的同步函数。

  Returns:
      模型回复；消息历史不写入提醒。
  """
  messages = request.messages
  if not messages or not isinstance(last := messages[-1], ToolMessage):
    return handler(request)

  rounds = 0
  for message in reversed(messages):
    if isinstance(message, HumanMessage):
      break
    if isinstance(message, ToolMessage):
      if message.name == "todo_write" and message.status == "success":
        break
    elif isinstance(message, AIMessage) and message.tool_calls:
      rounds += 1  # 一条模型回复只算一批。

  if rounds and rounds % 3 == 0:
    result = last.model_copy(update={"content": f"{last.text}\n\n{REMINDER}"})
    request = request.override(messages=[*messages[:-1], result])
  return handler(request)
