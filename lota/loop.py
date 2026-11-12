"""Agent 循环：调用模型、执行工具、回传结果，直到模型停止调用工具。"""

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable

from lota.tools import TOOLS

TOOL_MAP = {item.name: item for item in TOOLS}


def agent_loop(
  llm: Runnable[LanguageModelInput, AIMessage],
  messages: list[BaseMessage],
) -> AIMessage:
  """反复调用模型并执行工具，直到模型不再请求工具。

  Args:
      llm: 已绑定工具的聊天模型。
      messages: 对话历史，本轮产生的消息会原地追加进去。

  Returns:
      模型最后一次回复，其中不含工具调用。
  """
  while True:
    response = llm.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
      return response

    for call in response.tool_calls:
      print(f"\033[33m$ {call['args']['command']}\033[0m")
      output = TOOL_MAP[call["name"]].invoke(call["args"])
      print(output[:200])

      # 用调用 ID 将结果与模型的工具请求配对。
      messages.append(
        ToolMessage(
          content=output,
          tool_call_id=call["id"],
          name=call["name"],
        )
      )
