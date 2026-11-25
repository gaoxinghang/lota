"""提供不访问模型 API 的工具调用测试模型。"""

from langchain_core.language_models.fake_chat_models import (
  FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from lota.middleware.memory.extraction import EXTRACTION_SYSTEM


class ToolCallingModel(FakeMessagesListChatModel):
  """按预设顺序回复，并记录模型收到的消息。

  Attributes:
      calls: 除记忆提取外，每次模型调用收到的消息列表。
      extraction_calls: 记忆提取收到的消息列表。
      extraction_reply: 固定的提取回复，默认不保存任何内容。
  """

  calls: list[list[BaseMessage]] = Field(default_factory=list)
  extraction_calls: list[list[BaseMessage]] = Field(default_factory=list)
  extraction_reply: str = "[]"

  def bind_tools(self, tools, **kwargs):
    """接受工具绑定，返回仍使用预设回复的模型。

    Args:
        tools: 工具列表，仅用于适配 Agent 的绑定接口。
        **kwargs: 框架传入的绑定选项，不影响预设回复。

    Returns:
        当前测试模型。
    """
    return self


  def _generate(self, messages, *args, **kwargs):
    """独立记录提取调用，不消耗正常的预设回复。"""
    if messages and messages[0].text == EXTRACTION_SYSTEM:
      self.extraction_calls.append(list(messages))
      reply = AIMessage(content=self.extraction_reply)
      return ChatResult(generations=[ChatGeneration(message=reply)])
    self.calls.append(list(messages))
    return super()._generate(messages, *args, **kwargs)
