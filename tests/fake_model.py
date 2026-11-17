"""提供不访问模型 API 的工具调用测试模型。"""

from langchain_core.language_models.fake_chat_models import (
  FakeMessagesListChatModel,
)
from langchain_core.messages import BaseMessage
from pydantic import Field


class ToolCallingModel(FakeMessagesListChatModel):
  """按预设顺序回复，并记录模型收到的消息。

  Attributes:
      calls: 每次模型调用收到的消息列表。
  """

  calls: list[list[BaseMessage]] = Field(default_factory=list)

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
    """记录本次输入后返回下一条预设回复。"""
    self.calls.append(list(messages))
    return super()._generate(messages, *args, **kwargs)
