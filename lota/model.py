"""通过 LangChain 统一接口创建聊天模型。"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def build_model() -> BaseChatModel:
  """读取 .env 并创建使用 OpenAI 协议的聊天模型。

  Returns:
      尚未绑定工具的聊天模型。

  Raises:
      ImportError: 未安装 langchain-openai。
  """
  load_dotenv(".env", override=True, interpolate=False)

  return init_chat_model(
    model=os.getenv("MODEL_ID"),
    model_provider="openai",
    max_tokens=8000,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
  )
