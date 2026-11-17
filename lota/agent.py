"""创建带执行前权限检查的 LangChain Agent。"""

import sys

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lota.permission import permission_middleware
from lota.tools import TOOLS


def build_agent(model: BaseChatModel) -> CompiledStateGraph:
  """创建供同步调用的 Agent，保持工具按请求顺序执行。

  Args:
      model: 尚未绑定工具的聊天模型。

  Returns:
      带权限中间件的 Agent，通过 invoke 传入 messages 状态调用。
  """
  return create_agent(
    model=model,
    tools=TOOLS,
    middleware=[permission_middleware],
  ).with_config(
    max_concurrency=1,
    # 原循环没有次数限制，避免框架默认步数上限提前结束任务。
    recursion_limit=sys.maxsize,
  )
