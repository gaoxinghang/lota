"""创建带权限检查和会话计划的 LangChain Agent。"""

import sys

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lota.middleware.permission import permission_middleware
from lota.middleware.todo_reminder import todo_reminder
from lota.tools.files import edit_file, glob, read_file, write_file
from lota.tools.shell import bash
from lota.tools.todo import build_todo_tool


def build_agent(model: BaseChatModel) -> CompiledStateGraph:
  """创建供同步调用的 Agent，保持工具按请求顺序执行。

  Args:
      model: 尚未绑定工具的聊天模型。

  Returns:
      带权限和计划中间件的 Agent，使用 invoke 调用。
  """
  return create_agent(
    model=model,
    tools=[bash, read_file, write_file, edit_file, glob, build_todo_tool()],
    middleware=[permission_middleware, todo_reminder],
  ).with_config(
    max_concurrency=1,
    # 原循环没有次数限制，避免框架默认步数上限提前结束任务。
    recursion_limit=sys.maxsize,
  )
