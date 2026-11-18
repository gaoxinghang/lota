"""创建带权限检查和会话计划的 LangChain Agent。"""

import sys

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lota.middleware.permission import permission_middleware
from lota.middleware.todo_reminder import todo_reminder
from lota.middleware.tool_logging import tool_logging
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
    system_prompt=(
      "可用工具：\n"
      "- `bash`：执行 Shell 命令。\n"
      "- `read_file`：读取文本文件。\n"
      "- `write_file`：写入或覆盖文本文件。\n"
      "- `edit_file`：替换文件中第一处匹配的文本。\n"
      "- `glob`：按模式查找文件或目录。\n"
      "- `todo_write`：维护 TODO 清单。\n\n"
      "复杂任务先用 `todo_write` 列出计划，执行中更新进度。"
      "每连续三批工具调用未成功更新清单时，"
      "系统会在该批最后一条工具结果末尾提醒你更新 TODO 清单。"
    ),
    tools=[bash, read_file, write_file, edit_file, glob, build_todo_tool()],
    middleware=[tool_logging, permission_middleware, todo_reminder],
  ).with_config(
    max_concurrency=1,
    # 原循环没有次数限制，避免框架默认步数上限提前结束任务。
    recursion_limit=sys.maxsize,
  )
