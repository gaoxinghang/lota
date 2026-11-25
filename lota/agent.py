"""创建带权限检查、会话计划和按需技能加载的 LangChain Agent。"""

import sys

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lota import workspace
from lota.middleware.context_compact import build_context_compact
from lota.middleware.memory import MemoryMiddleware
from lota.middleware.permission import permission_middleware
from lota.middleware.tool_logging import tool_logging
from lota.tools.files import edit_file, glob, read_file, write_file
from lota.tools.shell import bash
from lota.tools.skills import build_skill_tool
from lota.tools.subagent import build_subagent_tool
from lota.tools.todo import build_todo_tool


def build_agent(model: BaseChatModel) -> CompiledStateGraph:
  """创建供同步调用的 Agent，保持工具按请求顺序执行。

  Args:
      model: 尚未绑定工具的聊天模型。

  Returns:
      带权限检查、计划和技能工具的 Agent，使用 invoke 调用。
  """
  base_tools = [bash, read_file, write_file, edit_file, glob]
  child_agent = create_agent(
    model=model,
    system_prompt="你是子 Agent。用可用工具完成子任务，只返回最终结论。",
    tools=base_tools,
    middleware=[permission_middleware],
    checkpointer=False,
  )

  parent_tools = base_tools.copy()
  skill_tool, skill_prompt = build_skill_tool(workspace.WORKDIR)
  if skill_tool is not None:
    parent_tools.append(skill_tool)
  parent_tools.extend([build_todo_tool(), build_subagent_tool(child_agent)])

  return create_agent(
    model=model,
    system_prompt=(
      "最终回复会被语音播报。像平时聊天一样，用自然的口语直接回答，"
      "先说结论，默认一两句短句就够。不要复述问题、加客套开场，"
      "也不必逐项汇报工具操作。能用短句说清，就不用标题或列表。"
      "只有用户明确要求展开，或需要交代关键风险、限制时，才多说几句。"
      "\n\n可用工具：\n"
      "- `bash`：执行 Shell 命令。\n"
      "- `read_file`：读取文本文件。\n"
      "- `write_file`：写入或覆盖文本文件。\n"
      "- `edit_file`：替换文件中第一处匹配的文本。\n"
      "- `glob`：按模式查找文件或目录。\n"
      "- `todo_write`：维护 TODO 清单。\n"
      "- `run_subagent`：委派独立的子任务，只返回结论。\n\n"
      "涉及多个步骤的任务，使用 `todo_write`。"
      + skill_prompt
    ),
    tools=parent_tools,
    middleware=[
      tool_logging, permission_middleware, build_context_compact(model),
      MemoryMiddleware(model, workspace.WORKDIR),
    ],
  ).with_config(
    max_concurrency=1,
    # 原循环没有次数限制，避免框架默认步数上限提前结束任务。
    recursion_limit=sys.maxsize,
  )
