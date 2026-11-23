"""把独立的子 Agent 封装成工具。"""

from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from pydantic import StringConstraints


def build_subagent_tool(child_agent: CompiledStateGraph) -> BaseTool:
  """创建绑定到子 Agent 的 run_subagent 工具。

  Args:
      child_agent: 不保存消息历史的子 Agent。

  Returns:
      可注册到父 Agent 的委派工具。
  """

  @tool
  def run_subagent(
    prompt: Annotated[
      str, StringConstraints(strip_whitespace=True, min_length=1)
    ],
  ) -> str:
    """委派一个独立的子任务，只返回最终结论。

    Args:
        prompt: 非空的任务描述，包含完成任务所需的上下文。

    Returns:
        子 Agent 的最终文本，或达到步数上限的提示。
    """
    try:
      result = child_agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        # 独立的第四步在每次模型调用前再占一个图步。
        config={"max_concurrency": 1, "recursion_limit": 44},
      )
    except GraphRecursionError:
      return "Error: Subagent reached its step limit."
    return result["messages"][-1].text

  return run_subagent
