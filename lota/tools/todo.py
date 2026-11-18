"""维护当前 Agent 会话的 TODO 清单。"""

from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, StringConstraints


class TodoItem(BaseModel):
  """一条带状态的计划项。

  Attributes:
      content: 去除首尾空白后非空的任务描述。
      status: 待办、进行中或已完成。
  """

  content: Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
  ]
  status: Literal["pending", "in_progress", "completed"]


class TodoManager:
  """保存一个 Agent 会话当前的清单。"""

  def __init__(self) -> None:
    """创建空清单。"""
    self._items: list[TodoItem] = []


  def todo_write(
    self, todos: Annotated[list[TodoItem], Field(max_length=20)]
  ) -> str:
    """用最多 20 项替换清单，同时最多一项进行中。

    Args:
        todos: 完整清单；传空列表清空计划。

    Returns:
        渲染后的清单和完成数。

    Raises:
        ValueError: 超过一项同时进行。
    """
    if sum(item.status == "in_progress" for item in todos) > 1:
      raise ValueError("Only one todo can be in_progress at a time")

    self._items = todos
    output = self.render()
    print(f"\n## Current Tasks\n{output}")
    return output


  def render(self) -> str:
    """返回清单及完成数，或返回空清单提示。"""
    if not self._items:
      return "No todos."

    markers = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
    lines = [f"{markers[item.status]} {item.content}" for item in self._items]
    done = sum(item.status == "completed" for item in self._items)
    lines.append(f"\n({done}/{len(self._items)} completed)")
    return "\n".join(lines)


def build_todo_tool() -> BaseTool:
  """创建绑定到独立会话清单的 todo_write 工具。

  Returns:
      可注册到 Agent 的清单工具。
  """
  return tool(TodoManager().todo_write)
