"""在工作区持久化任务、依赖和认领状态。"""

import re
from pathlib import Path
from secrets import token_hex
from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, StringConstraints

TASK_ID_PATTERN = re.compile(r"^task_[0-9a-f]{8}$")
TASK_OWNER = "agent"
TaskId = Annotated[str, StringConstraints(pattern=TASK_ID_PATTERN)]
TaskSubject = Annotated[
  str, StringConstraints(strip_whitespace=True, min_length=1)
]


class TaskRecord(BaseModel):
  """一条跨会话保存的任务记录。

  Attributes:
      id: 运行时生成的唯一任务 ID。
      subject: 非空的任务标题。
      description: 恢复工作所需的任务详情。
      status: 待办、进行中或已完成。
      owner: 认领任务的 Agent，未认领时为空。
      blocked_by: 前置任务的 ID 列表。
  """

  id: TaskId
  subject: TaskSubject
  description: str
  status: Literal["pending", "in_progress", "completed"]
  owner: str | None
  blocked_by: list[TaskId]


class TaskStore:
  """管理工作区 .tasks 目录中的任务，供单个 Agent 串行调用。"""

  def __init__(self, workdir: Path) -> None:
    """定位任务目录，不创建文件。

    Args:
        workdir: 存放 .tasks 的工作区。
    """
    self._root = workdir.resolve() / ".tasks"


  def _directory(self) -> Path:
    """返回任务目录；拒绝通过符号链接访问任务数据。"""
    if self._root.is_symlink():
      raise PermissionError(f"Task directory cannot be a symlink: {self._root}")
    return self._root


  def _path(self, task_id: str) -> Path:
    """根据合法 ID 定位任务文件，拒绝符号链接。"""
    if not TASK_ID_PATTERN.fullmatch(task_id):
      raise ValueError(f"Invalid task ID: {task_id}")
    path = self._directory() / f"{task_id}.json"
    if path.is_symlink():
      raise PermissionError(f"Task file cannot be a symlink: {path}")
    return path


  def _load(self, task_id: str) -> TaskRecord:
    """读取并校验单条任务，确认文件名和记录 ID 一致。"""
    task = TaskRecord.model_validate_json(self._path(task_id).read_bytes())
    if task.id != task_id:
      raise ValueError(f"Task file ID does not match {task_id}")
    return task


  def _save(self, task: TaskRecord) -> None:
    """把已校验的任务状态写回磁盘。"""
    self._path(task.id).write_text(
      task.model_dump_json(indent=2) + "\n", encoding="utf-8",
    )


  def _records(self) -> list[TaskRecord]:
    """按 ID 顺序读取全部任务。"""
    return [
      self._load(path.stem)
      for path in sorted(self._directory().glob("task_*.json"))
    ]


  def _would_cycle(self, task_id: str, dependency: TaskRecord) -> bool:
    """判断新依赖是否能沿既有依赖反向回到目标任务。"""
    pending = dependency.blocked_by.copy()
    seen: set[str] = set()
    while pending:
      current = pending.pop()
      if current == task_id:
        return True
      if current not in seen:
        seen.add(current)
        pending.extend(self._load(current).blocked_by)
    return False


  def _incomplete_dependencies(self, task: TaskRecord) -> list[str]:
    """返回未完成或已找不到记录的前置任务 ID。"""
    incomplete = []
    for dependency in task.blocked_by:
      try:
        if self._load(dependency).status == "completed":
          continue
      except FileNotFoundError:
        pass
      incomplete.append(dependency)
    return incomplete


  def create_task(self, subject: TaskSubject, description: str = "") -> str:
    """创建一条持久化任务，不添加依赖。

    Args:
        subject: 非空任务标题。
        description: 任务详情，供后续会话恢复工作。

    Returns:
        已创建任务的运行时 ID 和标题。
    """
    for _ in range(100):
      task = TaskRecord(
        id=f"task_{token_hex(4)}", subject=subject, description=description,
        status="pending", owner=None, blocked_by=[],
      )
      path = self._path(task.id)
      path.parent.mkdir(exist_ok=True)
      try:
        with path.open("x", encoding="utf-8") as file:
          file.write(task.model_dump_json(indent=2) + "\n")
      except FileExistsError:
        continue
      return f"Created {task.id}: {task.subject}"
    raise FileExistsError("Could not allocate a unique task ID")


  def update_task(
    self,
    task_id: TaskId,
    add_blocked_by: Annotated[list[TaskId], Field(min_length=1)],
  ) -> str:
    """为待办且未认领的任务添加依赖，重复添加不产生重复依赖。

    Args:
        task_id: 要添加依赖的任务 ID。
        add_blocked_by: 至少一个已创建的前置任务 ID。

    Returns:
        更新后的前置任务 ID 列表。

    Raises:
        ValueError: 任务已开始、自依赖或依赖成环。
        FileNotFoundError: 任务或前置任务不存在。
    """
    task = self._load(task_id)
    if task.status != "pending" or task.owner is not None:
      raise ValueError(f"{task_id} must be pending and unowned")

    added = []
    for dependency_id in dict.fromkeys(add_blocked_by):
      if dependency_id == task_id:
        raise ValueError("Task cannot depend on itself")
      dependency = self._load(dependency_id)
      if dependency_id in task.blocked_by:
        continue
      if self._would_cycle(task_id, dependency):
        raise ValueError(f"Dependency cycle: {task_id} -> {dependency_id}")
      added.append(dependency_id)

    if added:
      task.blocked_by.extend(added)
      self._save(task)
    return f"Updated {task_id} blocked_by: {', '.join(task.blocked_by)}"


  def list_tasks(self) -> str:
    """列出持久化任务的 ID、标题、状态、负责人和前置任务。"""
    lines = []
    for task in self._records():
      owner = task.owner or "(none)"
      dependencies = ", ".join(task.blocked_by) or "(none)"
      lines.append(
        f"{task.id}: {task.subject} [{task.status}] "
        f"owner: {owner}; blocked_by: {dependencies}"
      )
    return "\n".join(lines) or "No tasks. Use create_task to add some."


  def get_task(self, task_id: TaskId) -> str:
    """读取完整的持久化任务，包括详情。

    Args:
        task_id: 已创建的任务 ID。

    Returns:
        任务记录的 JSON 文本。
    """
    return self._load(task_id).model_dump_json(indent=2)


  def claim_task(self, task_id: TaskId) -> str:
    """认领未被认领、且所有前置任务均已完成的待办任务。

    Args:
        task_id: 要认领的任务 ID。

    Returns:
        认领结果。

    Raises:
        ValueError: 任务已开始、已被认领或仍被前置任务阻塞。
    """
    task = self._load(task_id)
    if task.status != "pending" or task.owner is not None:
      raise ValueError(f"{task_id} must be pending and unowned")
    incomplete = self._incomplete_dependencies(task)
    if incomplete:
      raise ValueError(f"Blocked by: {', '.join(incomplete)}")

    task.owner = TASK_OWNER
    task.status = "in_progress"
    self._save(task)
    return f"Claimed {task_id}: {task.subject}"


  def complete_task(self, task_id: TaskId) -> str:
    """完成本 Agent 已认领的任务，并报告刚解锁的下游任务。

    Args:
        task_id: 要完成的进行中任务 ID。

    Returns:
        完成结果和刚解锁的任务 ID、标题。

    Raises:
        ValueError: 任务不在进行中或负责人不是本 Agent。
    """
    task = self._load(task_id)
    if task.status != "in_progress" or task.owner != TASK_OWNER:
      raise ValueError(
        f"{task_id} must be in_progress and owned by {TASK_OWNER}"
      )

    unblocked = [
      f"{candidate.id}: {candidate.subject}"
      for candidate in self._records()
      if candidate.status == "pending" and candidate.owner is None
      and task_id in candidate.blocked_by
      and set(self._incomplete_dependencies(candidate)) == {task_id}
    ]
    task.status = "completed"
    self._save(task)
    result = f"Completed {task_id}: {task.subject}"
    if unblocked:
      result += f"\nUnblocked: {', '.join(unblocked)}"
    return result


def build_task_tools(workdir: Path) -> list[BaseTool]:
  """创建共享同一个工作区任务存储的六个任务工具。

  Args:
      workdir: 持久化任务数据的工作区。

  Returns:
      可注册到父 Agent 的任务工具列表。
  """
  store = TaskStore(workdir)
  return [tool(action) for action in (
    store.create_task, store.update_task, store.list_tasks, store.get_task,
    store.claim_task, store.complete_task,
  )]
