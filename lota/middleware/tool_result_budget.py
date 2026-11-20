"""在调用模型前转存超预算批次中的大工具结果。"""

from pathlib import Path
from tempfile import NamedTemporaryFile

from langchain.agents.middleware import AgentState, before_model
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime

from lota import workspace

TOOL_RESULT_BATCH_CHAR_LIMIT = 200_000
LARGE_RESULT_CHAR_LIMIT = 30_000
PREVIEW_CHAR_LIMIT = 2_000


def _persist_result(output: str) -> Path:
  """将完整文本写入工作区，返回用于工具读取的相对路径。

  Args:
      output: 要保存的完整工具结果。

  Returns:
      新建输出文件相对于工作区的路径。

  Raises:
      OSError: 输出目录越界，或无法创建目录、写入结果。
  """
  workdir = workspace.WORKDIR.resolve()
  output_dir = (workdir / ".task_outputs" / "tool-results").resolve()
  if not output_dir.is_relative_to(workdir):
    raise PermissionError("Tool result directory is outside the workspace")
  output_dir.mkdir(parents=True, exist_ok=True)

  with NamedTemporaryFile(
    mode="w", encoding="utf-8", newline="", suffix=".txt",
    dir=output_dir, delete=False,
  ) as saved:
    saved.write(output)
  return Path(saved.name).relative_to(workdir)


@before_model
def tool_result_budget(
  state: AgentState, runtime: Runtime
) -> dict[str, list[ToolMessage]] | None:
  """预算超限时，按从大到小的顺序用路径和预览替换大结果。

  Args:
      state: 当前 Agent 状态，只检查末尾连续的工具消息。
      runtime: LangGraph 运行时，由框架传入。

  Returns:
      按原消息 ID 更新的工具消息；没有要替换的结果时返回 None。
  """
  batch: list[tuple[ToolMessage, str]] = []
  for message in reversed(state["messages"]):
    if not isinstance(message, ToolMessage):
      break
    if isinstance(message.content, str):
      batch.append((message, message.content))

  total = sum(len(output) for _, output in batch)
  if total <= TOOL_RESULT_BATCH_CHAR_LIMIT:
    return None

  updates = []
  for message, output in sorted(batch, key=lambda pair: -len(pair[1])):
    if total <= TOOL_RESULT_BATCH_CHAR_LIMIT:
      break
    if len(output) <= LARGE_RESULT_CHAR_LIMIT:
      break

    path = _persist_result(output)
    preview = (
      f"<persisted-output>\nFull output: {path}\n"
      f"Preview:\n{output[:PREVIEW_CHAR_LIMIT]}\n</persisted-output>"
    )
    updates.append(message.model_copy(update={"content": preview}))
    total += len(preview) - len(output)

  return {"messages": updates} if updates else None
