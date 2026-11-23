"""上下文超限时，把已读的旧工具结果替换成原文路径。"""

from langchain.agents.middleware import AgentState, before_model
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from lota import workspace
from lota.middleware.tool_result_budget import (
  PERSISTED_RESULT_PATH_KEY, persist_result,
)

CONTEXT_CHAR_LIMIT = 50_000
TARGET_CONTEXT_CHARS = 40_000
KEEP_RECENT_RESULTS = 3
MIN_RESULT_CHARS = 120


@before_model
def micro_compact(
  state: AgentState, runtime: Runtime
) -> dict[str, list[ToolMessage]] | None:
  """按历史顺序缩短旧工具结果，保留最近三条已读结果。

  Args:
    state: 工具结果转存与历史裁剪后的 Agent 状态。
    runtime: LangGraph 运行时，由框架传入。

  Returns:
    按原消息 ID 更新的工具结果；没有缩短时返回 None。
  """
  messages = state["messages"]
  last_reply = next(
    (
      index for index in range(len(messages) - 1, -1, -1)
      if isinstance(messages[index], AIMessage)
    ),
    0,
  )
  seen = [
    message for message in messages[:last_reply]
    if isinstance(message, ToolMessage)
  ]
  candidates = seen[:-KEEP_RECENT_RESULTS]
  if not candidates:
    return None

  total = sum(len(message.model_dump_json()) for message in messages)
  if total <= CONTEXT_CHAR_LIMIT:
    return None

  updates = []
  for message in candidates:
    if total <= TARGET_CONTEXT_CHARS:
      break
    output = message.content
    if not isinstance(output, str) or len(output) <= MIN_RESULT_CHARS:
      continue

    path = message.additional_kwargs.get(PERSISTED_RESULT_PATH_KEY)
    if path is None:
      path = str(persist_result(output))
    elif not (workspace.WORKDIR / path).is_file():
      continue

    notice = f"[旧工具结果已保存到 {path}]"
    if len(notice) >= len(output):
      continue

    compacted = message.model_copy(update={
      "content": notice,
      "additional_kwargs": message.additional_kwargs | {
        PERSISTED_RESULT_PATH_KEY: path,
      },
    })
    updates.append(compacted)
    total += len(compacted.model_dump_json()) - len(message.model_dump_json())

  return {"messages": updates} if updates else None
