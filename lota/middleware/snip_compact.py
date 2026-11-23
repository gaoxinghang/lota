"""归档完整消息列表，再裁剪中段历史。"""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from langchain.agents.middleware import AgentState, before_model
from langchain_core.messages import (
  BaseMessage, HumanMessage, RemoveMessage, ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from lota import workspace

TRIGGER_MESSAGES = 50
TARGET_MESSAGES = 30
KEEP_HEAD_MESSAGES = 3
ARCHIVED_REQUEST_KEY = "lota_archived_request"


def write_transcript(messages: list[BaseMessage]) -> Path:
  """将裁剪前的消息保存为可由 LangChain 还原的 JSONL。

  Args:
    messages: 按顺序保存的全部消息。

  Returns:
    归档文件相对于工作区的路径。

  Raises:
    OSError: 归档目录越界，或无法创建目录、写入归档。
  """
  workdir = workspace.WORKDIR.resolve()
  directory = (workdir / ".transcripts").resolve()
  if not directory.is_relative_to(workdir):
    raise PermissionError("Transcript directory is outside the workspace")
  directory.mkdir(parents=True, exist_ok=True)

  with NamedTemporaryFile(
    mode="w", encoding="utf-8", prefix="transcript_", suffix=".jsonl",
    dir=directory, delete=False,
  ) as transcript:
    for message in messages:
      record = {"type": message.type, "data": message.model_dump(mode="json")}
      transcript.write(json.dumps(record, ensure_ascii=False) + "\n")
  return Path(transcript.name).relative_to(workdir)


def _request_in_middle(
  messages: list[BaseMessage], head_end: int, tail_start: int
) -> str | None:
  """仅当当前用户请求即将被裁剪时，返回其文本。

  Args:
    messages: 裁剪前的消息列表。
    head_end: 保留的开头之后的第一个索引。
    tail_start: 保留的结尾的起始索引。

  Returns:
    即将被裁剪的请求文本，包括旧提示保留的请求；否则为 None。
  """
  for index in range(len(messages) - 1, head_end - 1, -1):
    message = messages[index]
    if not isinstance(message, HumanMessage):
      continue
    request = message.additional_kwargs.get(ARCHIVED_REQUEST_KEY, message.text)
    if request is not None:
      return request if index < tail_start else None
  return None


@before_model
def snip_compact(
  state: AgentState, runtime: Runtime
) -> dict[str, list[BaseMessage]] | None:
  """消息达到阈值时先归档，再以保留的头尾和提示替换历史。

  Args:
    state: 当前 Agent 状态，已应用工具结果预算。
    runtime: LangGraph 运行时，由框架传入。

  Returns:
    替换历史的消息更新；未达到阈值或无法缩短历史时返回 None。
  """
  messages = state["messages"]
  if len(messages) < TRIGGER_MESSAGES:
    return None

  head_end = KEEP_HEAD_MESSAGES
  tail_start = len(messages) - (TARGET_MESSAGES - KEEP_HEAD_MESSAGES - 1)

  while head_end < len(messages) and isinstance(messages[head_end], ToolMessage):
    head_end += 1

  while tail_start > head_end and isinstance(messages[tail_start], ToolMessage):
    tail_start -= 1

  removed = tail_start - head_end
  if removed <= 1:
    return None

  request = _request_in_middle(messages, head_end, tail_start)
  path = write_transcript(messages)
  notice = f"[已归档 {removed} 条消息，完整记录：{path}]"
  if request is not None:
    notice += f"\n\n当前用户请求：\n{request}"
  marker = HumanMessage(
    content=notice, additional_kwargs={ARCHIVED_REQUEST_KEY: request},
  )
  return {"messages": [
    RemoveMessage(id=REMOVE_ALL_MESSAGES),
    *messages[:head_end], marker, *messages[tail_start:],
  ]}
