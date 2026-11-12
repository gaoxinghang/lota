"""入口：读一行，跑一轮，打印回复。"""

import os

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from lota.loop import agent_loop
from lota.model import build_model
from lota.tools import TOOLS


def main() -> None:
  """启动交互式 REPL，输入 q 或按 Ctrl-C 退出。"""
  llm = build_model().bind_tools(TOOLS)
  shell = "cmd.exe" if os.name == "nt" else "the system shell"
  system = (
    f"You are a coding agent at {os.getcwd()}. "
    f"The bash tool runs commands through {shell}. "
    "Use bash to solve tasks. Act, don't explain."
  )
  history: list[BaseMessage] = [SystemMessage(content=system)]

  print("lota · s01 agent loop —— 输入 q 退出\n")
  while True:
    try:
      query = input("lota › ").strip()
    except (EOFError, KeyboardInterrupt):
      break

    if query.lower() in {"q", "exit"}:
      break
    if not query:
      continue

    history.append(HumanMessage(content=query))
    reply = agent_loop(llm, history)
    print(f"\n{reply.text}\n")


if __name__ == "__main__":
  main()
