"""入口：读一行，跑一轮，打印回复。"""

import os

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from lota.agent import build_agent
from lota.model import build_model


def main() -> None:
  """启动交互式 REPL，输入 q 或按 Ctrl-C 退出。"""
  try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
  except ImportError:
    pass

  agent = build_agent(build_model())
  shell = "cmd.exe" if os.name == "nt" else "the system shell"
  system = (
    f"You are a coding agent at {os.getcwd()}. "
    f"The bash tool runs commands through {shell}. "
    "Use tools to solve tasks. Prefer file tools for file operations. "
    "Do not ask for approval yourself. For destructive operations, "
    "still request the tool call; the harness will approve or deny "
    "before execution."
  )
  history: list[BaseMessage] = [SystemMessage(content=system)]

  print("lota · s03 permission —— 输入 q 退出\n")
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
    result = agent.invoke({"messages": history})
    history = result["messages"]
    print(f"\n{history[-1].text}\n")


if __name__ == "__main__":
  main()
