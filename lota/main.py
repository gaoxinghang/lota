"""入口：读一行，跑一轮，打印回复。"""

from langchain_core.messages import BaseMessage, HumanMessage

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
  history: list[BaseMessage] = []

  print("lota · 命令行编程助手 —— 输入 q 退出\n")
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
