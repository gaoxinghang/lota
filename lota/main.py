"""入口：读取输入、运行 Agent，显示并播报回复。"""

from contextlib import redirect_stdout
from io import StringIO

from langchain_core.messages import BaseMessage, HumanMessage

from lota.agent import build_agent
from lota.model import build_model
from lota.voice import speak


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

  print("lota · 你的 AI 助理 —— 输入 q 退出\n")
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
    reply = history[-1].text
    print(f"\n{reply}\n", flush=True)
    if reply.strip():
      try:
        with redirect_stdout(StringIO()):
          speak(reply)
      except KeyboardInterrupt:
        print("[voice] 播报已停止")
      except Exception as exc:
        print(f"[voice] 播报失败：{exc}")


if __name__ == "__main__":
  main()
