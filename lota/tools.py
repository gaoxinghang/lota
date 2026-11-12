"""Shell 工具：执行命令并返回输出。"""

import subprocess

from langchain_core.tools import tool


@tool
def bash(command: str) -> str:
  """在进程当前工作目录执行 shell 命令，并返回合并后的输出。

  Args:
      command: 要执行的 shell 命令。

  Returns:
      标准输出与错误输出，最多 50000 字符；无输出时为 "(no output)"。
  """
  try:
    done = subprocess.run(
      command,
      shell=True,
      capture_output=True,
      text=True,
      errors="replace",
      timeout=120,
    )
  except subprocess.TimeoutExpired:
    return "Error: Timeout (120s)"
  except OSError as exc:
    return f"Error: {exc}"

  output = (done.stdout + done.stderr).strip()
  return output[:50000] if output else "(no output)"


TOOLS = [bash]
