"""Shell 工具，执行权限由 Agent 中间件审批。"""

import subprocess

from langchain_core.tools import tool

from lota import workspace


@tool
def bash(command: str) -> str:
  """在启动时的工作目录执行 shell 命令，并返回合并后的输出。

  Args:
      command: 要执行的 shell 命令。

  Returns:
      标准输出与错误输出，最多 50000 字符；无输出时为 "(no output)"。
      超时返回 "Error: Timeout (120s)"。

  Raises:
      OSError: 子进程无法启动，例如工作目录不存在。
  """
  try:
    done = subprocess.run(
      command,
      shell=True,
      cwd=workspace.WORKDIR,
      capture_output=True,
      text=True,
      errors="replace",
      timeout=120,
    )
  except subprocess.TimeoutExpired:
    return "Error: Timeout (120s)"

  output = (done.stdout + done.stderr).strip()
  return output[:50000] if output else "(no output)"
