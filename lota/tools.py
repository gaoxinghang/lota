"""Shell 与文件工具：独立注册，执行权限由 Agent 中间件审批。"""

import subprocess
from glob import glob as glob_paths
from pathlib import Path

from langchain_core.tools import tool

WORKDIR = Path.cwd()


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
      cwd=WORKDIR,
      capture_output=True,
      text=True,
      errors="replace",
      timeout=120,
    )
  except subprocess.TimeoutExpired:
    return "Error: Timeout (120s)"

  output = (done.stdout + done.stderr).strip()
  return output[:50000] if output else "(no output)"


@tool
def read_file(path: str, limit: int | None = None) -> str:
  """读取 UTF-8 文本文件，可限制返回行数；调用方负责权限审批。

  Args:
      path: 要读取的文件路径。
      limit: 最多返回的行数，必须为正整数；省略时读取全部行。

  Returns:
      文件内容；截断时附上剩余行数。

  Raises:
      ValueError: 行数无效或文件不是 UTF-8 文本。
      OSError: 文件无法读取。
  """
  if limit is not None and limit <= 0:
    raise ValueError("limit must be a positive integer")

  lines = (WORKDIR / path).resolve().read_text(encoding="utf-8").splitlines()
  if limit is not None and len(lines) > limit:
    remaining = len(lines) - limit
    lines = lines[:limit] + [f"... ({remaining} more lines)"]
  return "\n".join(lines)


@tool
def write_file(path: str, content: str) -> str:
  """写入 UTF-8 文本，覆盖文件并创建父目录；调用方负责权限审批。

  Args:
      path: 要写入的文件路径。
      content: 完整的文件内容。

  Returns:
      写入的字符数与文件路径。

  Raises:
      ValueError: 内容无法编码为 UTF-8。
      OSError: 目录无法创建或文件无法写入。
  """
  file_path = (WORKDIR / path).resolve()
  file_path.parent.mkdir(parents=True, exist_ok=True)
  file_path.write_text(content, encoding="utf-8")
  return f"Wrote {len(content)} characters to {path}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
  """精确替换 UTF-8 文件的第一处匹配文本；调用方负责权限审批。

  Args:
      path: 要编辑的文件路径。
      old_text: 要查找的原文，不得为空。
      new_text: 替换后的文本，可以为空。

  Returns:
      已编辑的文件路径。

  Raises:
      ValueError: 原文为空或未匹配，或文本不符合 UTF-8。
      OSError: 文件无法读取或写入。
  """
  if not old_text:
    raise ValueError("old_text must not be empty")

  file_path = (WORKDIR / path).resolve()
  content = file_path.read_text(encoding="utf-8")
  if old_text not in content:
    raise ValueError(f"Text not found in {path}")

  file_path.write_text(
    content.replace(old_text, new_text, 1), encoding="utf-8"
  )
  return f"Edited {path}"


@tool
def glob(pattern: str) -> str:
  """查找匹配 Glob 模式的路径，支持 ** 递归；调用方负责权限审批。

  Args:
      pattern: 相对于工作目录或绝对路径的模式，例如 **/*.py。

  Returns:
      排序去重后的路径，最多 200 条；无匹配时为 "(no matches)"。
  """
  matches = sorted(set(
    glob_paths(pattern, root_dir=WORKDIR, recursive=True)
  ))
  shown = matches[:200]
  if len(matches) > 200:
    shown.append("... (more matches omitted; narrow the pattern)")
  return "\n".join(shown) if shown else "(no matches)"


TOOLS = [bash, read_file, write_file, edit_file, glob]
