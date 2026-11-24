"""在工作区保存跨会话记忆，并从记录生成 Markdown 索引。"""

import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

INDEX_NAME = "MEMORY.md"


class MemoryRecord(BaseModel):
  """一条可跨会话保存的记忆。

  Attributes:
      name: 记忆名称，用于生成文件名。
      description: 索引显示的简短介绍。
      type: 用户偏好、长期反馈、项目事实或外部参考。
      body: Markdown 正文。
  """

  model_config = ConfigDict(
    str_strip_whitespace=True, str_min_length=1, frozen=True,
  )

  name: str
  description: str
  type: Literal["user", "feedback", "project", "reference"]
  body: str


class MemoryStore:
  """管理工作区 .memory 目录中的记录和派生索引。"""

  def __init__(self, workdir: Path) -> None:
    """定位记忆目录，不创建文件。

    Args:
        workdir: 存放 .memory 的工作区。
    """
    self._root = workdir.resolve() / ".memory"


  def _path(self, filename: str) -> Path:
    """只接受记忆目录下不经过符号链接的 Markdown 文件名。"""
    name = Path(filename)
    if name.name != filename or name.suffix != ".md":
      raise ValueError(f"Invalid memory filename: {filename}")

    path = self._root / name
    if self._root.is_symlink() or path.is_symlink():
      raise PermissionError(f"Symlinks are not allowed in memory: {path}")
    return path


  def save(self, record: MemoryRecord) -> Path:
    """保存一条新记录并重建索引，不覆盖已有记录。

    索引重建失败时，已写入的记忆文件仍保留。

    Args:
        record: 已校验的记忆。

    Returns:
        新记录的绝对路径。

    Raises:
        ValueError: 名称或已有记录格式无效。
        OSError: 文件已存在、遇到符号链接或读写失败。
        yaml.YAMLError: 已有记录的 YAML 无法解析。
    """
    slug = re.sub(r"[^\w]+", "-", record.name.lower()).strip("-_")
    filename = f"{slug}.md"
    if not slug or filename == INDEX_NAME.lower():
      raise ValueError(f"Invalid memory name: {record.name}")

    path = self._path(filename)
    header = yaml.safe_dump(
      record.model_dump(exclude={"body"}), allow_unicode=True, sort_keys=False,
    ).rstrip()
    self._root.mkdir(exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
      file.write(f"---\n{header}\n---\n\n{record.body}\n")
    self.rebuild_index()
    return path


  def read(self, filename: str) -> MemoryRecord:
    """按文件名读取一条记忆，不接受索引文件。

    Args:
        filename: 记忆文件名，例如 python-style.md。

    Returns:
        校验后的元数据和正文。

    Raises:
        ValueError: 文件名、文件头或记录字段无效。
        OSError: 文件不存在、遇到符号链接或读取失败。
        yaml.YAMLError: 文件头的 YAML 无法解析。
    """
    if filename.lower() == INDEX_NAME.lower():
      raise ValueError("The memory index is not a memory record")

    content = self._path(filename).read_text(encoding="utf-8")
    header, separator, body = content.removeprefix("---\n").partition(
      "\n---\n"
    )
    if not content.startswith("---\n") or not separator:
      raise ValueError(f"Missing YAML frontmatter: {filename}")

    metadata = yaml.safe_load(header)
    if not isinstance(metadata, dict):
      raise ValueError(f"Invalid memory metadata: {filename}")
    return MemoryRecord.model_validate(metadata | {"body": body})


  def rebuild_index(self) -> None:
    """按文件名排序，从记录生成索引，不修改记录本身。

    Raises:
        ValueError: 记录格式无效。
        OSError: 遇到符号链接或读写失败。
        yaml.YAMLError: 记录的 YAML 无法解析。
    """
    index = self._path(INDEX_NAME)
    self._root.mkdir(exist_ok=True)
    rows = []
    for path in sorted(self._root.glob("*.md")):
      if path.name.lower() == INDEX_NAME.lower():
        continue
      record = self.read(path.name)
      name = " ".join(record.name.split())
      description = " ".join(record.description.split())
      rows.append(f"- [{name}]({path.name}) - {description}\n")

    with TemporaryDirectory(dir=self._root) as directory:
      staged = Path(directory) / INDEX_NAME
      staged.write_text("".join(rows), encoding="utf-8")
      staged.replace(index)
