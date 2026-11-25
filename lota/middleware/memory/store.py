"""保存 Markdown 记忆、维护索引，并支持整库替换与回滚。"""

import logging
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, StringConstraints

INDEX_NAME = "MEMORY.md"
NAME_PATTERN = r"[A-Za-z0-9_]+"
INDEX_ROW = re.compile(
  rf"- \[({NAME_PATTERN})\]\(({NAME_PATTERN}\.md)\) - (.+)"
)
logger = logging.getLogger(__name__)


class MemoryRecord(BaseModel):
  """一条可跨会话保存的记忆。

  Attributes:
      name: 仅含英文字母、数字和下划线的记忆标识。
      description: 索引显示的简短介绍。
      type: 用户偏好、长期反馈、项目事实或外部参考。
      body: Markdown 正文。
  """

  model_config = ConfigDict(
    str_strip_whitespace=True, str_min_length=1, frozen=True,
  )

  name: Annotated[
    str, StringConstraints(strip_whitespace=False, pattern=f"^{NAME_PATTERN}$")
  ]
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
    """将 Markdown 文件名转换为记忆目录下的绝对路径。

    Args:
        filename: 单个以 .md 结尾的文件名，不含目录。

    Returns:
        对应的绝对路径，文件不必已存在。

    Raises:
        ValueError: 文件名包含路径或不以 .md 结尾。
        PermissionError: 记忆目录或目标文件是符号链接。
    """
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
    path = self._write_record(record)
    self.rebuild_index()
    return path


  def _write_record(self, record: MemoryRecord) -> Path:
    """创建记录文件，不覆盖已有文件，也不更新索引。

    Args:
        record: 要存盘的记忆。

    Returns:
        新记录的绝对路径。

    Raises:
        ValueError: 名称与索引文件冲突。
        OSError: 文件已存在、路径不安全或写入失败。
    """
    filename = f"{record.name.lower()}.md"
    if filename == INDEX_NAME.lower():
      raise ValueError(f"Invalid memory name: {record.name}")

    path = self._path(filename)
    header = yaml.safe_dump(
      record.model_dump(exclude={"body"}), allow_unicode=True, sort_keys=False,
    ).rstrip()
    self._root.mkdir(exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
      file.write(f"---\n{header}\n---\n\n{record.body}\n")
    return path


  def read(self, filename: str) -> MemoryRecord:
    """按文件名读取一条记忆，不接受索引文件。

    Args:
        filename: 记忆文件名，例如 python_style.md。

    Returns:
        校验后的元数据和正文。

    Raises:
        ValueError: 文件名、文件头或记录字段无效。
        OSError: 文件不存在、遇到符号链接或读取失败。
        yaml.YAMLError: 文件头的 YAML 无法解析。
    """
    if filename.lower() == INDEX_NAME.lower():
      raise ValueError("The memory index is not a memory record")

    path = self._path(filename)
    return parse_record(filename, path.read_text(encoding="utf-8"))


  def catalog(self) -> list[tuple[str, str]]:
    """从索引读取文件名和简介，不打开记忆正文。

    Returns:
        按索引顺序排列的 (文件名, 名称及简介)；索引不存在时为空。

    Raises:
        OSError: 索引无法读取或遇到符号链接。
        UnicodeError: 索引不是 UTF-8 文本。
    """
    index = self._path(INDEX_NAME)
    try:
      rows = index.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
      return []

    entries = []
    for row in rows:
      match = INDEX_ROW.fullmatch(row)
      if match is None:
        continue
      name, filename, description = match.groups()
      if filename.lower() == INDEX_NAME.lower():
        continue
      entries.append((filename, f"{name} - {description}"))
    return entries


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
      description = " ".join(record.description.split())
      rows.append(f"- [{record.name}]({path.name}) - {description}\n")

    with TemporaryDirectory(dir=self._root) as directory:
      staged = Path(directory) / INDEX_NAME
      staged.write_text("".join(rows), encoding="utf-8")
      staged.replace(index)


  def snapshot(self) -> dict[str, str]:
    """读取整个记忆目录，不跟随符号链接。

    Returns:
        文件名到原始文本的映射，包含索引（若存在）。

    Raises:
        OSError: 无法读取目录或文件，或遇到符号链接。
        ValueError: 有非 Markdown 文件、子目录或无效 UTF-8 文本。
    """
    self._path(INDEX_NAME)
    files = {}
    for entry in sorted(self._root.iterdir()):
      path = self._path(entry.name)
      if not path.is_file():
        raise ValueError(f"Not a memory file: {entry.name}")
      files[entry.name] = path.read_text(encoding="utf-8")
    return files


  def replace_records(
    self, snapshot: dict[str, str], records: list[MemoryRecord]
  ) -> None:
    """暂存完整新库，替换失败时还原旧目录。

    Args:
        snapshot: 整理前读取的记录和索引，用于检测期间的改动。
        records: 全部通过校验的替换记录。

    Raises:
        OSError: 文件读写或目录替换失败。
        ValueError: 旧目录有改动，或记录格式不合法。
    """
    staging = Path(mkdtemp(prefix=".memory-stage-", dir=self._root.parent))
    backup = staging / "backup"
    published = False
    try:
      staged_store = MemoryStore(staging)
      for record in records:
        staged_store._write_record(record)
      staged_store.rebuild_index()
      if self.snapshot() != snapshot:
        raise ValueError("Memory changed during consolidation")

      self._root.rename(backup)
      try:
        staged_store._root.rename(self._root)
      except OSError:
        backup.rename(self._root)
        raise
      published = True
    finally:
      if published or not backup.exists():
        shutil.rmtree(staging, ignore_errors=True)
      else:
        logger.warning("Memory backup retained at %s", backup)


def parse_record(filename: str, content: str) -> MemoryRecord:
  """将含 YAML 头部的 Markdown 文本解析为记忆记录。

  Args:
      filename: 文件名，用于报告格式错误。
      content: 完整的 Markdown 文本。

  Returns:
      通过校验的记忆记录。

  Raises:
      ValueError: 头部缺失或记录字段无效。
      yaml.YAMLError: 无法解析 YAML 头部。
  """
  header, separator, body = content.removeprefix("---\n").partition("\n---\n")
  if not content.startswith("---\n") or not separator:
    raise ValueError(f"Missing YAML frontmatter: {filename}")

  metadata = yaml.safe_load(header)
  if not isinstance(metadata, dict):
    raise ValueError(f"Invalid memory metadata: {filename}")
  return MemoryRecord.model_validate(metadata | {"body": body})
