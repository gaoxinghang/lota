"""验证记忆记录、索引和工作区路径边界。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from lota.middleware.memory import MemoryRecord, MemoryStore


class MemoryStoreTest(unittest.TestCase):
  """在临时工作区测试本地记忆存储，不调用模型。"""

  def setUp(self):
    """创建临时工作区和一条测试记忆。"""
    self.workdir = Path(self.enterContext(TemporaryDirectory())).resolve()
    self.root = self.workdir / ".memory"
    self.store = MemoryStore(self.workdir)
    self.record = MemoryRecord(
      name="Python_Style2",
      description="缩进偏好",
      type="user",
      body="使用 2 个空格。\n\n---\n\n适用于 Python。",
    )


  def test_save_and_read_in_new_store(self):
    """新实例能从 UTF-8 Markdown 读回完整记录。"""
    self.assertFalse(self.root.exists())
    path = self.store.save(self.record)

    self.assertEqual(path, self.root / "python_style2.md")
    self.assertEqual(path.read_text(encoding="utf-8"), (
      "---\nname: Python_Style2\ndescription: 缩进偏好\ntype: user\n---\n\n"
      "使用 2 个空格。\n\n---\n\n适用于 Python。\n"
    ))
    self.assertEqual(MemoryStore(self.workdir).read(path.name), self.record)


  def test_rebuild_index_from_records(self):
    """索引按文件名排序，可重建且不包含正文。"""
    self.store.save(self.record)
    self.store.save(MemoryRecord(
      name="Agent_docs", description="项目文档\n入口", type="reference",
      body="README.md",
    ))
    index = self.root / "MEMORY.md"
    expected = (
      "- [Agent_docs](agent_docs.md) - 项目文档 入口\n"
      "- [Python_Style2](python_style2.md) - 缩进偏好\n"
    )
    self.assertEqual(index.read_text(encoding="utf-8"), expected)

    index.unlink()
    self.store.rebuild_index()
    self.assertEqual(index.read_text(encoding="utf-8"), expected)


  def test_duplicate_and_reserved_names_do_not_overwrite(self):
    """已有记录和索引不能被当成新记忆覆盖。"""
    path = self.store.save(self.record)
    with self.assertRaises(FileExistsError):
      self.store.save(MemoryRecord(
        name="python_style2", description="新说明", type="user",
        body="新正文",
      ))
    self.assertEqual(self.store.read(path.name), self.record)
    with self.assertRaises(ValueError):
      self.store.save(MemoryRecord(
        name="memory", description="保留名称", type="project",
        body="正文",
      ))
    with self.assertRaises(ValueError):
      self.store.read("MEMORY.md")


  def test_name_only_accepts_ascii_letters_digits_and_underscores(self):
    """不接受中文、空白、连字符和需要转义的符号。"""
    for name in ("", "中文", "two words", " x ", "x\n", "x-y", "[x]", r"a\b"):
      with self.subTest(name=name), self.assertRaises(ValidationError):
        MemoryRecord(name=name, description="简介", type="user", body="正文")


  def test_invalid_record_is_rejected(self):
    """损坏的记录不会按有效记忆读取。"""
    self.root.mkdir()
    path = self.root / "invalid.md"
    path.write_text(
      "---\nname: Invalid\ndescription: 说明\ntype: unknown\n---\n正文\n",
      encoding="utf-8",
    )
    with self.assertRaises(ValidationError):
      self.store.read(path.name)


  def test_read_rejects_paths_and_symlinks(self):
    """只允许读取记忆目录内不经过符号链接的普通路径。"""
    outside = Path(self.enterContext(TemporaryDirectory())) / "outside.md"
    outside.write_text("保持原样", encoding="utf-8")
    self.root.mkdir()
    link = self.root / "python_style2.md"
    link.symlink_to(outside)

    for filename in ("../outside.md", str(outside)):
      with self.subTest(filename=filename), self.assertRaises(ValueError):
        self.store.read(filename)
    with self.assertRaises(PermissionError):
      self.store.read(link.name)
    with self.assertRaises(PermissionError):
      self.store.save(self.record)
    self.assertEqual(outside.read_text(encoding="utf-8"), "保持原样")


  def test_index_and_root_symlinks_are_rejected(self):
    """索引链接和目录链接都不能把写入引到外面。"""
    outside = Path(self.enterContext(TemporaryDirectory()))
    external_index = outside / "MEMORY.md"
    external_index.write_text("保持原样", encoding="utf-8")
    self.root.mkdir()
    index = self.root / "MEMORY.md"
    index.symlink_to(external_index)
    with self.assertRaises(PermissionError):
      self.store.rebuild_index()

    index.unlink()
    self.root.rmdir()
    self.root.symlink_to(outside, target_is_directory=True)
    with self.assertRaises(PermissionError):
      self.store.read("python_style2.md")
    with self.assertRaises(PermissionError):
      self.store.save(self.record)
    self.assertEqual(list(outside.iterdir()), [external_index])
    self.assertEqual(external_index.read_text(encoding="utf-8"), "保持原样")
