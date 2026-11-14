"""在临时工作目录中验证文件工具。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from lota import tools


class FileToolsTest(unittest.TestCase):
  """覆盖文件操作、截断行为和工作目录边界。"""

  def setUp(self):
    """创建隔离的工作目录与外部目录。"""
    temporary = TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name).resolve()
    self.workspace = root / "workspace"
    self.workspace.mkdir()
    self.outside = root / "outside"
    self.outside.mkdir()
    self.enterContext(patch.object(tools, "WORKDIR", self.workspace))


  def test_write_and_read(self):
    """写入时创建父目录，支持中文与覆盖已有内容。"""
    result = tools.write_file.invoke({
      "path": "nested/demo.txt", "content": "你好\n世界",
    })
    self.assertEqual(result, "Wrote 5 characters to nested/demo.txt")
    self.assertEqual(
      tools.read_file.invoke({"path": "nested/demo.txt"}), "你好\n世界"
    )

    tools.write_file.invoke({"path": "nested/demo.txt", "content": ""})
    self.assertEqual(
      tools.read_file.invoke({"path": "nested/demo.txt"}), ""
    )


  def test_read_limit(self):
    """读取行数受限时标注剩余行数，拒绝非正数。"""
    (self.workspace / "demo.txt").write_text(
      "one\ntwo\nthree\n", encoding="utf-8"
    )
    for limit, expected in [
      (1, "one\n... (2 more lines)"),
      (3, "one\ntwo\nthree"),
      (4, "one\ntwo\nthree"),
    ]:
      with self.subTest(limit=limit):
        self.assertEqual(
          tools.read_file.invoke({"path": "demo.txt", "limit": limit}),
          expected,
        )

    for limit in [0, -1]:
      with self.subTest(limit=limit), self.assertRaises(ValueError):
        tools.read_file.invoke({"path": "demo.txt", "limit": limit})


  def test_edit_first_match(self):
    """编辑只替换第一处匹配，也允许删除文本。"""
    file_path = self.workspace / "demo.txt"
    file_path.write_text("hello hello", encoding="utf-8")
    tools.edit_file.invoke({
      "path": "demo.txt", "old_text": "hello", "new_text": "你好",
    })
    self.assertEqual(file_path.read_text(encoding="utf-8"), "你好 hello")

    tools.edit_file.invoke({
      "path": "demo.txt", "old_text": "你好 ", "new_text": "",
    })
    self.assertEqual(file_path.read_text(encoding="utf-8"), "hello")


  def test_invalid_edit_keeps_file(self):
    """空原文与未匹配原文报错，不改变文件。"""
    file_path = self.workspace / "demo.txt"
    file_path.write_text("hello", encoding="utf-8")
    for old_text in ["", "missing"]:
      with self.subTest(old_text=old_text):
        with self.assertRaises(ValueError):
          tools.edit_file.invoke({
            "path": "demo.txt", "old_text": old_text, "new_text": "new",
          })
        self.assertEqual(file_path.read_text(encoding="utf-8"), "hello")


  def test_missing_and_non_text_files(self):
    """不存在的文件与非 UTF-8 文件抛出可回传的错误。"""
    with self.assertRaises(FileNotFoundError):
      tools.read_file.invoke({"path": "missing.txt"})

    (self.workspace / "binary").write_bytes(b"\xff")
    with self.assertRaises(UnicodeError):
      tools.read_file.invoke({"path": "binary"})


  def test_paths_outside_workspace(self):
    """所有文件工具拒绝相对越界、绝对越界与外部符号链接。"""
    target = self.outside / "demo.txt"
    target.write_text("unchanged", encoding="utf-8")
    (self.workspace / "link").symlink_to(
      self.outside, target_is_directory=True
    )
    for path in ["../outside/demo.txt", str(target), "link/demo.txt"]:
      calls = [
        (tools.read_file, {"path": path}),
        (tools.write_file, {"path": path, "content": "changed"}),
        (tools.edit_file, {
          "path": path, "old_text": "unchanged", "new_text": "changed",
        }),
        (tools.glob, {"pattern": path}),
      ]
      for handler, args in calls:
        with self.subTest(path=path, tool=handler.name):
          with self.assertRaises(ValueError):
            handler.invoke(args)
    self.assertEqual(target.read_text(encoding="utf-8"), "unchanged")


  def test_internal_symlink(self):
    """工作目录内的绝对路径和内部符号链接可以正常读取。"""
    target = self.workspace / "demo.txt"
    target.write_text("hello", encoding="utf-8")
    (self.workspace / "link.txt").symlink_to(target)
    for path in [str(target), "link.txt"]:
      with self.subTest(path=path):
        self.assertEqual(tools.read_file.invoke({"path": path}), "hello")


  def test_glob_recursive(self):
    """递归结果排序去重，并过滤指向外部文件的符号链接。"""
    (self.workspace / "nested").mkdir()
    for path in ["z.py", "a.py", "nested/b.py"]:
      (self.workspace / path).touch()
    outside_file = self.outside / "outside.py"
    outside_file.touch()
    (self.workspace / "outside.py").symlink_to(outside_file)
    self.assertEqual(
      tools.glob.invoke({"pattern": "**/**/*.py"}),
      "a.py\nnested/b.py\nz.py",
    )
    self.assertEqual(
      tools.glob.invoke({"pattern": "*.missing"}), "(no matches)"
    )


  def test_glob_limit(self):
    """匹配超过 200 条时截断并提示收窄模式。"""
    for index in range(201):
      (self.workspace / f"{index:03}.txt").touch()
    lines = tools.glob.invoke({"pattern": "*.txt"}).splitlines()
    self.assertEqual(len(lines), 201)
    self.assertEqual(lines[0], "000.txt")
    self.assertEqual(lines[199], "199.txt")
    self.assertIn("more matches omitted", lines[-1])
