"""验证技能扫描、缓存和路径边界。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lota.tools.skills import SkillLoader


class SkillLoaderTest(unittest.TestCase):
  """使用临时工作区验证 SKILL.md 的注册规则。"""

  def setUp(self):
    """创建临时工作区和技能目录。"""
    self.workdir = Path(self.enterContext(TemporaryDirectory()))
    self.skills = self.workdir / "skills"
    self.skills.mkdir()


  def write_skill(self, folder: str, content: str) -> Path:
    """写入一个测试技能。

    Args:
        folder: 技能目录名。
        content: 技能文件内容。

    Returns:
        写入的文件路径。
    """
    path = self.skills / folder / "SKILL.md"
    path.parent.mkdir()
    path.write_text(content, encoding="utf-8")
    return path


  def test_catalog_and_cached_content(self):
    """目录只显示元数据，工具返回启动时读取的全文。"""
    content = (
      "---\nname: review\ndescription: >\n"
      "  审查改动\n  并检查测试。\n---\n\n# 正文\n"
    )
    path = self.write_skill("review-folder", content)
    loader = SkillLoader(self.workdir)
    path.write_text("changed", encoding="utf-8")

    self.assertEqual(loader.catalog(), "- review: 审查改动 并检查测试。")
    self.assertEqual(loader.load_skill("review"), content)


  def test_invalid_manifests_are_skipped(self):
    """缺少元数据、语法错误或非 UTF-8 的文件不影响可用技能。"""
    self.write_skill("a-no-header", "# 没有 YAML 头部\n")
    self.write_skill("b-invalid-yaml", "---\nname: [\n---\n")
    self.write_skill(
      "c-empty-description", '---\nname: missing\ndescription: ""\n---\n'
    )
    binary = self.write_skill("d-binary", "")
    binary.write_bytes(b"\xff")
    self.write_skill("e-valid", "---\nname: review\ndescription: 审查\n---\n")

    self.assertEqual(SkillLoader(self.workdir).catalog(), "- review: 审查")


  def test_duplicate_skill_keeps_first_and_warns(self):
    """同名技能不会覆盖按路径排序最先注册的版本。"""
    first = "---\nname: review\ndescription: 第一份\n---\n"
    self.write_skill("b-review", "---\nname: review\ndescription: 第二份\n---\n")
    self.write_skill("a-review", first)

    with self.assertLogs("lota.tools.skills", level="WARNING") as logs:
      loader = SkillLoader(self.workdir)

    self.assertIn("Ignoring duplicate skill 'review'", logs.output[0])
    self.assertEqual(loader.load_skill("review"), first)


  def test_external_symlinks_are_not_registered(self):
    """技能文件、子目录和技能根目录都不能通过链接越界读取。"""
    outside = Path(self.enterContext(TemporaryDirectory()))
    folder = outside / "external"
    folder.mkdir()
    file = folder / "SKILL.md"
    file.write_text(
      "---\nname: external\ndescription: 不应出现\n---\n", encoding="utf-8"
    )

    (self.skills / "linked-dir").symlink_to(folder, target_is_directory=True)
    linked_file = self.skills / "linked-file" / "SKILL.md"
    linked_file.parent.mkdir()
    linked_file.symlink_to(file)
    self.assertEqual(SkillLoader(self.workdir).catalog(), "")

    (self.skills / "linked-dir").unlink()
    linked_file.unlink()
    linked_file.parent.rmdir()
    self.skills.rmdir()
    self.skills.symlink_to(outside, target_is_directory=True)
    self.assertEqual(SkillLoader(self.workdir).catalog(), "")
