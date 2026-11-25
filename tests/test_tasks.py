"""验证持久化任务的状态变更、依赖规则和存储路径。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from lota.tools.tasks import TaskStore


class TaskStoreTest(unittest.TestCase):
  """只在临时工作区测试 s10 任务系统。"""

  def setUp(self):
    """构建任务存储，不提前创建任务目录。"""
    self.workdir = Path(self.enterContext(TemporaryDirectory()))
    self.directory = self.workdir / ".tasks"
    self.store = TaskStore(self.workdir)


  def create(self, subject: str, description: str = "") -> str:
    """创建任务并从返回值提取 ID。"""
    return self.store.create_task(subject, description).split()[1].rstrip(":")


  def test_persist_across_store_instances(self):
    """新建存储实例后仍可读取完整记录并完成已认领的任务。"""
    self.assertIn("No tasks", self.store.list_tasks())
    self.assertFalse(self.directory.exists())
    task_id = self.create("  设计接口  ", "写清输入和输出")
    self.assertRegex(task_id, r"^task_[0-9a-f]{8}$")
    self.store.claim_task(task_id)

    reopened = TaskStore(self.workdir)
    record = {
      "id": task_id,
      "subject": "设计接口",
      "description": "写清输入和输出",
      "status": "in_progress",
      "owner": "agent",
      "blocked_by": [],
    }
    path = self.directory / f"{task_id}.json"
    self.assertEqual(json.loads(path.read_text(encoding="utf-8")), record)
    self.assertEqual(json.loads(reopened.get_task(task_id)), record)
    self.assertIn(f"{task_id}: 设计接口 [in_progress]", reopened.list_tasks())

    reopened.complete_task(task_id)
    restored = json.loads(TaskStore(self.workdir).get_task(task_id))
    self.assertEqual(restored["status"], "completed")


  def test_claim_requires_all_dependencies_and_completion_reports_unlocks(self):
    """只报告刚解锁的直接下游，其他未满足的依赖仍阻止认领。"""
    design = self.create("设计")
    inputs = self.create("准备数据")
    build = self.create("实现")
    tests = self.create("测试")
    docs = self.create("文档")
    self.store.update_task(build, [design, inputs])
    self.store.update_task(tests, [build])
    self.store.update_task(docs, [design])

    with self.assertRaisesRegex(ValueError, "Blocked by"):
      self.store.claim_task(build)

    self.store.claim_task(design)
    result = self.store.complete_task(design)
    self.assertIn(f"Unblocked: {docs}: 文档", result)
    self.assertNotIn(build, result)
    with self.assertRaisesRegex(ValueError, inputs):
      self.store.claim_task(build)

    self.store.claim_task(inputs)
    result = self.store.complete_task(inputs)
    self.assertIn(f"Unblocked: {build}: 实现", result)
    self.assertNotIn(docs, result)
    self.assertNotIn(tests, result)
    self.store.claim_task(build)
    self.assertIn(f"Unblocked: {tests}: 测试", self.store.complete_task(build))
    self.assertIn("Claimed", self.store.claim_task(tests))


  def test_update_deduplicates_and_rejects_invalid_graph_changes(self):
    """重复依赖不增加边，失败的依赖更新不保存任何边。"""
    first = self.create("first")
    second = self.create("second")
    third = self.create("third")
    independent = self.create("independent")
    missing = self.create("missing")
    (self.directory / f"{missing}.json").unlink()
    self.store.update_task(second, [first, first])
    self.store.update_task(second, [first])
    self.store.update_task(third, [second])

    with self.assertRaises(FileNotFoundError):
      self.store.update_task(first, [independent, missing])
    with self.assertRaisesRegex(ValueError, "depend on itself"):
      self.store.update_task(first, [first])
    with self.assertRaisesRegex(ValueError, "Dependency cycle"):
      self.store.update_task(first, [third])

    self.assertEqual(json.loads(self.store.get_task(first))["blocked_by"], [])
    self.assertEqual(
      json.loads(self.store.get_task(second))["blocked_by"], [first],
    )


  def test_state_and_owner_checks(self):
    """不允许跳过认领、重复认领、更新已开始的任务或替别人完成。"""
    first = self.create("first")
    second = self.create("second")
    with self.assertRaisesRegex(ValueError, "in_progress"):
      self.store.complete_task(first)

    self.store.claim_task(first)
    with self.assertRaisesRegex(ValueError, "pending and unowned"):
      self.store.claim_task(first)
    with self.assertRaisesRegex(ValueError, "pending and unowned"):
      self.store.update_task(first, [second])

    record = json.loads(self.store.get_task(first))
    record["owner"] = "other"
    path = self.directory / f"{first}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with self.assertRaisesRegex(ValueError, "owned by agent"):
      self.store.complete_task(first)
    self.assertEqual(json.loads(self.store.get_task(first)), record)


  def test_missing_dependency_blocks_claim(self):
    """已从磁盘删除的依赖不会被视为完成。"""
    dependency = self.create("dependency")
    target = self.create("target")
    self.store.update_task(target, [dependency])
    (self.directory / f"{dependency}.json").unlink()
    with self.assertRaisesRegex(ValueError, f"Blocked by: {dependency}"):
      self.store.claim_task(target)


  def test_empty_subject_does_not_create_directory(self):
    """无标题的任务不会写盘。"""
    with self.assertRaises(ValueError):
      self.create("   ")
    self.assertFalse(self.directory.exists())


  def test_id_collision_does_not_overwrite(self):
    """两次生成相同 ID 时重新分配，原任务不变。"""
    with patch(
      "lota.tools.tasks.token_hex",
      side_effect=["deadbeef", "deadbeef", "cafebabe"],
    ):
      first = self.create("first")
      second = self.create("second")
    self.assertEqual((first, second), ("task_deadbeef", "task_cafebabe"))
    self.assertEqual(json.loads(self.store.get_task(first))["subject"], "first")


  def test_rejects_invalid_paths_and_symlinks(self):
    """任务 ID 不能当路径使用，任务目录和文件也不跟随符号链接。"""
    outside = Path(self.enterContext(TemporaryDirectory()))
    with self.assertRaisesRegex(ValueError, "Invalid task ID"):
      self.store.get_task("../outside")

    self.directory.symlink_to(outside, target_is_directory=True)
    with self.assertRaises(PermissionError):
      self.create("cannot follow directory")
    self.assertEqual(list(outside.iterdir()), [])

    self.directory.unlink()
    self.directory.mkdir()
    external = outside / "task_deadbeef.json"
    external.write_text("outside", encoding="utf-8")
    (self.directory / external.name).symlink_to(external)
    with self.assertRaises(PermissionError):
      self.store.get_task(external.stem)
