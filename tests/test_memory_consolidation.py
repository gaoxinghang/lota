"""离线验证长期记忆的自动整理和恢复。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from fake_model import ToolCallingModel
from lota import workspace
from lota.agent import build_agent
from lota.middleware.memory import MemoryRecord, MemoryStore
from lota.middleware.memory.consolidation import (
  CONSOLIDATION_SYSTEM, consolidate_memories,
)

QUERY = "以后 Python 都用两个空格缩进。"
NEW_RECORD = {
  "name": "python_indent", "type": "user",
  "description": "Python 缩进偏好", "body": QUERY,
}


class MemoryConsolidationTest(unittest.TestCase):
  """验证整理只在符合条件时更新临时工作区。"""

  def setUp(self):
    """创建隔离的工作区和记忆存储。"""
    self.workdir = Path(self.enterContext(TemporaryDirectory())).resolve()
    self.enterContext(patch.object(workspace, "WORKDIR", self.workdir))
    self.store = MemoryStore(self.workdir)
    self.root = self.workdir / ".memory"


  def seed(self, count: int) -> list[dict[str, str]]:
    """写入指定数量的旧记录，并返回它们的字段。"""
    records = []
    for index in range(count):
      record = MemoryRecord(
        name=f"record_{index}", type="user",
        description=f"旧记录 {index}", body=f"偏好 {index}",
      )
      self.store.save(record)
      records.append(record.model_dump())
    return records


  def snapshot(self) -> dict[str, bytes]:
    """读取当前目录的所有文件，以便逐字节比较。"""
    return {path.name: path.read_bytes() for path in self.root.iterdir()}


  def test_new_record_at_threshold_triggers_consolidation(self):
    """第十条保存后，整理合并同义项并重建可召回的索引。"""
    originals = self.seed(8)
    self.store.save(MemoryRecord(
      name="old_indent", type="user", description="代码格式",
      body="Python 缩进宽度为 2。",
    ))
    model = ToolCallingModel(
      responses=[
        AIMessage(content="[]"), AIMessage(content="好的。"),
        AIMessage(content=json.dumps([*originals, NEW_RECORD])),
      ],
      extraction_reply=json.dumps([NEW_RECORD | {
        "scope": "persistent", "evidence": QUERY,
      }]),
    )
    result = build_agent(model).invoke({"messages": [
      HumanMessage(content=QUERY),
    ]})
    self.assertEqual(result["messages"][-1].text, "好的。")
    self.assertEqual(len(model.calls), 3)
    self.assertEqual(model.calls[2][0].text, CONSOLIDATION_SYSTEM)
    self.assertEqual(len(json.loads(model.calls[2][-1].text)), 10)
    catalog = [filename for filename, _ in self.store.catalog()]
    self.assertEqual(len(catalog), 9)
    self.assertIn("python_indent.md", catalog)
    self.assertFalse((self.root / "old_indent.md").exists())

    model = ToolCallingModel(responses=[
      AIMessage(content="[0]"), AIMessage(content="两个空格。"),
    ])
    build_agent(model).invoke({"messages": [HumanMessage(content="缩进？")]})
    self.assertIn(QUERY, model.calls[-1][0].text)


  def test_no_new_record_or_below_threshold_skips_model(self):
    """不足十条或只有重复候选时都不请求整理模型。"""
    self.seed(8)
    for name in ("python_indent", "another_indent"):
      model = ToolCallingModel(
        responses=[AIMessage(content="[]"), AIMessage(content="好的。")],
        extraction_reply=json.dumps([NEW_RECORD | {
          "name": name, "scope": "persistent", "evidence": QUERY,
        }]),
      )
      build_agent(model).invoke({"messages": [HumanMessage(content=QUERY)]})
      self.assertEqual(len(model.calls), 2)
      if name == "python_indent":
        self.assertEqual(len(self.store.catalog()), 9)
        self.store.save(MemoryRecord(
          name="tenth", type="project", description="额外记录", body="正文",
        ))
    self.assertEqual(len(self.store.catalog()), 10)


  def test_invalid_result_keeps_entire_store(self):
    """空结果、一条坏记录或文件名冲突都不会部分替换。"""
    originals = self.seed(10)
    old = self.snapshot()
    for items in (
      [], [originals[0], {"name": "broken"}],
      [originals[0], originals[0] | {"name": "RECORD_0"}],
      [originals[0] | {"name": "memory"}],
    ):
      with self.subTest(items=items):
        model = ToolCallingModel(responses=[
          AIMessage(content=json.dumps(items)),
        ])
        with self.assertLogs("lota.middleware.memory", level="WARNING"):
          consolidate_memories(model, self.store)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(self.snapshot(), old)


  def test_oversized_input_is_not_sent(self):
    """超过预算的整库不会截断后再交给模型。"""
    self.seed(9)
    self.store.save(MemoryRecord(
      name="large", type="project", description="长记录", body="甲" * 20_000,
    ))
    old = self.snapshot()
    model = ToolCallingModel(responses=[])
    consolidate_memories(model, self.store)
    self.assertEqual(model.calls, [])
    self.assertEqual(self.snapshot(), old)


  def test_store_changed_during_consolidation_is_not_replaced(self):
    """模型整理期间出现的新记录不会被旧方案覆盖。"""
    originals = self.seed(10)
    model = ToolCallingModel(responses=[
      AIMessage(content=json.dumps(originals[:1])),
    ])
    invoke = model.invoke

    def save_while_model_runs(*args, **kwargs):
      self.store.save(MemoryRecord.model_validate(NEW_RECORD))
      return invoke(*args, **kwargs)

    with (
      patch.object(type(model), "invoke", side_effect=save_while_model_runs),
      self.assertLogs("lota.middleware.memory", level="WARNING"),
    ):
      consolidate_memories(model, self.store)
    self.assertEqual(len(self.store.catalog()), 11)
    self.assertEqual(self.store.read("python_indent.md").body, QUERY)


  def test_failed_publish_restores_old_records_and_index(self):
    """旧目录移走后，发布新库失败会还原原文件和索引。"""
    originals = self.seed(10)
    old = self.snapshot()
    model = ToolCallingModel(responses=[
      AIMessage(content=json.dumps(originals[:1])),
    ])
    rename = Path.rename

    def fail_publish(path, target):
      if path.name == ".memory" and target == self.root:
        self.assertFalse(self.root.exists())
        raise OSError("模拟发布失败")
      return rename(path, target)

    with (
      patch.object(Path, "rename", fail_publish),
      self.assertLogs("lota.middleware.memory", level="WARNING"),
    ):
      consolidate_memories(model, self.store)
    self.assertEqual(len(model.calls), 1)
    self.assertEqual(self.snapshot(), old)
    self.assertEqual(list(self.workdir.glob(".memory-stage-*")), [])
