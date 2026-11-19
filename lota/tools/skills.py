"""扫描工作区技能，并按名称提供完整的 SKILL.md。"""

import logging
from pathlib import Path

import yaml
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)


def _parse_metadata(content: str) -> tuple[str, str] | None:
  """提取 YAML 头部中的非空技能名称和描述。

  Args:
      content: SKILL.md 的完整文本。

  Returns:
      技能名称和简介；头部无效时返回 None。
  """
  lines = content.splitlines()
  if not lines or lines[0] != "---":
    return None

  try:
    end = lines.index("---", 1)
    metadata = yaml.safe_load("\n".join(lines[1:end]))
  except (ValueError, yaml.YAMLError):
    return None

  if not isinstance(metadata, dict):
    return None
  name = metadata.get("name")
  description = metadata.get("description")
  if not isinstance(name, str) or not isinstance(description, str):
    return None
  name = name.strip()
  description = " ".join(description.split())
  return (name, description) if name and description else None


class SkillLoader:
  """保存 Agent 启动时扫描到的技能目录和原文。"""

  def __init__(self, workdir: Path) -> None:
    """从工作区的 skills 目录建立本次会话的技能快照。

    Args:
        workdir: Agent 的工作目录。
    """
    self._skills: dict[str, tuple[str, str]] = {}
    try:
      workdir = workdir.resolve()
      skills_root = (workdir / "skills").resolve()
    except (OSError, RuntimeError):
      return
    if not skills_root.is_relative_to(workdir):
      return

    for manifest in sorted(skills_root.glob("*/SKILL.md")):
      try:
        path = manifest.resolve()
        if not path.is_relative_to(skills_root) or not path.is_file():
          continue
        content = path.read_text(encoding="utf-8")
      except (OSError, RuntimeError, UnicodeError):
        continue

      metadata = _parse_metadata(content)
      if metadata is None:
        continue
      name, description = metadata
      if name in self._skills:
        logger.warning("Ignoring duplicate skill %r at %s", name, manifest)
        continue
      self._skills[name] = (description, content)


  def catalog(self) -> str:
    """渲染只包含技能名称和简介的目录。

    Returns:
        每行一项的技能列表；没有技能时为空字符串。
    """
    return "\n".join(
      f"- {name}: {description}"
      for name, (description, _) in self._skills.items()
    )


  def load_skill(self, name: str) -> str:
    """按目录名称加载完整技能说明，不执行其中的脚本。

    Args:
        name: 技能目录里的名称，不是文件路径。

    Returns:
        完整的 SKILL.md 文本，包括 YAML 头部。

    Raises:
        ValueError: 技能未注册。
    """
    name = name.strip()
    if name not in self._skills:
      available = ", ".join(self._skills) or "none"
      raise ValueError(f"Unknown skill {name!r}. Available: {available}")
    return self._skills[name][1]


def build_skill_tool(workdir: Path) -> tuple[BaseTool | None, str]:
  """从工作区构建技能工具和对应的提示词片段。

  Args:
      workdir: Agent 的工作目录。

  Returns:
      load_skill 工具和提示词片段；没有可用技能时返回 (None, "")。
  """
  loader = SkillLoader(workdir)
  catalog = loader.catalog()
  if not catalog:
    return None, ""

  prompt = (
    f"\n\n可用技能：\n{catalog}\n"
    "任务适用某项技能时，先用 `load_skill` 读取完整说明。"
  )
  return tool(loader.load_skill), prompt
