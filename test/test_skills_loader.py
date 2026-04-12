from __future__ import annotations

from pathlib import Path

from src.skills.loader import find_skill_markdown_files, load_skills_from_dir


def test_find_and_load_skills(tmp_path: Path) -> None:
    # Create nested skills: category/skill/SKILL.md
    skill_dir = tmp_path / "category" / "my_skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("""---
name: my-skill
description: Test skill
paths:
  - src/**
---
# Skill body
Run this skill.
""")

    files = find_skill_markdown_files(str(tmp_path))
    assert len(files) == 1
    skills = load_skills_from_dir(str(tmp_path))
    assert len(skills) == 1
    s = skills[0]
    assert s.name == "my-skill"
    assert "Run this skill" in s.body
