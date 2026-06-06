"""TemplateRegistry: prompt templates for reusable workflows (/review, /explain, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TemplateRegistry:
    """Loads and expands prompt templates from .pi/prompts/*.md files."""

    def __init__(self, template_dirs: list[Path] | None = None) -> None:
        self._templates: dict[str, str] = {}
        if template_dirs:
            for d in template_dirs:
                self._discover(d)

    def _discover(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for f in sorted(directory.glob("*.md")):
            name = f.stem  # e.g. "review.md" → "review"
            self._templates[name] = f.read_text(encoding="utf-8")

    def expand(self, name: str, *args: str) -> str:
        """Expand a template by name with positional args.

        Args:
            name: Template name (e.g. "review", "debug")
            *args: Arguments that replace $1, $2, etc. $@ joins all args.
        """
        template = self._templates.get(name)
        if template is None:
            return f"Template '{name}' not found. Available: {', '.join(sorted(self._templates.keys()))}"

        result = template
        for i, arg in enumerate(args, start=1):
            result = result.replace(f"${i}", arg)
        result = result.replace("$@", " ".join(args))
        return result

    def list(self) -> list[str]:
        """List available template names."""
        return sorted(self._templates.keys())

    def has(self, name: str) -> bool:
        return name in self._templates

    @classmethod
    def discover(cls) -> TemplateRegistry:
        """Auto-discover templates from common locations."""
        dirs: list[Path] = []
        # .pi/prompts/ in cwd or project root
        for candidate in [Path(".pi/prompts"), Path(".logician/prompts")]:
            if candidate.exists():
                dirs.append(candidate)
        # User's home directory
        home = Path.home() / ".logician" / "prompts"
        if home.exists():
            dirs.append(home)
        return cls(dirs)
