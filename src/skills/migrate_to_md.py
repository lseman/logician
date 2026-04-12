from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Optional


def _extract_skill_dict_from_py(path: Path) -> Optional[Dict[str, Any]]:
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return None
    # naive search for __skill__ assignment
    idx = src.find("__skill__")
    if idx == -1:
        return None
    try:
        module = ast.parse(src)
    except Exception:
        return None
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__skill__":
                    try:
                        val = ast.literal_eval(node.value)
                        if isinstance(val, dict):
                            return val
                    except Exception:
                        return None
    return None


def _read_readme_or_doc(path: Path) -> str:
    # prefer README.md
    readme = path / "README.md"
    if readme.exists():
        try:
            return readme.read_text(encoding="utf-8")
        except Exception:
            pass
    # otherwise look for a module .py and its module docstring
    for f in path.iterdir():
        if f.suffix == ".py":
            try:
                src = f.read_text(encoding="utf-8")
                module = ast.parse(src)
                doc = ast.get_docstring(module)
                if doc:
                    return doc
            except Exception:
                pass
    return ""


def _write_skill_md(dest: Path, meta: Dict[str, Any], body: str) -> None:
    # Prefer PyYAML if available for nicer formatting
    try:
        import yaml  # type: ignore

        fm = dict(meta)
        if "name" not in fm:
            fm["name"] = meta.get("name") or dest.parent.name
        raw = yaml.safe_dump(fm, sort_keys=False)
        content = f"---\n{raw}---\n\n{body or ''}"
        dest.write_text(content, encoding="utf-8")
        return
    except Exception:
        pass

    fm_lines = ["---"]
    name = meta.get("name") or meta.get("__name__") or dest.parent.name
    fm_lines.append(f"name: {name}")
    desc = meta.get("description") or ""
    if isinstance(desc, str) and desc:
        fm_lines.append(f"description: {desc}")
    # Generic dump for common iterable fields
    for key in ("aliases", "triggers", "preferred_tools", "example_queries", "when_not_to_use"):
        val = meta.get(key)
        if val:
            fm_lines.append(f"{key}:")
            if isinstance(val, list):
                for item in val:
                    fm_lines.append(f"  - {item}")
            else:
                fm_lines.append(f"  - {val}")
    fm_lines.append("---\n")
    content = "\n".join(fm_lines) + (body or "")
    dest.write_text(content, encoding="utf-8")


def migrate_skills(root: Path, dry_run: bool = False) -> None:
    root = root.resolve()
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        target = sub / "SKILL.md"
        # try to find metadata
        py_file = None
        meta = None
        for f in sub.iterdir():
            if f.suffix == ".py":
                m = _extract_skill_dict_from_py(f)
                if m:
                    meta = m
                    py_file = f
                    break

        body = _read_readme_or_doc(sub)
        if not meta and not body:
            # nothing to migrate
            continue

        if dry_run:
            print(f"Would create {target}")
            continue

        _write_skill_md(target, meta or {}, body)
        print(f"Created {target}")

        # Move the implementation into scripts/ and add shim at original location
        if py_file:
            scripts_dir = sub / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            new_path = scripts_dir / py_file.name
            try:
                py_file.replace(new_path)
            except Exception:
                try:
                    # fallback to copy
                    new_path.write_bytes(py_file.read_bytes())
                    py_file.unlink()
                except Exception:
                    print(f"Failed to move {py_file} to {new_path}")
                    continue

            # write shim
            shim = sub / py_file.name
            shim_contents = f"from .scripts import {py_file.stem} as _mod\nfor _k, _v in vars(_mod).items():\n    if not _k.startswith('_'):\n        globals()[_k] = _v\n"
            try:
                shim.write_text(shim_contents, encoding="utf-8")
            except Exception:
                print(f"Failed to write shim for {shim}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent / "skills"
    # migrate coding and academic
    for d in (base / "coding", base / "academic"):
        if d.exists():
            migrate_skills(d)
    print("Migration complete.")
