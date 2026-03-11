"""Code exploration tools — structural understanding without reading full files.

These tools are the first stop when working with an unfamiliar codebase.
Use them BEFORE read_file to avoid reading thousands of lines unnecessarily.

Tool inventory
--------------
get_file_outline  -- AST-based: list all imports, classes, functions, and line numbers
find_symbol       -- locate every definition (def/class) matching a name across the codebase
get_project_map   -- one-line summary of important code/config/doc files in a directory
rg_search         -- fast ripgrep text search with context lines (falls back to pure Python)
fd_find           -- fast file/directory finder by name pattern (falls back to pure Python glob)
"""

from __future__ import annotations

import json as _json_mod
from typing import Any, Literal


if "_safe_json" not in globals():

    def _safe_json(obj: Any) -> str:  # type: ignore[misc]
        try:
            return _json_mod.dumps(obj, ensure_ascii=False)
        except Exception:
            return _json_mod.dumps({"status": "error", "error": repr(obj)})


import ast
import re
import shutil
import subprocess
import time
from pathlib import Path
from skills.coding.bootstrap.runtime_access import get_coding_runtime, tool

__skill__ = {
    "name": "Explore",
    "description": "Use for codebase exploration, search, outlines, symbol lookup, and structural inspection.",
    "aliases": ["code search", "codebase map", "symbol search", "inspect structure"],
    "triggers": [
        "find where this is defined",
        "search the codebase",
        "inspect the project structure",
        "find references to this symbol",
    ],
    "preferred_tools": ["get_file_outline", "rg_search", "fd_find"],
    "example_queries": [
        "where is this class defined",
        "find references to this helper",
        "map the project structure before editing",
    ],
    "when_not_to_use": ["the exact file and edit location are already known"],
    "next_skills": ["file_ops", "edit_block", "multi_edit", "search_replace"],
    "workflow": [
        "Start here when the task touches unfamiliar code.",
        "Prefer narrow search first, broader map second.",
        "Hand off to file_ops, edit_block, or multi_edit once the target is clear.",
    ],
}

_PROJECT_MAP_CACHE: dict[tuple[str, int, str], dict[str, Any]] = {}
_PROJECT_MAP_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".rs",
    ".go",
    ".java",
    ".kt",
    ".rb",
    ".php",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".swift",
}
_PROJECT_MAP_CONFIG_FILENAMES = {
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "cargo.toml",
    "cargo.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "poetry.lock",
    "pdm.lock",
    "go.mod",
    "go.sum",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    "makefile",
    "justfile",
    ".env",
    ".env.example",
}
_PROJECT_MAP_DOC_FILENAMES = {
    "readme",
    "readme.md",
    "contributing.md",
    "architecture.md",
    "design.md",
    "skills.md",
    "skill.md",
    "soul.md",
}
_PROJECT_MAP_LANGUAGE_LABELS = {
    "py": "Python",
    "js": "JavaScript",
    "jsx": "React JSX",
    "ts": "TypeScript",
    "tsx": "React TSX",
    "rs": "Rust",
    "go": "Go",
    "java": "Java",
    "kt": "Kotlin",
    "rb": "Ruby",
    "php": "PHP",
    "c": "C",
    "cc": "C++",
    "cpp": "C++",
    "h": "C/C++ header",
    "hpp": "C++ header",
    "cs": "C#",
    "swift": "Swift",
    "toml": "TOML",
    "json": "JSON",
    "yaml": "YAML",
    "yml": "YAML",
    "md": "Markdown",
    "text": "Text",
    "dockerfile": "Docker",
    "makefile": "Make",
}
_SOURCE_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "js",
    ".jsx": "jsx",
    ".ts": "ts",
    ".tsx": "tsx",
    ".rs": "rs",
    ".go": "go",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_exclude_paths(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []

    seen: set[str] = set()
    excludes: list[str] = []
    for line in text.splitlines():
        for part in line.split(","):
            item = part.strip().replace("\\", "/").strip("/")
            if item.startswith("./"):
                item = item[2:].strip("/")
            if not item or item in seen:
                continue
            seen.add(item)
            excludes.append(item)
    return excludes


def _is_excluded_relative_path(rel_path: Path, exclude_paths: list[str]) -> bool:
    if not exclude_paths:
        return False

    rel_posix = rel_path.as_posix()
    rel_parts = rel_path.parts
    for excluded in exclude_paths:
        if "/" in excluded:
            if rel_posix == excluded or rel_posix.startswith(f"{excluded}/"):
                return True
            continue
        if excluded in rel_parts:
            return True
    return False


def _outline_python(source: str) -> dict:
    """Parse a Python source string and return its structural outline."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"parse_error": str(exc)}

    imports: list[dict] = []
    functions: list[dict] = []
    classes: list[dict] = []

    for node in ast.iter_child_nodes(tree):
        # Imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {"type": "import", "name": alias.name, "line": node.lineno}
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            imports.append(
                {
                    "type": "from_import",
                    "module": module,
                    "names": names,
                    "line": node.lineno,
                }
            )
        # Module-level functions
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            returns = ""
            if node.returns:
                try:
                    returns = ast.unparse(node.returns)
                except Exception:
                    pass
            decorators = []
            for dec in node.decorator_list:
                try:
                    decorators.append(ast.unparse(dec))
                except Exception:
                    pass
            doc = ast.get_docstring(node) or ""
            functions.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "args": args,
                    "returns": returns,
                    "decorators": decorators,
                    "docstring_first_line": doc.splitlines()[0][:120] if doc else "",
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                }
            )
        # Classes
        elif isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                try:
                    bases.append(ast.unparse(b))
                except Exception:
                    pass
            decorators = []
            for dec in node.decorator_list:
                try:
                    decorators.append(ast.unparse(dec))
                except Exception:
                    pass
            doc = ast.get_docstring(node) or ""
            methods: list[dict] = []
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    margs = [a.arg for a in child.args.args if a.arg != "self"]
                    mreturns = ""
                    if child.returns:
                        try:
                            mreturns = ast.unparse(child.returns)
                        except Exception:
                            pass
                    mdec = []
                    for dec in child.decorator_list:
                        try:
                            mdec.append(ast.unparse(dec))
                        except Exception:
                            pass
                    mdoc = ast.get_docstring(child) or ""
                    methods.append(
                        {
                            "name": child.name,
                            "line": child.lineno,
                            "end_line": getattr(child, "end_lineno", child.lineno),
                            "args": margs,
                            "returns": mreturns,
                            "decorators": mdec,
                            "docstring_first_line": mdoc.splitlines()[0][:120]
                            if mdoc
                            else "",
                        }
                    )
            classes.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "bases": bases,
                    "decorators": decorators,
                    "docstring_first_line": doc.splitlines()[0][:120] if doc else "",
                    "methods": methods,
                }
            )

    return {"imports": imports, "functions": functions, "classes": classes}


def _outline_generic(source: str, lang: str) -> dict:
    """Rough outline for non-Python files using regex."""
    functions: list[dict] = []
    classes: list[dict] = []

    if lang in ("js", "ts", "jsx", "tsx"):
        fn_re = re.compile(
            r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(|"
            r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(",
            re.MULTILINE,
        )
        cls_re = re.compile(r"class\s+(\w+)", re.MULTILINE)
    else:
        fn_re = re.compile(r"^\s*(?:def|fn|func)\s+(\w+)\s*\(", re.MULTILINE)
        cls_re = re.compile(r"^\s*(?:class|struct|impl)\s+(\w+)", re.MULTILINE)

    for lineno, line in enumerate(source.splitlines(), 1):
        m = fn_re.match(line.strip()) or fn_re.search(line)
        if m:
            name = m.group(1) or m.group(2)
            if name:
                functions.append({"name": name, "line": lineno})
        m2 = cls_re.search(line)
        if m2:
            classes.append({"name": m2.group(1), "line": lineno})

    return {"functions": functions, "classes": classes}


def _outline_js_family(source: str, lang: str) -> dict:
    imports: list[dict] = []
    functions: list[dict] = []
    classes: list[dict] = []

    import_from_re = re.compile(
        r'^\s*import\s+(?P<what>.+?)\s+from\s+["\'](?P<module>.+?)["\'];?\s*$'
    )
    import_side_effect_re = re.compile(r'^\s*import\s+["\'](?P<module>.+?)["\'];?\s*$')
    export_from_re = re.compile(
        r'^\s*export\s+\{(?P<what>.+?)\}\s+from\s+["\'](?P<module>.+?)["\'];?\s*$'
    )
    function_patterns = [
        re.compile(
            r'^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$][\w$]*)\s*\('
        ),
        re.compile(
            r'^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>'
        ),
    ]
    class_patterns = [
        ("class", re.compile(r'^\s*(?:export\s+)?(?:default\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)\b')),
        ("interface", re.compile(r'^\s*(?:export\s+)?interface\s+(?P<name>[A-Za-z_$][\w$]*)\b')),
        ("type", re.compile(r'^\s*(?:export\s+)?type\s+(?P<name>[A-Za-z_$][\w$]*)\s*=')),
        ("enum", re.compile(r'^\s*(?:export\s+)?enum\s+(?P<name>[A-Za-z_$][\w$]*)\b')),
    ]

    for lineno, line in enumerate(source.splitlines(), 1):
        if match := import_from_re.match(line):
            imports.append(
                {
                    "type": "import",
                    "module": match.group("module"),
                    "names": [match.group("what").strip()],
                    "line": lineno,
                }
            )
            continue
        if match := import_side_effect_re.match(line):
            imports.append(
                {
                    "type": "import",
                    "module": match.group("module"),
                    "names": [],
                    "line": lineno,
                }
            )
            continue
        if match := export_from_re.match(line):
            imports.append(
                {
                    "type": "export_from",
                    "module": match.group("module"),
                    "names": [item.strip() for item in match.group("what").split(",") if item.strip()],
                    "line": lineno,
                }
            )
            continue

        for pattern in function_patterns:
            match = pattern.match(line)
            if match:
                functions.append(
                    {
                        "name": match.group("name"),
                        "line": lineno,
                        "kind": "function",
                    }
                )
                break

        for kind, pattern in class_patterns:
            match = pattern.match(line)
            if match:
                classes.append(
                    {
                        "name": match.group("name"),
                        "line": lineno,
                        "kind": kind,
                    }
                )
                break

    return {"imports": imports, "functions": functions, "classes": classes}


def _outline_rust(source: str) -> dict:
    imports: list[dict] = []
    functions: list[dict] = []
    classes: list[dict] = []

    use_re = re.compile(r"^\s*use\s+(.+?);\s*$")
    fn_re = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)\s*\(")
    struct_re = re.compile(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_]\w*)\b")
    enum_re = re.compile(r"^\s*(?:pub\s+)?enum\s+([A-Za-z_]\w*)\b")
    trait_re = re.compile(r"^\s*(?:pub\s+)?trait\s+([A-Za-z_]\w*)\b")
    impl_re = re.compile(r"^\s*impl(?:<[^>]+>)?\s+([A-Za-z_]\w*)\b")

    for lineno, line in enumerate(source.splitlines(), 1):
        if match := use_re.match(line):
            imports.append({"type": "use", "module": match.group(1).strip(), "line": lineno})
            continue
        if match := fn_re.match(line):
            functions.append({"name": match.group(1), "line": lineno, "kind": "function"})
            continue
        for kind, pattern in (
            ("struct", struct_re),
            ("enum", enum_re),
            ("trait", trait_re),
            ("impl", impl_re),
        ):
            match = pattern.match(line)
            if match:
                classes.append({"name": match.group(1), "line": lineno, "kind": kind})
                break

    return {"imports": imports, "functions": functions, "classes": classes}


def _outline_go(source: str) -> dict:
    imports: list[dict] = []
    functions: list[dict] = []
    classes: list[dict] = []

    single_import_re = re.compile(r'^\s*import\s+"(.+?)"\s*$')
    func_re = re.compile(
        r"^\s*func\s+(?:\((?P<receiver>[^)]+)\)\s*)?(?P<name>[A-Za-z_]\w*)\s*\("
    )
    type_re = re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+(struct|interface)\b")
    in_import_block = False

    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if in_import_block:
            if stripped == ")":
                in_import_block = False
                continue
            if stripped.startswith('"') and stripped.endswith('"'):
                imports.append({"type": "import", "module": stripped.strip('"'), "line": lineno})
            continue
        if stripped == "import (":
            in_import_block = True
            continue
        if match := single_import_re.match(line):
            imports.append({"type": "import", "module": match.group(1), "line": lineno})
            continue
        if match := func_re.match(line):
            kind = "method" if match.group("receiver") else "function"
            functions.append({"name": match.group("name"), "line": lineno, "kind": kind})
            continue
        if match := type_re.match(line):
            classes.append({"name": match.group(1), "line": lineno, "kind": match.group(2)})

    return {"imports": imports, "functions": functions, "classes": classes}


def _outline_for_language(source: str, language: str) -> dict:
    if language == "python":
        return _outline_python(source)
    if language in {"js", "jsx", "ts", "tsx"}:
        return _outline_js_family(source, language)
    if language == "rs":
        return _outline_rust(source)
    if language == "go":
        return _outline_go(source)
    return _outline_generic(source, language)


def _detect_source_language(path: Path) -> str:
    return _SOURCE_LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), path.suffix.lstrip(".").lower() or "text")


def _is_supported_source_path(path: Path) -> bool:
    return path.suffix.lower() in _SOURCE_LANGUAGE_BY_SUFFIX


def _definition_patterns_for_language(language: str, name: str) -> list[tuple[str, re.Pattern[str]]]:
    escaped = re.escape(name)
    if language == "python":
        return [
            (
                "definition",
                re.compile(rf"^([ \t]*)(?:(async\s+)?def|class)\s+({escaped})\s*[\(:]", re.MULTILINE),
            )
        ]
    if language in {"js", "jsx", "ts", "tsx"}:
        return [
            ("function", re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+({escaped})\s*\(", re.MULTILINE)),
            ("function", re.compile(rf"^\s*(?:export\s+)?(?:const|let|var)\s+({escaped})\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>", re.MULTILINE)),
            ("class", re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?class\s+({escaped})\b", re.MULTILINE)),
            ("interface", re.compile(rf"^\s*(?:export\s+)?interface\s+({escaped})\b", re.MULTILINE)),
            ("type", re.compile(rf"^\s*(?:export\s+)?type\s+({escaped})\s*=", re.MULTILINE)),
            ("enum", re.compile(rf"^\s*(?:export\s+)?enum\s+({escaped})\b", re.MULTILINE)),
        ]
    if language == "rs":
        return [
            ("function", re.compile(rf"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+({escaped})\s*\(", re.MULTILINE)),
            ("struct", re.compile(rf"^\s*(?:pub\s+)?struct\s+({escaped})\b", re.MULTILINE)),
            ("enum", re.compile(rf"^\s*(?:pub\s+)?enum\s+({escaped})\b", re.MULTILINE)),
            ("trait", re.compile(rf"^\s*(?:pub\s+)?trait\s+({escaped})\b", re.MULTILINE)),
            ("impl", re.compile(rf"^\s*impl(?:<[^>]+>)?\s+({escaped})\b", re.MULTILINE)),
        ]
    if language == "go":
        return [
            ("function", re.compile(rf"^\s*func\s+(?:\([^)]+\)\s*)?({escaped})\s*\(", re.MULTILINE)),
            ("type", re.compile(rf"^\s*type\s+({escaped})\s+(?:struct|interface)\b", re.MULTILINE)),
        ]
    return [
        ("symbol", re.compile(rf"\b{escaped}\b", re.MULTILINE)),
    ]


def _call_pattern(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(rf"\b{escaped}\s*\(", re.MULTILINE)


def _line_number_at_offset(source: str, offset: int) -> int:
    return source[:offset].count("\n") + 1


def _extract_signature(lines: list[str], lineno: int, language: str) -> str:
    terminators = {":"} if language == "python" else {"{", ";", "=>"}
    sig_lines: list[str] = []
    for i in range(lineno - 1, min(lineno + 3, len(lines))):
        current = lines[i]
        sig_lines.append(current)
        stripped = current.strip()
        if any(token in stripped for token in terminators) and not stripped.endswith(","):
            break
    return " ".join(line.strip() for line in sig_lines)


def _snippet(lines: list[str], lineno: int, *, after: int = 2) -> str:
    ctx_start = max(0, lineno - 2)
    ctx_end = min(len(lines), lineno + after)
    return "\n".join(
        f"{ctx_start + i + 1}: {ln}" for i, ln in enumerate(lines[ctx_start:ctx_end])
    )


def _iter_source_files(root: Path, file_glob: str) -> list[Path]:
    if not file_glob.strip():
        file_glob = "**/*"
    out: list[Path] = []
    for fpath in sorted(root.glob(file_glob)):
        if fpath.is_file() and _is_supported_source_path(fpath):
            out.append(fpath)
    return out


def _query_terms(text: str) -> list[str]:
    return [part for part in re.split(r"[^a-z0-9_]+", str(text or "").lower()) if part]


def _path_relevance(path: str, query: str) -> int:
    rel = Path(path)
    path_text = rel.as_posix().lower()
    name = rel.name.lower()
    stem = rel.stem.lower()
    score = 0
    for term in _query_terms(query):
        if stem == term:
            score += 14
        elif name == term or name.startswith(f"{term}."):
            score += 12
        if f"/{term}/" in f"/{path_text}/":
            score += 6
        elif term in path_text:
            score += 3
    depth = len(rel.parts)
    score += max(0, 4 - min(depth, 4))
    if any(part.lower() in {"test", "tests", "__tests__", "spec"} for part in rel.parts):
        score -= 4
    return score


def _symbol_kind_weight(kind: str) -> int:
    return {
        "class": 30,
        "struct": 30,
        "interface": 28,
        "trait": 28,
        "enum": 27,
        "type": 25,
        "function": 24,
        "definition": 24,
        "def": 24,
        "method": 18,
        "impl": 14,
        "symbol": 12,
        "call": 4,
    }.get(str(kind or "").lower(), 10)


def _rank_symbol_matches(matches: list[dict], query: str) -> list[dict]:
    ranked: list[dict] = []
    for item in matches:
        score = _symbol_kind_weight(str(item.get("kind", "")))
        score += _path_relevance(str(item.get("file", "")), query)
        signature = str(item.get("signature", "")).lower()
        query_lower = str(query or "").lower()
        if signature.startswith(("def ", "class ", "fn ", "func ", "type ", "interface ")):
            score += 4
        if query_lower and query_lower in signature:
            score += 3
        ranked_item = dict(item)
        ranked_item["score"] = score
        ranked.append(ranked_item)
    ranked.sort(
        key=lambda item: (
            -int(item.get("score", 0)),
            str(item.get("file", "")),
            int(item.get("line", 0)),
        )
    )
    return ranked


def _rank_search_matches(
    matches: list[dict],
    *,
    query: str,
    fixed_string: bool,
    case_sensitive: bool,
) -> list[dict]:
    ranked: list[dict] = []
    query_text = str(query or "")
    query_lower = query_text.lower()
    for item in matches:
        score = _path_relevance(str(item.get("file", "")), query_text)
        if item.get("match", True):
            score += 20
        text = str(item.get("text", ""))
        text_cmp = text if case_sensitive else text.lower()
        needle = query_text if case_sensitive else query_lower
        if fixed_string and needle and needle in text_cmp:
            score += 10
            if text.strip().startswith(query_text):
                score += 3
        elif not fixed_string and query_lower and query_lower in text.lower():
            score += 4
        ranked_item = dict(item)
        ranked_item["score"] = score
        ranked.append(ranked_item)
    ranked.sort(
        key=lambda item: (
            -int(item.get("score", 0)),
            str(item.get("file", "")),
            int(item.get("line", 0)),
            0 if item.get("match", True) else 1,
        )
    )
    return ranked


def _rollup_matches_by_file(
    matches: list[dict],
    *,
    limit: int = 8,
    include_kind_counts: bool = False,
) -> list[dict]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in matches:
        file = str(item.get("file", ""))
        if not file:
            continue
        bucket = grouped.setdefault(
            file,
            {
                "file": file,
                "score": 0,
                "count": 0,
                "first_line": int(item.get("line", 0) or 0),
                "languages": set(),
                "kinds": {},
                "preview": "",
            },
        )
        bucket["score"] = max(int(bucket["score"]), int(item.get("score", 0)))
        if item.get("match", True):
            bucket["count"] += 1
        if int(item.get("line", 0) or 0) and (
            bucket["first_line"] == 0 or int(item.get("line", 0)) < int(bucket["first_line"])
        ):
            bucket["first_line"] = int(item.get("line", 0))
        language = str(item.get("language", "")).strip()
        if language:
            bucket["languages"].add(language)
        kind = str(item.get("kind", "")).strip()
        if include_kind_counts and kind:
            bucket["kinds"][kind] = int(bucket["kinds"].get(kind, 0)) + 1
        if not bucket["preview"]:
            bucket["preview"] = str(item.get("signature") or item.get("text") or "").strip()[:160]

    out: list[dict] = []
    for item in grouped.values():
        payload = {
            "file": item["file"],
            "count": item["count"],
            "first_line": item["first_line"],
            "languages": sorted(item["languages"]),
            "preview": item["preview"],
        }
        if include_kind_counts:
            payload["kinds"] = dict(
                sorted(item["kinds"].items(), key=lambda pair: (-int(pair[1]), pair[0]))
            )
        payload["score"] = item["score"]
        out.append(payload)

    out.sort(
        key=lambda item: (
            -int(item.get("score", 0)),
            -int(item.get("count", 0)),
            str(item.get("file", "")),
        )
    )
    for item in out:
        item.pop("score", None)
    return out[:limit]


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _coerce_int(value: Any, *, name: str, default: int) -> tuple[int, str | None]:
    if value is None:
        return default, None
    if isinstance(value, bool):
        return int(value), None
    if isinstance(value, int):
        return value, None
    if isinstance(value, float):
        if value.is_integer():
            return int(value), None
        return 0, f"Invalid {name}: expected integer, got {value!r}"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default, None
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text), None
        return 0, f"Invalid {name}: expected integer, got {value!r}"
    return 0, f"Invalid {name}: expected integer, got {type(value).__name__}"


def _project_map_git_identity(root: Path) -> tuple[str, str | None, str | None]:
    """Return (identity, repo_root, head) for cache invalidation.

    For git repos, identity includes HEAD + working tree dirty hash so cache is
    invalidated on commits and local edits. Outside git repos, identity falls
    back to root mtime.
    """
    try:
        p = root if root.is_dir() else root.parent
        top = subprocess.run(
            ["git", "-C", str(p), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if top.returncode != 0:
            raise RuntimeError("not a git repo")
        repo_root = top.stdout.strip()
        head = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if head.returncode != 0 or not head.stdout.strip():
            raise RuntimeError("cannot resolve HEAD")
        head_sha = head.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", repo_root, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        dirty_sig = str(hash((dirty.stdout or "").strip()))
        identity = f"git:{repo_root}:{head_sha}:{dirty_sig}"
        return identity, repo_root, head_sha
    except Exception:
        mtime_ns = 0
        try:
            mtime_ns = int(root.stat().st_mtime_ns)
        except Exception:
            pass
        return f"fs:{str(root)}:{mtime_ns}", None, None


def _project_map_kind_and_language(rel: Path) -> tuple[str, str] | None:
    name = rel.name.lower()
    suffix = rel.suffix.lower()

    if suffix in _PROJECT_MAP_CODE_EXTENSIONS:
        return "source", suffix.lstrip(".")
    if name in _PROJECT_MAP_CONFIG_FILENAMES:
        if name == "dockerfile":
            return "config", "dockerfile"
        if name in {"makefile", "justfile"}:
            return "config", "makefile"
        if suffix:
            return "config", suffix.lstrip(".")
        return "config", "text"
    if name in _PROJECT_MAP_DOC_FILENAMES:
        return "doc", "md" if suffix == ".md" else "text"
    return None


def _strip_comment_prefix(text: str) -> str:
    value = text.strip()
    for prefix in ("#", "//", "/*", "*", "--", ";", '"', "'", "'''", '"""'):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
    return value.strip("*/-#;\"' ")


def _leading_summary_line(source: str) -> str:
    for line in source.splitlines()[:24]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("import ", "from ", "export ", "{", "}", "[", "]")):
            continue
        if re.match(
            r"^(?:async\s+def|def|class|function|export\s+function|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=)",
            stripped,
        ):
            continue
        if stripped.startswith(("#", "//", "/*", "*", "--", ";", '"', "'", "'''", '"""')):
            text = _strip_comment_prefix(stripped)
            if text:
                return text[:120]
            continue
        if stripped.startswith("# "):
            return stripped[2:].strip()[:120]
        if stripped.startswith("[") and stripped.endswith("]"):
            return stripped[1:-1].strip()[:120]
        if ":" in stripped and len(stripped) < 120:
            return stripped[:120]
        if stripped.lower().startswith(("name =", "version =", '"name":', '"scripts":')):
            return stripped[:120]
    return ""


def _symbol_preview(functions: list[dict], classes: list[dict]) -> list[str]:
    names = [item.get("name", "") for item in classes[:3]] + [
        item.get("name", "") for item in functions[:3]
    ]
    return [name for name in names if name]


def _fallback_summary(rel: Path, kind: str, language: str, symbols: list[str]) -> str:
    name = rel.name.lower()
    if symbols:
        return "Defines " + ", ".join(symbols[:4])
    if name == "__init__.py":
        return "Python package initializer"
    if name == "pyproject.toml":
        return "Python project configuration"
    if name == "package.json":
        return "Node package manifest"
    if name == "cargo.toml":
        return "Rust crate manifest"
    if name == "go.mod":
        return "Go module definition"
    if name == "dockerfile":
        return "Container build recipe"
    if name in {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}:
        return "Container service definition"
    if name in {"makefile", "justfile"}:
        return "Build and task automation"
    if kind == "doc":
        return "Project documentation"
    if kind == "config":
        label = _PROJECT_MAP_LANGUAGE_LABELS.get(language, language.upper() or "config")
        return f"{label} configuration"
    label = _PROJECT_MAP_LANGUAGE_LABELS.get(language, language.upper() or "source")
    return f"{label} source file"


def _project_map_entry(root: Path, rel: Path) -> dict[str, Any] | None:
    kind_and_language = _project_map_kind_and_language(rel)
    if kind_and_language is None:
        return None

    kind, language = kind_and_language
    fpath = root / rel

    try:
        source = fpath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    total_lines = source.count("\n") + 1
    functions: list[dict] = []
    classes: list[dict] = []
    summary = ""

    if language == "py":
        outline = _outline_for_language(source, "python")
        if "parse_error" not in outline:
            functions = outline.get("functions", [])
            classes = outline.get("classes", [])
        try:
            tree = ast.parse(source)
            doc = ast.get_docstring(tree)
            if doc:
                summary = doc.splitlines()[0][:120]
        except SyntaxError:
            pass
    elif kind == "source":
        outline = _outline_for_language(source, language)
        functions = outline.get("functions", [])
        classes = outline.get("classes", [])

    if not summary:
        summary = _leading_summary_line(source)

    symbols = _symbol_preview(functions, classes)
    if not summary:
        summary = _fallback_summary(rel, kind, language, symbols)

    return {
        "path": str(rel),
        "kind": kind,
        "language": language,
        "lines": total_lines,
        "n_functions": len(functions),
        "n_classes": len(classes),
        "symbols": symbols,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def get_file_outline(path: str) -> str:
    """Use when: You need to understand what's in a file before diving into it.

    Triggers: outline file, file structure, what functions, what classes, methods in file,
              understand codebase, look at file, explore file, inspect module, map code.
    Avoid when: You already know the exact line range you need — use read_file directly.
    Inputs:
      path (str, required): Absolute or relative path to the source file.
    Returns: JSON with imports, classes/types, and functions — all with line numbers.
    Side effects: Read-only.

    Example output:
      {
        "status": "ok",
        "path": "/project/src/agent.py",
        "total_lines": 312,
        "language": "python",
        "imports": [{"type": "from_import", "module": "pathlib", "names": ["Path"], "line": 3}],
        "classes": [{"name": "Agent", "line": 45, "end_line": 200, "bases": ["BaseAgent"],
                     "methods": [{"name": "__init__", "line": 46}, {"name": "run", "line": 60}]}],
        "functions": [{"name": "create_agent", "line": 210, "end_line": 240, "args": ["url", "config"]}]
      }
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return _safe_json({"status": "error", "error": f"File not found: {path}"})
    if not p.is_file():
        return _safe_json({"status": "error", "error": f"Not a file: {path}"})

    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})

    total_lines = source.count("\n") + 1
    lang = _detect_source_language(p)
    outline = _outline_for_language(source, lang)

    if "parse_error" in outline:
        return _safe_json(
            {
                "status": "partial",
                "path": str(p),
                "total_lines": total_lines,
                "language": lang,
                "parse_error": outline["parse_error"],
                "note": "Use read_file to inspect this file directly.",
            }
        )

    return _safe_json(
        {
            "status": "ok",
            "path": str(p),
            "total_lines": total_lines,
            "language": lang,
            **outline,
        }
    )


@tool
def find_symbol(
    name: str,
    directory: str = ".",
    file_glob: str = "**/*",
    include_calls: bool = False,
) -> str:
    """Use when: You want to jump to the definition of a specific function or class.

    Triggers: find function, where is defined, go to definition, locate class, find symbol,
              where does X live, which file defines, find implementation.
    Avoid when: You already know the file — use read_file or get_file_outline.
    Inputs:
      name (str, required): Exact function or class name to find.
      directory (str, optional): Root directory to search (default ".").
      file_glob (str, optional): File pattern (default "**/*", filtered to supported source files).
      include_calls (bool, optional): Also return call sites, not just definitions (default False).
    Returns: JSON list of matches: {file, line, kind, signature, snippet}.
    Side effects: Read-only.

    Example: find_symbol("create_agent") →
      [{"file": "src/agent/factory.py", "line": 12, "kind": "def",
        "signature": "def create_agent(url, config=None):", "snippet": "...3 lines of context..."}]
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return _safe_json({"status": "error", "error": f"Not a directory: {directory}"})

    matches: list[dict] = []
    MAX = 60
    call_re = _call_pattern(name) if include_calls else None

    for fpath in _iter_source_files(root, file_glob):
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines = source.splitlines()
        language = _detect_source_language(fpath)
        seen_lines: set[tuple[str, int]] = set()
        for kind, pattern in _definition_patterns_for_language(language, name):
            for m in pattern.finditer(source):
                lineno = _line_number_at_offset(source, m.start())
                dedupe_key = (kind, lineno)
                if dedupe_key in seen_lines:
                    continue
                seen_lines.add(dedupe_key)
                signature = _extract_signature(lines, lineno, language)
                matches.append(
                    {
                        "file": str(fpath.relative_to(root)),
                        "line": lineno,
                        "kind": kind,
                        "language": language,
                        "signature": signature[:200],
                        "snippet": _snippet(lines, lineno),
                    }
                )
                if len(matches) >= MAX:
                    break
            if len(matches) >= MAX:
                break

        if include_calls and call_re:
            for m in call_re.finditer(source):
                lineno = _line_number_at_offset(source, m.start())
                matches.append(
                    {
                        "file": str(fpath.relative_to(root)),
                        "line": lineno,
                        "kind": "call",
                        "language": language,
                        "signature": lines[lineno - 1].strip()[:200],
                        "snippet": _snippet(lines, lineno, after=1),
                    }
                )

        if len(matches) >= MAX:
            break

    if not matches:
        return _safe_json(
            {
                "status": "ok",
                "name": name,
                "count": 0,
                "matches": [],
                "note": f"No definition of '{name}' found. Check spelling or widen directory/glob.",
            }
        )

    matches = _rank_symbol_matches(matches, name)
    top_files = _rollup_matches_by_file(matches, limit=8, include_kind_counts=True)
    by_language: dict[str, int] = {}
    for item in matches:
        language = str(item.get("language", "")).strip()
        if language:
            by_language[language] = by_language.get(language, 0) + 1

    return _safe_json(
        {
            "status": "ok",
            "name": name,
            "count": len(matches),
            "truncated": len(matches) >= MAX,
            "top_files": top_files,
            "by_language": by_language,
            "matches": matches,
        }
    )


@tool
def get_project_map(directory: str = ".", max_depth: int = 3, exclude: str = "") -> str:
    """Use when: You need a bird's-eye view of a package or project.

    Triggers: map project, project structure, package overview, what files do what,
              understand repo, show all modules, codebase overview.
    Avoid when: You need function-level detail — use get_file_outline per file.
    Inputs:
      directory (str, optional): Root of the project to scan (default ".").
      max_depth (int, optional): Folder depth limit (default 3).
      exclude (str, optional): Comma/newline-separated subpaths to skip, e.g. "node_modules,dist".
    Returns: JSON list of {path, kind, language, summary, n_functions, n_classes, lines}.
    Side effects: Read-only.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return _safe_json({"status": "error", "error": f"Not a directory: {directory}"})
    max_depth = max(0, int(max_depth))
    exclude_paths = _parse_exclude_paths(exclude)

    cache_key = (str(root), max_depth, "|".join(exclude_paths))
    identity, repo_root, head_sha = _project_map_git_identity(root)
    cached = _PROJECT_MAP_CACHE.get(cache_key)
    if cached and cached.get("identity") == identity:
        payload = dict(cached.get("payload", {}))
        payload["cache_hit"] = True
        payload["cache_identity"] = identity
        payload["generated_at"] = cached.get("generated_at", payload.get("generated_at"))
        return _safe_json(payload)

    files: list[dict] = []
    language_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}

    for fpath in sorted(root.rglob("*")):
        if not fpath.is_file():
            continue
        try:
            rel = fpath.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) - 1 > max_depth:
            continue
        if _is_excluded_relative_path(rel, exclude_paths):
            continue
        if any(part.startswith((".", "_")) for part in rel.parts[:-1]):
            continue

        entry = _project_map_entry(root, rel)
        if entry is None:
            continue

        files.append(entry)
        language = str(entry.get("language", "text"))
        kind = str(entry.get("kind", "source"))
        language_counts[language] = language_counts.get(language, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    payload = {
        "status": "ok",
        "root": str(root),
        "file_count": len(files),
        "files": files,
        "by_language": language_counts,
        "by_kind": kind_counts,
        "cache_hit": False,
        "cache_identity": identity,
        "generated_at": int(time.time()),
    }
    if repo_root and head_sha:
        payload["git_repo_root"] = repo_root
        payload["git_head"] = head_sha

    _PROJECT_MAP_CACHE[cache_key] = {
        "identity": identity,
        "payload": payload,
        "generated_at": payload["generated_at"],
    }

    # Bound cache size to keep long-running sessions predictable.
    if len(_PROJECT_MAP_CACHE) > 48:
        for stale_key in list(_PROJECT_MAP_CACHE.keys())[: len(_PROJECT_MAP_CACHE) - 48]:
            _PROJECT_MAP_CACHE.pop(stale_key, None)

    return _safe_json(payload)


# ---------------------------------------------------------------------------
# Internal: run-command helper (reuse bootstrap's if available, else fallback)
# ---------------------------------------------------------------------------


def _explore_run(cmd: str, cwd: str | None = None, timeout: int = 30) -> dict:
    """Run a command using the shared coding runtime, else subprocess directly."""
    _rt = get_coding_runtime(globals())
    if _rt is not None:
        try:
            return _rt.run_cmd(cmd, cwd=cwd, timeout=timeout)
        except RuntimeError:
            pass
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
        )
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "exit_code": -1, "stdout": "", "stderr": "timeout"}
    except Exception as exc:
        return {"status": "error", "exit_code": -1, "stdout": "", "stderr": str(exc)}


# ---------------------------------------------------------------------------
# rg_search / fd_find  (system-tool wrappers with pure-Python fallback)
# ---------------------------------------------------------------------------


@tool
def rg_search(
    pattern: str,
    directory: str = ".",
    file_glob: str = "",
    file_type: str = "",
    context_lines: int = 0,
    case_sensitive: bool = False,
    fixed_string: bool = False,
    max_results: int = 80,
) -> str:
    """Use when: Search for any text/pattern across a codebase quickly.

    Triggers: search code, find text, grep, rg, ripgrep, where is string, find pattern,
              find all uses, search files, look for, locate text, who calls.
    Avoid when: You want symbol definitions only — use find_symbol instead.
    Inputs:
      pattern (str, required): Regex or literal string to search for.
      directory (str, optional): Root directory (default ".").
      file_glob (str, optional): Glob pattern to restrict files, e.g. "*.py" or "src/**/*.ts".
      file_type (str, optional): ripgrep file type name, e.g. "py", "js", "ts", "rust", "md".
                                 Only used when rg is available. Ignored in fallback mode.
      context_lines (int, optional): Lines of context before/after each match (default 0).
      case_sensitive (bool, optional): Match case exactly (default False = case-insensitive).
      fixed_string (bool, optional): Treat pattern as literal string, not regex (default False).
      max_results (int, optional): Stop after this many matches (default 80).
    Returns: JSON with matches list {file, line, text} plus tool_used ("rg" or "python").
    Side effects: Read-only; spawns subprocess if rg is available.

    Examples:
      rg_search("ThinkingConfig")                    -- find all uses of a symbol
      rg_search("TODO", file_type="py")               -- find all Python TODOs
      rg_search("def run", context_lines=2)           -- show 2 lines around each match
      rg_search("import torch", fixed_string=True)    -- exact literal search
    """
    pattern = str(pattern)
    directory = str(directory or ".")
    file_glob = str(file_glob or "")
    file_type = str(file_type or "")
    case_sensitive = _coerce_bool(case_sensitive, default=False)
    fixed_string = _coerce_bool(fixed_string, default=False)
    context_lines_i, err = _coerce_int(
        context_lines, name="context_lines", default=0
    )
    if err:
        return _safe_json({"status": "error", "error": err})
    max_results_i, err = _coerce_int(max_results, name="max_results", default=80)
    if err:
        return _safe_json({"status": "error", "error": err})
    context_lines = max(0, context_lines_i)
    max_results = max(1, max_results_i)

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return _safe_json({"status": "error", "error": f"Not a directory: {directory}"})

    # ---- Try ripgrep first ----
    if shutil.which("rg"):
        parts = ["rg", "--line-number", "--no-heading", "--color=never"]
        if not case_sensitive:
            parts.append("--ignore-case")
        if fixed_string:
            parts.append("--fixed-strings")
        if context_lines > 0:
            parts += ["-C", str(context_lines)]
        if file_type:
            parts += ["--type", file_type]
        if file_glob:
            parts += ["--glob", file_glob]
        parts += ["--max-count", str(max_results)]
        parts.append(pattern)
        parts.append(str(root))

        cmd = " ".join(
            p if re.match(r"^[\w./:@=+,-]+$", p) else f"'{p}'" for p in parts
        )
        r = _explore_run(cmd, timeout=30)

        raw_lines = r["stdout"].splitlines()
        matches: list[dict] = []
        # rg --no-heading format: "filepath:lineno:text"  or  "filepath-lineno-text" for context
        line_re = re.compile(r"^(.+?):(\d+):(.*)$")
        ctx_re = re.compile(r"^(.+?)-(\d+)-(.*)$")
        for raw in raw_lines:
            m = line_re.match(raw)
            if m:
                matches.append(
                    {
                        "file": m.group(1),
                        "line": int(m.group(2)),
                        "text": m.group(3),
                        "match": True,
                    }
                )
            elif context_lines > 0:
                mc = ctx_re.match(raw)
                if mc:
                    matches.append(
                        {
                            "file": mc.group(1),
                            "line": int(mc.group(2)),
                            "text": mc.group(3),
                            "match": False,
                        }
                    )

        # Make file paths relative to root when possible
        for entry in matches:
            try:
                entry["file"] = str(Path(entry["file"]).relative_to(root))
            except ValueError:
                pass

        matches = _rank_search_matches(
            matches,
            query=pattern,
            fixed_string=fixed_string,
            case_sensitive=case_sensitive,
        )
        real_count = len([m for m in matches if m.get("match", True)])
        truncated = real_count >= max_results
        return _safe_json(
            {
                "status": "ok",
                "tool_used": "rg",
                "pattern": pattern,
                "count": real_count,
                "truncated": truncated,
                "top_files": _rollup_matches_by_file(matches, limit=8),
                "matches": matches[:max_results],
            }
        )

    # ---- Pure-Python fallback ----
    glob_pat = file_glob if file_glob else "**/*"
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        rx = re.compile(re.escape(pattern) if fixed_string else pattern, flags)
    except re.error as exc:
        return _safe_json({"status": "error", "error": f"Invalid regex: {exc}"})

    matches = []
    for fpath in sorted(root.glob(glob_pat)):
        if not fpath.is_file():
            continue
        try:
            src_lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for lineno, text in enumerate(src_lines, 1):
            if rx.search(text):
                entry = {
                    "file": str(fpath.relative_to(root)),
                    "line": lineno,
                    "text": text.rstrip(),
                    "match": True,
                }
                matches.append(entry)
                if context_lines > 0:
                    for delta in range(1, context_lines + 1):
                        for sign, before in ((-delta, True), (delta, False)):
                            idx = lineno + sign - 1
                            if 0 <= idx < len(src_lines):
                                ctx_entry = {
                                    "file": str(fpath.relative_to(root)),
                                    "line": idx + 1,
                                    "text": src_lines[idx].rstrip(),
                                    "match": False,
                                }
                                if before:
                                    matches.insert(-1, ctx_entry)
                                else:
                                    matches.append(ctx_entry)
                if len([m for m in matches if m.get("match")]) >= max_results:
                    break
        if len([m for m in matches if m.get("match")]) >= max_results:
            break

    matches = _rank_search_matches(
        matches,
        query=pattern,
        fixed_string=fixed_string,
        case_sensitive=case_sensitive,
    )
    real_count = len([m for m in matches if m.get("match")])
    return _safe_json(
        {
            "status": "ok",
            "tool_used": "python",
            "pattern": pattern,
            "count": real_count,
            "truncated": real_count >= max_results,
            "top_files": _rollup_matches_by_file(matches, limit=8),
            "matches": matches,
        }
    )


@tool
def find_references(name: str, directory: str = ".", file_glob: str = "") -> str:
    """Use when: You need to locate where a specific symbol is used or called.

    Triggers: find references, usages of, who calls, where is used, find usages.
    Avoid when: You want to find the definition (use find_symbol instead).
    Inputs:
      name (str, required): The exact name of the symbol (e.g. 'compute_statistics').
      directory (str, optional): Root folder to search in (default '.').
      file_glob (str, optional): Glob to filter files (default '').
    Returns: JSON with matches and snippets.
    """
    pattern = rf"\b{name}\b"
    return rg_search(
        pattern=pattern,
        directory=directory,
        file_glob=file_glob,
        context_lines=1,
        max_results=30
    )


@tool
def fd_find(
    pattern: str,
    directory: str = ".",
    file_type: Literal["f", "d", ""] = "f",
    extension: str = "",
    max_depth: int = 8,
    max_results: int = 50,
    hidden: bool = False,
) -> str:
    """Use when: You want to find files or directories by name pattern across the project.

    Triggers: find file, locate file, where is file, search for file, find directory,
              which file is named, fd, find by name, file search, where lives.
    Avoid when: You want to search file *contents* — use rg_search or search_in_files.
    Inputs:
      pattern (str, required): Name pattern to match. fd uses smart-case regex, e.g.
                               "config", "test_.*\\.py", "__init__".
      directory (str, optional): Root directory to search (default ".").
      file_type (str, optional): "f" = files only (default), "d" = directories only, "" = both.
      extension (str, optional): Filter by extension, e.g. "py", "json", "md" (no dot).
      max_depth (int, optional): Maximum directory depth (default 8).
      max_results (int, optional): Stop after N results (default 50).
      hidden (bool, optional): Include hidden files/dirs (default False).
    Returns: JSON with list of matching paths relative to directory.
    Side effects: Read-only.

    Examples:
      fd_find("config")                     -- find all files with 'config' in name
      fd_find("test_", extension="py")      -- find all test_*.py files
      fd_find("__pycache__", file_type="d") -- find all __pycache__ dirs
      fd_find("README", extension="md")     -- find README.md files
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return _safe_json({"status": "error", "error": f"Not a directory: {directory}"})

    # ---- Try fd first ----
    if shutil.which("fd"):
        parts = ["fd", "--color=never", "--max-depth", str(max_depth)]
        if file_type in ("f", "d"):
            parts += ["--type", file_type]
        if extension:
            parts += ["--extension", extension.lstrip(".")]
        if hidden:
            parts.append("--hidden")
        parts.append(pattern)
        parts.append(str(root))

        cmd = " ".join(
            p if re.match(r"^[\w./:@=+,-]+$", p) else f"'{p}'" for p in parts
        )
        r = _explore_run(cmd, timeout=20)
        raw_paths = [ln.strip() for ln in r["stdout"].splitlines() if ln.strip()]

        # Make relative
        rel_paths = []
        for rp in raw_paths[:max_results]:
            try:
                rel_paths.append(str(Path(rp).relative_to(root)))
            except ValueError:
                rel_paths.append(rp)

        return _safe_json(
            {
                "status": "ok",
                "tool_used": "fd",
                "pattern": pattern,
                "count": len(rel_paths),
                "truncated": len(raw_paths) > max_results,
                "paths": rel_paths,
            }
        )

    # ---- Pure-Python fallback ----
    # Build a glob pattern from the extension + max_depth
    if extension:
        ext = extension.lstrip(".")
        glob_pats = [f"{'*/' * d}*.{ext}" for d in range(max_depth + 1)]
    else:
        glob_pats = [f"{'*/' * d}*" for d in range(max_depth + 1)]

    try:
        name_re = re.compile(pattern, re.IGNORECASE)
    except re.error:
        name_re = re.compile(re.escape(pattern), re.IGNORECASE)

    found: list[str] = []
    seen: set[str] = set()
    for gp in glob_pats:
        for fpath in sorted(root.glob(gp)):
            if not hidden and any(part.startswith(".") for part in fpath.parts):
                continue
            is_dir = fpath.is_dir()
            if file_type == "f" and is_dir:
                continue
            if file_type == "d" and not is_dir:
                continue
            if name_re.search(fpath.name):
                rel = str(fpath.relative_to(root))
                if rel not in seen:
                    seen.add(rel)
                    found.append(rel)
                    if len(found) >= max_results:
                        break
        if len(found) >= max_results:
            break

    return _safe_json(
        {
            "status": "ok",
            "tool_used": "python",
            "pattern": pattern,
            "count": len(found),
            "truncated": len(found) >= max_results,
            "paths": found,
        }
    )


__tools__ = [get_file_outline, find_symbol, get_project_map, rg_search, find_references, fd_find]
