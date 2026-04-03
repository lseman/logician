"""Read-only exploration primitives promoted from the coding explore skill."""

from __future__ import annotations

import ast
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from ..FileReadTool.state import resolve_tool_path

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


def _err(message: str) -> dict[str, Any]:
    return {"status": "error", "error": message}


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


def _outline_python(source: str) -> dict[str, Any]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"parse_error": str(exc)}

    imports: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"type": "import", "name": alias.name, "line": node.lineno})
        elif isinstance(node, ast.ImportFrom):
            imports.append(
                {
                    "type": "from_import",
                    "module": node.module or "",
                    "names": [alias.name for alias in node.names],
                    "line": node.lineno,
                }
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [arg.arg for arg in node.args.args]
            returns = ""
            if node.returns:
                try:
                    returns = ast.unparse(node.returns)
                except Exception:
                    returns = ""
            decorators: list[str] = []
            for dec in node.decorator_list:
                try:
                    decorators.append(ast.unparse(dec))
                except Exception:
                    continue
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
        elif isinstance(node, ast.ClassDef):
            bases: list[str] = []
            decorators: list[str] = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except Exception:
                    continue
            for dec in node.decorator_list:
                try:
                    decorators.append(ast.unparse(dec))
                except Exception:
                    continue
            doc = ast.get_docstring(node) or ""
            methods: list[dict[str, Any]] = []
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_doc = ast.get_docstring(child) or ""
                    methods.append(
                        {
                            "name": child.name,
                            "line": child.lineno,
                            "end_line": getattr(child, "end_lineno", child.lineno),
                            "args": [arg.arg for arg in child.args.args if arg.arg != "self"],
                            "returns": ast.unparse(child.returns) if child.returns else "",
                            "decorators": [
                                ast.unparse(dec)
                                for dec in child.decorator_list
                                if hasattr(ast, "unparse")
                            ],
                            "docstring_first_line": method_doc.splitlines()[0][:120]
                            if method_doc
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


def _outline_generic(source: str, lang: str) -> dict[str, Any]:
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []

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
        match = fn_re.match(line.strip()) or fn_re.search(line)
        if match:
            name = match.group(1) or match.group(2)
            if name:
                functions.append({"name": name, "line": lineno})
        class_match = cls_re.search(line)
        if class_match:
            classes.append({"name": class_match.group(1), "line": lineno})

    return {"functions": functions, "classes": classes}


def _outline_js_family(source: str, _lang: str) -> dict[str, Any]:
    imports: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []

    import_from_re = re.compile(
        r'^\s*import\s+(?P<what>.+?)\s+from\s+["\'](?P<module>.+?)["\'];?\s*$'
    )
    import_side_effect_re = re.compile(r'^\s*import\s+["\'](?P<module>.+?)["\'];?\s*$')
    export_from_re = re.compile(
        r'^\s*export\s+\{(?P<what>.+?)\}\s+from\s+["\'](?P<module>.+?)["\'];?\s*$'
    )
    function_patterns = [
        re.compile(
            r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$][\w$]*)\s*\("
        ),
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"
        ),
    ]
    class_patterns = [
        (
            "class",
            re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)\b"),
        ),
        ("interface", re.compile(r"^\s*(?:export\s+)?interface\s+(?P<name>[A-Za-z_$][\w$]*)\b")),
        ("type", re.compile(r"^\s*(?:export\s+)?type\s+(?P<name>[A-Za-z_$][\w$]*)\s*=")),
        ("enum", re.compile(r"^\s*(?:export\s+)?enum\s+(?P<name>[A-Za-z_$][\w$]*)\b")),
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
                    "names": [
                        item.strip() for item in match.group("what").split(",") if item.strip()
                    ],
                    "line": lineno,
                }
            )
            continue

        for pattern in function_patterns:
            if match := pattern.match(line):
                functions.append({"name": match.group("name"), "line": lineno, "kind": "function"})
                break

        for kind, pattern in class_patterns:
            if match := pattern.match(line):
                classes.append({"name": match.group("name"), "line": lineno, "kind": kind})
                break

    return {"imports": imports, "functions": functions, "classes": classes}


def _outline_rust(source: str) -> dict[str, Any]:
    imports: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []

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
            if match := pattern.match(line):
                classes.append({"name": match.group(1), "line": lineno, "kind": kind})
                break

    return {"imports": imports, "functions": functions, "classes": classes}


def _outline_go(source: str) -> dict[str, Any]:
    imports: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []

    single_import_re = re.compile(r'^\s*import\s+"(.+?)"\s*$')
    func_re = re.compile(r"^\s*func\s+(?:\((?P<receiver>[^)]+)\)\s*)?(?P<name>[A-Za-z_]\w*)\s*\(")
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
            functions.append(
                {
                    "name": match.group("name"),
                    "line": lineno,
                    "kind": "method" if match.group("receiver") else "function",
                }
            )
            continue
        if match := type_re.match(line):
            classes.append({"name": match.group(1), "line": lineno, "kind": match.group(2)})

    return {"imports": imports, "functions": functions, "classes": classes}


def _outline_for_language(source: str, language: str) -> dict[str, Any]:
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
    return _SOURCE_LANGUAGE_BY_SUFFIX.get(
        path.suffix.lower(), path.suffix.lstrip(".").lower() or "text"
    )


def _is_supported_source_path(path: Path) -> bool:
    return path.suffix.lower() in _SOURCE_LANGUAGE_BY_SUFFIX


def _definition_patterns_for_language(
    language: str, name: str
) -> list[tuple[str, re.Pattern[str]]]:
    escaped = re.escape(name)
    if language == "python":
        return [
            (
                "definition",
                re.compile(
                    rf"^([ \t]*)(?:(async\s+)?def|class)\s+({escaped})\s*[\(:]", re.MULTILINE
                ),
            )
        ]
    if language in {"js", "jsx", "ts", "tsx"}:
        return [
            (
                "function",
                re.compile(
                    rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+({escaped})\s*\(",
                    re.MULTILINE,
                ),
            ),
            (
                "function",
                re.compile(
                    rf"^\s*(?:export\s+)?(?:const|let|var)\s+({escaped})\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>",
                    re.MULTILINE,
                ),
            ),
            (
                "class",
                re.compile(
                    rf"^\s*(?:export\s+)?(?:default\s+)?class\s+({escaped})\b", re.MULTILINE
                ),
            ),
            (
                "interface",
                re.compile(rf"^\s*(?:export\s+)?interface\s+({escaped})\b", re.MULTILINE),
            ),
            ("type", re.compile(rf"^\s*(?:export\s+)?type\s+({escaped})\s*=", re.MULTILINE)),
            ("enum", re.compile(rf"^\s*(?:export\s+)?enum\s+({escaped})\b", re.MULTILINE)),
        ]
    if language == "rs":
        return [
            (
                "function",
                re.compile(
                    rf"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+({escaped})\s*\(",
                    re.MULTILINE,
                ),
            ),
            ("struct", re.compile(rf"^\s*(?:pub\s+)?struct\s+({escaped})\b", re.MULTILINE)),
            ("enum", re.compile(rf"^\s*(?:pub\s+)?enum\s+({escaped})\b", re.MULTILINE)),
            ("trait", re.compile(rf"^\s*(?:pub\s+)?trait\s+({escaped})\b", re.MULTILINE)),
            ("impl", re.compile(rf"^\s*impl(?:<[^>]+>)?\s+({escaped})\b", re.MULTILINE)),
        ]
    if language == "go":
        return [
            (
                "function",
                re.compile(rf"^\s*func\s+(?:\([^)]+\)\s*)?({escaped})\s*\(", re.MULTILINE),
            ),
            ("type", re.compile(rf"^\s*type\s+({escaped})\s+(?:struct|interface)\b", re.MULTILINE)),
        ]
    return [("symbol", re.compile(rf"\b{escaped}\b", re.MULTILINE))]


def _call_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(name)}\s*\(", re.MULTILINE)


def _line_number_at_offset(source: str, offset: int) -> int:
    return source[:offset].count("\n") + 1


def _extract_signature(lines: list[str], lineno: int, language: str) -> str:
    terminators = {":"} if language == "python" else {"{", ";", "=>"}
    sig_lines: list[str] = []
    for index in range(lineno - 1, min(lineno + 3, len(lines))):
        current = lines[index]
        sig_lines.append(current)
        stripped = current.strip()
        if any(token in stripped for token in terminators) and not stripped.endswith(","):
            break
    return " ".join(line.strip() for line in sig_lines)


def _snippet(lines: list[str], lineno: int, *, after: int = 2) -> str:
    ctx_start = max(0, lineno - 2)
    ctx_end = min(len(lines), lineno + after)
    return "\n".join(
        f"{ctx_start + index + 1}: {line}" for index, line in enumerate(lines[ctx_start:ctx_end])
    )


def _iter_source_files(root: Path, file_glob: str) -> list[Path]:
    if not file_glob.strip():
        file_glob = "**/*"
    return [
        candidate
        for candidate in sorted(root.glob(file_glob))
        if candidate.is_file() and _is_supported_source_path(candidate)
    ]


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


def _rank_symbol_matches(matches: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
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


def _rollup_matches_by_file(
    matches: list[dict[str, Any]],
    *,
    limit: int = 8,
    include_kind_counts: bool = False,
) -> list[dict[str, Any]]:
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

    out: list[dict[str, Any]] = []
    for item in grouped.values():
        payload: dict[str, Any] = {
            "file": item["file"],
            "count": item["count"],
            "first_line": item["first_line"],
            "languages": sorted(item["languages"]),
            "preview": item["preview"],
            "score": item["score"],
        }
        if include_kind_counts:
            payload["kinds"] = dict(
                sorted(item["kinds"].items(), key=lambda pair: (-int(pair[1]), pair[0]))
            )
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


def _project_map_git_identity(root: Path) -> tuple[str, str | None, str | None]:
    try:
        target = root if root.is_dir() else root.parent
        top = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--show-toplevel"],
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
        return f"git:{repo_root}:{head_sha}:{dirty_sig}", repo_root, head_sha
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


def _symbol_preview(functions: list[dict[str, Any]], classes: list[dict[str, Any]]) -> list[str]:
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
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
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


def get_file_outline(path: str) -> dict[str, Any]:
    try:
        target = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))
    if not target.exists():
        return _err(f"File not found: {path}")
    if not target.is_file():
        return _err(f"Not a file: {path}")

    try:
        source = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return _err(str(exc))

    total_lines = source.count("\n") + 1
    language = _detect_source_language(target)
    outline = _outline_for_language(source, language)
    if "parse_error" in outline:
        return {
            "status": "partial",
            "path": str(target),
            "total_lines": total_lines,
            "language": language,
            "parse_error": outline["parse_error"],
            "note": "Use read_file to inspect this file directly.",
        }

    return {
        "status": "ok",
        "path": str(target),
        "total_lines": total_lines,
        "language": language,
        **outline,
    }


def find_symbol(
    name: str,
    directory: str = ".",
    file_glob: str = "**/*",
    include_calls: bool = False,
) -> dict[str, Any]:
    if not str(name or "").strip():
        return _err("name must not be empty")
    try:
        root = resolve_tool_path(directory)
    except ValueError as exc:
        return _err(str(exc))
    if not root.is_dir():
        return _err(f"Not a directory: {directory}")

    matches: list[dict[str, Any]] = []
    max_matches = 60
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
            for match in pattern.finditer(source):
                lineno = _line_number_at_offset(source, match.start())
                dedupe_key = (kind, lineno)
                if dedupe_key in seen_lines:
                    continue
                seen_lines.add(dedupe_key)
                matches.append(
                    {
                        "file": str(fpath.relative_to(root)),
                        "line": lineno,
                        "kind": kind,
                        "language": language,
                        "signature": _extract_signature(lines, lineno, language)[:200],
                        "snippet": _snippet(lines, lineno),
                    }
                )
                if len(matches) >= max_matches:
                    break
            if len(matches) >= max_matches:
                break

        if include_calls and call_re:
            for match in call_re.finditer(source):
                lineno = _line_number_at_offset(source, match.start())
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

        if len(matches) >= max_matches:
            break

    if not matches:
        return {
            "status": "ok",
            "name": name,
            "count": 0,
            "matches": [],
            "note": f"No definition of '{name}' found. Check spelling or widen directory/glob.",
        }

    matches = _rank_symbol_matches(matches, name)
    by_language: dict[str, int] = {}
    for item in matches:
        language = str(item.get("language", "")).strip()
        if language:
            by_language[language] = by_language.get(language, 0) + 1

    return {
        "status": "ok",
        "name": name,
        "count": len(matches),
        "truncated": len(matches) >= max_matches,
        "top_files": _rollup_matches_by_file(matches, limit=8, include_kind_counts=True),
        "by_language": by_language,
        "matches": matches,
    }


def get_project_map(directory: str = ".", max_depth: int = 3, exclude: str = "") -> dict[str, Any]:
    try:
        root = resolve_tool_path(directory)
    except ValueError as exc:
        return _err(str(exc))
    if not root.is_dir():
        return _err(f"Not a directory: {directory}")

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
        return payload

    files: list[dict[str, Any]] = []
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

    payload: dict[str, Any] = {
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
    if len(_PROJECT_MAP_CACHE) > 48:
        for stale_key in list(_PROJECT_MAP_CACHE.keys())[: len(_PROJECT_MAP_CACHE) - 48]:
            _PROJECT_MAP_CACHE.pop(stale_key, None)
    return payload
