from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from .repo_registry import ensure_repo_artifacts

_GRAPH_SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".rs",
}

_GRAPH_TEXT_EXTENSIONS = {
    ".md",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}

_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+(?:.+?\s+from\s+)?|export\s+.+?\s+from\s+)["']([^"']+)["']"""
)
_JS_SYMBOL_RE = re.compile(
    r"""^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|interface|type|const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)""",
    re.MULTILINE,
)
_JS_NAMED_IMPORT_LINE_RE = re.compile(
    r"""^\s*import\s*\{([^}]+)\}\s*from\s*["']([^"']+)["']""",
)
_JS_NAMED_EXPORT_LINE_RE = re.compile(
    r"""^\s*export\s*\{([^}]+)\}(?:\s*from\s*["']([^"']+)["'])?""",
)
_JS_CALL_RE = re.compile(
    r"""\b(?:[A-Za-z_][A-Za-z0-9_]*\.)*([A-Za-z_][A-Za-z0-9_]*)\s*\(""",
)
_IDENTIFIER_RE = re.compile(r"""\b([A-Za-z_][A-Za-z0-9_]*)\b""")
_RUST_USE_RE = re.compile(r"^\s*use\s+([^;]+);", re.MULTILINE)
_RUST_MOD_RE = re.compile(r"^\s*mod\s+([A-Za-z_][A-Za-z0-9_]*);", re.MULTILINE)
_RUST_SYMBOL_RE = re.compile(
    r"^\s*(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|type)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
_RUST_IMPL_FOR_RE = re.compile(
    r"""^\s*impl(?:<[^>]+>\s*)?\s+([A-Za-z_][A-Za-z0-9_:<>]*)\s+for\s+([A-Za-z_][A-Za-z0-9_:<>]*)\s*\{?""",
)
_RUST_IMPL_RE = re.compile(
    r"""^\s*impl(?:<[^>]+>\s*)?\s+([A-Za-z_][A-Za-z0-9_:<>]*)\s*\{?""",
)
_RUST_TRAIT_DEF_RE = re.compile(
    r"""^\s*(?:pub\s+)?trait\s+([A-Za-z_][A-Za-z0-9_:<>]*)\s*\{?""",
)
_RUST_CALL_RE = re.compile(
    r"""\b(?:[A-Za-z_][A-Za-z0-9_]*::)*([A-Za-z_][A-Za-z0-9_]*)\s*\(""",
)
_RUST_QUALIFIED_CALL_RE = re.compile(
    r"""\b([A-Za-z_][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)\s*\(""",
)

_JS_TS_RESERVED_WORDS = {
    "if", "for", "while", "switch", "catch", "function", "class", "return",
    "const", "let", "var", "new", "typeof", "void", "delete", "else",
    "case", "break", "continue", "default", "import", "export", "from",
    "extends", "implements", "interface", "type", "async", "await",
    "try", "throw", "finally", "do", "in", "of", "this", "super",
}

_RUST_RESERVED_WORDS = {
    "fn", "struct", "enum", "trait", "type", "impl", "let", "mut", "pub",
    "use", "mod", "crate", "self", "super", "return", "match", "if",
    "else", "loop", "while", "for", "in", "where", "const", "static",
    "async", "await", "move", "ref", "dyn", "unsafe",
}


def _safe_jsonl_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _expand_brace_glob(glob_pattern: str) -> list[str]:
    match = re.search(r"\{([^}]+)\}", glob_pattern)
    if not match:
        return [glob_pattern]
    prefix = glob_pattern[: match.start()]
    suffix = glob_pattern[match.end() :]
    return [
        f"{prefix}{part.strip()}{suffix}"
        for part in match.group(1).split(",")
        if part.strip()
    ]


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


def _collect_matching_files(
    root: Path,
    glob_pattern: str,
    max_files: int,
    exclude: str = "",
) -> list[Path]:
    exclude_paths = _parse_exclude_paths(exclude)
    files: list[Path] = []
    for pat in _expand_brace_glob(glob_pattern):
        for fpath in sorted(root.glob(pat)):
            if not fpath.is_file() or fpath in files:
                continue
            try:
                rel_path = fpath.relative_to(root)
            except ValueError:
                continue
            if _is_excluded_relative_path(rel_path, exclude_paths):
                continue
            files.append(fpath)
            if len(files) >= max_files:
                return files
    return files


def _language_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in {".js", ".jsx"}:
        return "javascript"
    if suffix in {".ts", ".tsx"}:
        return "typescript"
    if suffix == ".rs":
        return "rust"
    if suffix in _GRAPH_TEXT_EXTENSIONS:
        return "text"
    return "unknown"


def _file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _GRAPH_SOURCE_EXTENSIONS:
        return "code"
    if suffix in {".md", ".rst", ".txt"}:
        return "doc"
    if suffix in {".toml", ".yaml", ".yml", ".json"}:
        return "config"
    return "file"


def _python_symbols_and_imports(
    rel_path: str,
    text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], []

    symbols: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                {
                    "id": f"symbol:{rel_path}:{node.name}:{node.lineno}",
                    "kind": "symbol",
                    "symbol_kind": "function",
                    "name": node.name,
                    "line": int(node.lineno or 0),
                }
            )
        elif isinstance(node, ast.ClassDef):
            symbols.append(
                {
                    "id": f"symbol:{rel_path}:{node.name}:{node.lineno}",
                    "kind": "symbol",
                    "symbol_kind": "class",
                    "name": node.name,
                    "line": int(node.lineno or 0),
                }
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "source": rel_path,
                        "module": alias.name,
                        "symbol": "",
                        "line": int(node.lineno or 0),
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(
                {
                    "source": rel_path,
                    "module": "." * int(node.level or 0) + module,
                    "symbol": ",".join(
                        alias.name for alias in node.names if str(alias.name or "").strip()
                    ),
                    "line": int(node.lineno or 0),
                }
            )
    return symbols, imports


def _parse_js_named_specifiers(spec: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for part in str(spec or "").split(","):
        item = part.strip()
        if not item:
            continue
        alias_match = re.match(
            r"""^([A-Za-z_][A-Za-z0-9_]*)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)$""",
            item,
        )
        if alias_match:
            imported = str(alias_match.group(1) or "").strip()
            local = str(alias_match.group(2) or "").strip()
        else:
            imported = item
            local = item
        if imported and local:
            pairs.append((imported, local))
    return pairs


def _js_symbols_and_imports(
    rel_path: str,
    text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    symbols: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []
    known_symbol_names: set[str] = set()
    seen_imports: set[tuple[str, str, str, int]] = set()
    for idx, line in enumerate(text.splitlines(), start=1):
        symbol_match = _JS_SYMBOL_RE.match(line)
        if symbol_match:
            name = symbol_match.group(1)
            if name not in known_symbol_names:
                symbols.append(
                    {
                        "id": f"symbol:{rel_path}:{name}:{idx}",
                        "kind": "symbol",
                        "symbol_kind": "declaration",
                        "name": name,
                        "line": idx,
                    }
                )
                known_symbol_names.add(name)
        named_import_match = _JS_NAMED_IMPORT_LINE_RE.match(line)
        if named_import_match:
            module = str(named_import_match.group(2) or "").strip()
            for imported, local in _parse_js_named_specifiers(
                str(named_import_match.group(1) or "")
            ):
                edge_key = (module, imported, local, idx)
                if edge_key in seen_imports:
                    continue
                seen_imports.add(edge_key)
                imports.append(
                    {
                        "source": rel_path,
                        "module": module,
                        "symbol": imported,
                        "local_symbol": local,
                        "line": idx,
                    }
                )
        named_export_match = _JS_NAMED_EXPORT_LINE_RE.match(line)
        if named_export_match:
            module = str(named_export_match.group(2) or "").strip()
            for exported, alias in _parse_js_named_specifiers(
                str(named_export_match.group(1) or "")
            ):
                if module:
                    edge_key = (module, exported, alias, idx)
                    if edge_key in seen_imports:
                        continue
                    seen_imports.add(edge_key)
                    imports.append(
                        {
                            "source": rel_path,
                            "module": module,
                            "symbol": exported,
                            "local_symbol": alias,
                            "line": idx,
                        }
                    )
                    continue
                if alias and alias not in known_symbol_names:
                    match_names = [alias]
                    if exported and exported != alias:
                        match_names.append(exported)
                    symbols.append(
                        {
                            "id": f"symbol:{rel_path}:{alias}:{idx}",
                            "kind": "symbol",
                            "symbol_kind": "export_alias",
                            "name": alias,
                            "line": idx,
                            "match_names": match_names,
                        }
                    )
                    known_symbol_names.add(alias)
    for match in _JS_IMPORT_RE.finditer(text):
        module = str(match.group(1) or "").strip()
        edge_key = (module, "", "", 0)
        if edge_key in seen_imports:
            continue
        seen_imports.add(edge_key)
        imports.append(
            {
                "source": rel_path,
                "module": module,
                "symbol": "",
                "line": 0,
            }
        )
    return symbols, imports


def _rust_base_name(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"<[^>]*>", "", text)
    text = text.replace("&", "").replace("mut ", "").replace("dyn ", "").strip()
    text = text.split(" where ", 1)[0].strip()
    text = text.rstrip("{").strip()
    parts = [part for part in text.split("::") if part]
    return parts[-1].strip() if parts else text


def _parse_rust_use_entries(module: str) -> list[dict[str, str]]:
    text = str(module or "").strip()
    if not text:
        return []
    brace_match = re.match(r"^(.*)::\{(.+)\}$", text)
    if brace_match:
        prefix = str(brace_match.group(1) or "").strip()
        spec = str(brace_match.group(2) or "")
        entries: list[dict[str, str]] = []
        for imported, local in _parse_js_named_specifiers(spec):
            entries.append(
                {
                    "module": prefix,
                    "symbol": imported,
                    "local_symbol": local,
                }
            )
        return entries

    alias_match = re.match(
        r"^(.*)::([A-Za-z_][A-Za-z0-9_]*)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)$",
        text,
    )
    if alias_match:
        return [
            {
                "module": str(alias_match.group(1) or "").strip(),
                "symbol": str(alias_match.group(2) or "").strip(),
                "local_symbol": str(alias_match.group(3) or "").strip(),
            }
        ]

    parts = [part.strip() for part in text.split("::") if part.strip()]
    if len(parts) >= 2:
        return [
            {
                "module": "::".join(parts[:-1]),
                "symbol": parts[-1],
                "local_symbol": parts[-1],
            }
        ]
    return [{"module": text, "symbol": "", "local_symbol": ""}]


def _rust_symbols_and_imports(
    rel_path: str,
    text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    symbols: list[dict[str, Any]] = []
    context_stack: list[dict[str, str | int]] = []
    brace_depth = 0
    for idx, line in enumerate(text.splitlines(), start=1):
        active_impl_target = ""
        active_impl_trait = ""
        active_trait_name = ""
        for context in context_stack:
            kind = str(context.get("kind") or "").strip()
            if kind == "impl":
                active_impl_target = str(context.get("target") or "").strip()
                active_impl_trait = str(context.get("trait") or "").strip()
            elif kind == "trait":
                active_trait_name = str(context.get("name") or "").strip()

        symbol_match = _RUST_SYMBOL_RE.match(line)
        if symbol_match:
            name = symbol_match.group(1)
            kind_match = re.search(r"\b(fn|struct|enum|trait|type)\b", line)
            symbol_kind = str(kind_match.group(1) if kind_match else "item")
            match_names = [name]
            if symbol_kind == "fn":
                if active_impl_target:
                    qualified = f"{_rust_base_name(active_impl_target)}::{name}"
                    if qualified not in match_names:
                        match_names.append(qualified)
                if active_impl_trait:
                    qualified = f"{_rust_base_name(active_impl_trait)}::{name}"
                    if qualified not in match_names:
                        match_names.append(qualified)
                elif active_trait_name:
                    qualified = f"{_rust_base_name(active_trait_name)}::{name}"
                    if qualified not in match_names:
                        match_names.append(qualified)
            symbols.append(
                {
                    "id": f"symbol:{rel_path}:{name}:{idx}",
                    "kind": "symbol",
                    "symbol_kind": symbol_kind,
                    "name": name,
                    "line": idx,
                    "match_names": match_names,
                }
            )
        pending_context: dict[str, str | int] | None = None
        trait_match = _RUST_IMPL_FOR_RE.match(line)
        if trait_match:
            pending_context = {
                "kind": "impl",
                "trait": _rust_base_name(str(trait_match.group(1) or "").strip()),
                "target": _rust_base_name(str(trait_match.group(2) or "").strip()),
            }
        else:
            impl_match = _RUST_IMPL_RE.match(line)
            if impl_match and not _RUST_IMPL_FOR_RE.match(line):
                pending_context = {
                    "kind": "impl",
                    "trait": "",
                    "target": _rust_base_name(str(impl_match.group(1) or "").strip()),
                }
            else:
                trait_def_match = _RUST_TRAIT_DEF_RE.match(line)
                if trait_def_match:
                    pending_context = {
                        "kind": "trait",
                        "name": _rust_base_name(str(trait_def_match.group(1) or "").strip()),
                    }
        open_count = line.count("{")
        close_count = line.count("}")
        next_depth = max(0, brace_depth + open_count - close_count)
        if pending_context and open_count > 0:
            context_stack.append(
                {
                    **pending_context,
                    "depth": next_depth,
                }
            )
        brace_depth = next_depth
        context_stack = [
            context
            for context in context_stack
            if int(context.get("depth", 0) or 0) <= brace_depth
        ]
    imports: list[dict[str, Any]] = []
    for match in _RUST_USE_RE.finditer(text):
        for entry in _parse_rust_use_entries(str(match.group(1) or "").strip()):
            imports.append(
                {
                    "source": rel_path,
                    "module": str(entry.get("module") or "").strip(),
                    "symbol": str(entry.get("symbol") or "").strip(),
                    "local_symbol": str(entry.get("local_symbol") or "").strip(),
                    "line": 0,
                }
            )
    for match in _RUST_MOD_RE.finditer(text):
        imports.append(
            {
                "source": rel_path,
                "module": f"mod:{str(match.group(1) or '').strip()}",
                "symbol": "",
                "line": 0,
            }
        )
    return symbols, imports


def _extract_symbols_and_imports(
    rel_path: str,
    path: Path,
    text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    language = _language_for_path(path)
    if language == "python":
        return _python_symbols_and_imports(rel_path, text)
    if language in {"javascript", "typescript"}:
        return _js_symbols_and_imports(rel_path, text)
    if language == "rust":
        return _rust_symbols_and_imports(rel_path, text)
    return [], []


def _python_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return str(node.id or "").strip()
    if isinstance(node, ast.Attribute):
        return str(node.attr or "").strip()
    return ""


def _python_call_and_reference_signals(text: str) -> tuple[set[str], set[str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set(), set()

    calls: set[str] = set()
    references: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _python_call_name(node.func)
            if name:
                calls.add(name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = str(node.id or "").strip()
            if name:
                references.add(name)
    return calls, references


def _js_call_and_reference_signals(text: str) -> tuple[set[str], set[str]]:
    calls = {
        str(match.group(1) or "").strip()
        for match in _JS_CALL_RE.finditer(text)
        if str(match.group(1) or "").strip()
        and str(match.group(1) or "").strip() not in _JS_TS_RESERVED_WORDS
    }
    references = {
        str(match.group(1) or "").strip()
        for match in _IDENTIFIER_RE.finditer(text)
        if str(match.group(1) or "").strip()
        and str(match.group(1) or "").strip() not in _JS_TS_RESERVED_WORDS
    }
    return calls, references


def _rust_call_and_reference_signals(text: str) -> tuple[set[str], set[str]]:
    calls = {
        str(match.group(1) or "").strip()
        for match in _RUST_CALL_RE.finditer(text)
        if str(match.group(1) or "").strip()
        and str(match.group(1) or "").strip() not in _RUST_RESERVED_WORDS
    }
    for match in _RUST_QUALIFIED_CALL_RE.finditer(text):
        qualifier = str(match.group(1) or "").strip()
        name = str(match.group(2) or "").strip()
        if qualifier and name:
            calls.add(f"{qualifier}::{name}")
    references = {
        str(match.group(1) or "").strip()
        for match in _IDENTIFIER_RE.finditer(text)
        if str(match.group(1) or "").strip()
        and str(match.group(1) or "").strip() not in _RUST_RESERVED_WORDS
    }
    return calls, references


def _extract_reference_signals(path: Path, text: str) -> tuple[set[str], set[str]]:
    language = _language_for_path(path)
    if language == "python":
        return _python_call_and_reference_signals(text)
    if language in {"javascript", "typescript"}:
        return _js_call_and_reference_signals(text)
    if language == "rust":
        return _rust_call_and_reference_signals(text)
    return set(), set()


def _normalized_signal_sets(
    calls: set[str],
    references: set[str],
    imports: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    alias_map: dict[str, str] = {}
    for item in imports:
        local = str(item.get("local_symbol") or "").strip()
        symbol = str(item.get("symbol") or "").strip()
        if local and symbol and local != symbol:
            alias_map[local] = symbol

    norm_calls = set(calls)
    norm_refs = set(references)

    for name in list(calls):
        mapped = alias_map.get(name)
        if mapped:
            norm_calls.add(mapped)
        if "::" in name:
            qualifier, member = name.split("::", 1)
            mapped_qualifier = alias_map.get(qualifier)
            if mapped_qualifier:
                norm_calls.add(f"{mapped_qualifier}::{member}")

    for name in list(references):
        mapped = alias_map.get(name)
        if mapped:
            norm_refs.add(mapped)
        if "::" in name:
            qualifier, member = name.split("::", 1)
            mapped_qualifier = alias_map.get(qualifier)
            if mapped_qualifier:
                norm_refs.add(f"{mapped_qualifier}::{member}")

    return norm_calls, norm_refs


def _resolve_python_import(module: str, source_rel_path: str, file_index: set[str]) -> str | None:
    source_path = Path(source_rel_path)
    level = 0
    while module.startswith("."):
        level += 1
        module = module[1:]
    base_dir = source_path.parent
    if level > 1:
        for _ in range(level - 1):
            base_dir = base_dir.parent
    module_parts = [part for part in module.split(".") if part]
    candidate_parts = list(base_dir.parts) + module_parts
    if candidate_parts:
        file_candidate = Path(*candidate_parts).with_suffix(".py").as_posix()
        if file_candidate in file_index:
            return file_candidate
        init_candidate = Path(*candidate_parts, "__init__.py").as_posix()
        if init_candidate in file_index:
            return init_candidate
    return None


def _resolve_js_import(module: str, source_rel_path: str, file_index: set[str]) -> str | None:
    if not module.startswith("."):
        return None
    base = (Path(source_rel_path).parent / module).as_posix()
    base_path = Path(base)
    candidates = [
        base_path.as_posix(),
        f"{base}.ts",
        f"{base}.tsx",
        f"{base}.js",
        f"{base}.jsx",
        str(base_path / "index.ts"),
        str(base_path / "index.tsx"),
        str(base_path / "index.js"),
        str(base_path / "index.jsx"),
    ]
    for candidate in candidates:
        normalized = Path(candidate).as_posix()
        if normalized in file_index:
            return normalized
    return None


def _resolve_rust_import(module: str, source_rel_path: str, file_index: set[str]) -> str | None:
    if module.startswith("mod:"):
        name = module.split(":", 1)[1].strip()
        source_path = Path(source_rel_path)
        candidates = [
            str(source_path.parent / f"{name}.rs"),
            str(source_path.parent / name / "mod.rs"),
        ]
        for candidate in candidates:
            normalized = Path(candidate).as_posix()
            if normalized in file_index:
                return normalized
        return None

    parts = [part for part in module.split("::") if part and part != "crate"]
    if not parts:
        return None
    candidates = [
        str(Path(*parts).with_suffix(".rs")),
        str(Path(*parts, "mod.rs")),
    ]
    for candidate in candidates:
        normalized = Path(candidate).as_posix()
        if normalized in file_index:
            return normalized
    return None


def _resolve_import_target(module: str, source_rel_path: str, path: Path, file_index: set[str]) -> str | None:
    language = _language_for_path(path)
    if language == "python":
        return _resolve_python_import(module, source_rel_path, file_index)
    if language in {"javascript", "typescript"}:
        return _resolve_js_import(module, source_rel_path, file_index)
    if language == "rust":
        return _resolve_rust_import(module, source_rel_path, file_index)
    return None


def build_repo_graph(
    repo: dict[str, Any],
    *,
    glob_pattern: str,
    max_files: int,
    exclude: str = "",
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    repo_id = str(repo.get("id") or "").strip()
    repo_name = str(repo.get("name") or repo_id or "repo").strip()
    repo_path = Path(str(repo.get("path") or "")).expanduser().resolve()
    if not repo_id or not repo_path.is_dir():
        raise FileNotFoundError(f"Invalid repo for graph build: {repo}")

    artifacts = ensure_repo_artifacts(
        repo_id,
        name=repo_name,
        path=str(repo_path),
        base_dir=base_dir,
    )
    graph_path = Path(artifacts["graph_path"])
    summary_path = Path(artifacts["summary_path"])

    files = _collect_matching_files(
        repo_path,
        glob_pattern=glob_pattern,
        max_files=max_files,
        exclude=exclude,
    )
    file_index = {
        str(path.relative_to(repo_path).as_posix())
        for path in files
    }

    records: list[dict[str, Any]] = []
    records.append(
        {
            "record_type": "node",
            "id": f"repo:{repo_id}",
            "kind": "repo",
            "repo_id": repo_id,
            "name": repo_name,
            "path": str(repo_path),
        }
    )

    node_count = 1
    edge_count = 0
    symbol_count = 0
    import_edge_count = 0
    call_edge_count = 0
    reference_edge_count = 0
    resolved_import_count = 0
    unresolved_import_count = 0
    language_counts: dict[str, int] = {}
    file_analysis: dict[str, dict[str, Any]] = {}
    symbol_defs_by_name: dict[str, list[dict[str, Any]]] = {}

    for path in files:
        rel_path = path.relative_to(repo_path).as_posix()
        language = _language_for_path(path)
        language_counts[language] = language_counts.get(language, 0) + 1
        file_node_id = f"file:{rel_path}"
        records.append(
            {
                "record_type": "node",
                "id": file_node_id,
                "kind": "file",
                "repo_id": repo_id,
                "name": path.name,
                "path": str(path),
                "rel_path": rel_path,
                "language": language,
                "file_kind": _file_kind(path),
            }
        )
        node_count += 1
        records.append(
            {
                "record_type": "edge",
                "kind": "contains",
                "repo_id": repo_id,
                "source": f"repo:{repo_id}",
                "target": file_node_id,
            }
        )
        edge_count += 1

        if path.suffix.lower() not in _GRAPH_SOURCE_EXTENSIONS:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        symbols, imports = _extract_symbols_and_imports(rel_path, path, text)
        calls, references = _extract_reference_signals(path, text)
        calls, references = _normalized_signal_sets(calls, references, imports)
        file_analysis[rel_path] = {
            "path": path,
            "language": language,
            "symbols": symbols,
            "imports": imports,
            "calls": calls,
            "references": references,
        }
        for symbol in symbols:
            symbol_node = {
                "record_type": "node",
                "repo_id": repo_id,
                "path": str(path),
                "rel_path": rel_path,
                "language": language,
                **symbol,
            }
            records.append(symbol_node)
            node_count += 1
            symbol_count += 1
            match_names = [
                str(name or "").strip()
                for name in list(symbol.get("match_names") or []) or [symbol.get("name")]
                if str(name or "").strip()
            ]
            for name in match_names:
                symbol_defs_by_name.setdefault(name, []).append(
                    {
                        "id": str(symbol.get("id") or "").strip(),
                        "name": str(symbol.get("name") or "").strip(),
                        "rel_path": rel_path,
                        "symbol_kind": str(symbol.get("symbol_kind") or "").strip(),
                        "line": int(symbol.get("line", 0) or 0),
                    }
                )
            records.append(
                {
                    "record_type": "edge",
                    "kind": "defines",
                    "repo_id": repo_id,
                    "source": file_node_id,
                    "target": symbol["id"],
                }
            )
            edge_count += 1

    seen_relation_edges: set[tuple[str, str, str]] = set()
    for rel_path, analysis in file_analysis.items():
        path = analysis["path"]
        file_node_id = f"file:{rel_path}"
        for item in analysis["imports"]:
            module = str(item.get("module") or "").strip()
            if not module:
                continue
            target_rel = _resolve_import_target(module, rel_path, path, file_index)
            edge = {
                "record_type": "edge",
                "kind": "imports",
                "repo_id": repo_id,
                "source": file_node_id,
                "target": f"file:{target_rel}" if target_rel else "",
                "target_rel_path": target_rel or "",
                "import_path": module,
                "symbol_name": str(item.get("symbol") or "").strip(),
                "local_symbol": str(item.get("local_symbol") or "").strip(),
                "line": int(item.get("line", 0) or 0),
            }
            if target_rel:
                resolved_import_count += 1
            else:
                unresolved_import_count += 1
            import_edge_count += 1
            edge_count += 1
            records.append(edge)

        for call_name in sorted(analysis["calls"]):
            for symbol in symbol_defs_by_name.get(call_name, []):
                edge_key = (file_node_id, "calls", str(symbol.get("id") or ""))
                if not edge_key[2] or edge_key in seen_relation_edges:
                    continue
                seen_relation_edges.add(edge_key)
                records.append(
                    {
                        "record_type": "edge",
                        "kind": "calls",
                        "repo_id": repo_id,
                        "source": file_node_id,
                        "target": str(symbol.get("id") or ""),
                        "target_rel_path": str(symbol.get("rel_path") or ""),
                        "symbol_name": call_name,
                        "symbol_kind": str(symbol.get("symbol_kind") or ""),
                        "line": int(symbol.get("line", 0) or 0),
                    }
                )
                edge_count += 1
                call_edge_count += 1

        for ref_name in sorted(analysis["references"]):
            for symbol in symbol_defs_by_name.get(ref_name, []):
                edge_key = (file_node_id, "references", str(symbol.get("id") or ""))
                if not edge_key[2] or edge_key in seen_relation_edges:
                    continue
                seen_relation_edges.add(edge_key)
                records.append(
                    {
                        "record_type": "edge",
                        "kind": "references",
                        "repo_id": repo_id,
                        "source": file_node_id,
                        "target": str(symbol.get("id") or ""),
                        "target_rel_path": str(symbol.get("rel_path") or ""),
                        "symbol_name": ref_name,
                        "symbol_kind": str(symbol.get("symbol_kind") or ""),
                        "line": int(symbol.get("line", 0) or 0),
                    }
                )
                edge_count += 1
                reference_edge_count += 1

    graph_path.write_text(
        "\n".join(_safe_jsonl_line(record) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )

    summary_lines = [
        f"# {repo_name}",
        "",
        f"- repo_id: {repo_id}",
        f"- path: {repo_path}",
        f"- graph_status: ready",
        f"- files_indexed: {len(files)}",
        f"- nodes: {node_count}",
        f"- edges: {edge_count}",
        f"- symbols: {symbol_count}",
        f"- import_edges: {import_edge_count}",
        f"- call_edges: {call_edge_count}",
        f"- reference_edges: {reference_edge_count}",
        f"- resolved_import_edges: {resolved_import_count}",
        f"- unresolved_import_edges: {unresolved_import_count}",
        "",
        "## Languages",
        "",
    ]
    for language, count in sorted(language_counts.items()):
        summary_lines.append(f"- {language}: {count}")
    summary_lines.append("")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "status": "ok",
        "repo_id": repo_id,
        "repo_name": repo_name,
        "graph_path": str(graph_path),
        "summary_path": str(summary_path),
        "files_indexed": len(files),
        "nodes": node_count,
        "edges": edge_count,
        "symbols": symbol_count,
        "import_edges": import_edge_count,
        "call_edges": call_edge_count,
        "reference_edges": reference_edge_count,
        "resolved_import_edges": resolved_import_count,
        "unresolved_import_edges": unresolved_import_count,
        "language_counts": language_counts,
    }


def load_repo_graph(repo: dict[str, Any]) -> dict[str, Any]:
    artifacts = dict(repo.get("artifacts") or {})
    graph_path = Path(str(artifacts.get("graph_path") or "")).expanduser()
    if not graph_path.exists():
        return {"nodes": {}, "edges": []}

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for line in graph_path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("record_type") or "") == "node":
            key = str(payload.get("id") or "").strip()
            if key:
                nodes[key] = payload
        elif str(payload.get("record_type") or "") == "edge":
            edges.append(payload)
    return {"nodes": nodes, "edges": edges}


def related_repo_context(
    repo: dict[str, Any],
    *,
    rel_paths: list[str],
    query: str = "",
    limit: int = 8,
) -> dict[str, Any]:
    graph = load_repo_graph(repo)
    nodes = dict(graph.get("nodes") or {})
    edges = list(graph.get("edges") or [])
    seed_paths = [str(path or "").strip() for path in rel_paths if str(path or "").strip()]
    if not seed_paths:
        return {"related_files": [], "related_symbols": []}

    file_scores: dict[str, int] = {path: 100 - idx * 10 for idx, path in enumerate(seed_paths)}
    symbol_rows: list[dict[str, Any]] = []
    query_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(query or ""))
    }

    source_ids = {f"file:{path}" for path in seed_paths}
    for edge in edges:
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        kind = str(edge.get("kind") or "").strip()
        target_rel_path = str(edge.get("target_rel_path") or "").strip()

        if source in source_ids and kind == "imports" and target_rel_path:
            file_scores[target_rel_path] = max(file_scores.get(target_rel_path, 0), 70)

        if source in source_ids and kind == "calls" and target_rel_path:
            file_scores[target_rel_path] = max(file_scores.get(target_rel_path, 0), 82)

        if source in source_ids and kind == "references" and target_rel_path:
            file_scores[target_rel_path] = max(file_scores.get(target_rel_path, 0), 74)

        if source in source_ids and kind in {"defines", "calls", "references"} and target:
            node = nodes.get(target, {})
            symbol_rows.append(
                {
                    "name": str(node.get("name") or "").strip(),
                    "symbol_kind": str(node.get("symbol_kind") or "").strip(),
                    "rel_path": str(node.get("rel_path") or "").strip(),
                    "line": int(node.get("line", 0) or 0),
                }
            )

        if kind == "imports" and target in source_ids:
            importer = str(edge.get("source") or "").strip()
            if importer.startswith("file:"):
                importer_rel = importer.split("file:", 1)[1]
                file_scores[importer_rel] = max(file_scores.get(importer_rel, 0), 65)

        if kind == "calls" and target_rel_path in seed_paths:
            caller = str(edge.get("source") or "").strip()
            if caller.startswith("file:"):
                caller_rel = caller.split("file:", 1)[1]
                file_scores[caller_rel] = max(file_scores.get(caller_rel, 0), 86)

        if kind == "references" and target_rel_path in seed_paths:
            referrer = str(edge.get("source") or "").strip()
            if referrer.startswith("file:"):
                referrer_rel = referrer.split("file:", 1)[1]
                file_scores[referrer_rel] = max(file_scores.get(referrer_rel, 0), 76)

    if query_terms:
        for node in nodes.values():
            if str(node.get("kind") or "") != "symbol":
                continue
            name = str(node.get("name") or "").strip()
            if not name:
                continue
            lname = name.lower()
            if not any(term in lname for term in query_terms):
                continue
            rel_path = str(node.get("rel_path") or "").strip()
            if rel_path:
                file_scores[rel_path] = max(file_scores.get(rel_path, 0), 60)
            symbol_rows.append(
                {
                    "name": name,
                    "symbol_kind": str(node.get("symbol_kind") or "").strip(),
                    "rel_path": rel_path,
                    "line": int(node.get("line", 0) or 0),
                }
            )

    related_files = [
        {"rel_path": rel_path, "score": score}
        for rel_path, score in sorted(
            file_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if rel_path not in seed_paths
    ][: max(0, int(limit))]

    dedup_symbols: list[dict[str, Any]] = []
    seen_symbol_keys: set[tuple[str, str, int]] = set()
    for row in symbol_rows:
        key = (
            str(row.get("name") or ""),
            str(row.get("rel_path") or ""),
            int(row.get("line", 0) or 0),
        )
        if key in seen_symbol_keys:
            continue
        seen_symbol_keys.add(key)
        dedup_symbols.append(row)
        if len(dedup_symbols) >= limit:
            break

    return {
        "related_files": related_files,
        "related_symbols": dedup_symbols[:limit],
    }
