"""Code exploration tools — structural understanding without reading full files.

These tools are the first stop when working with an unfamiliar codebase.
Use them BEFORE read_file to avoid reading thousands of lines unnecessarily.

Tool inventory
--------------
get_file_outline  -- AST-based: list all imports, classes, functions, and line numbers
find_symbol       -- locate every definition (def/class) matching a name across the codebase
get_project_map   -- one-line summary of every .py file in a directory
rg_search         -- fast ripgrep text search with context lines (falls back to pure Python)
fd_find           -- fast file/directory finder by name pattern (falls back to pure Python glob)
"""

from __future__ import annotations

import json as _json_mod
from typing import Any

if "llm" not in globals():

    class _NoOpLLM:
        def tool(self, func=None, *, name=None, description=None):
            return func if func is not None else (lambda f: f)

    llm = _NoOpLLM()

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
from pathlib import Path

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@llm.tool(
    description=(
        "Return a structural outline of a source file: all imports, classes (with methods), "
        "and module-level functions with their line numbers. "
        "Use this BEFORE read_file to understand a file without reading all of it."
    )
)
def get_file_outline(path: str) -> str:
    """Use when: You need to understand what's in a file before diving into it.

    Triggers: outline file, file structure, what functions, what classes, methods in file,
              understand codebase, look at file, explore file, inspect module, map code.
    Avoid when: You already know the exact line range you need — use read_file directly.
    Inputs:
      path (str, required): Absolute or relative path to the source file.
    Returns: JSON with imports, classes (with methods), and functions — all with line numbers.
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
    suffix = p.suffix.lstrip(".").lower()

    if suffix == "py":
        lang = "python"
        outline = _outline_python(source)
    elif suffix in ("js", "ts", "jsx", "tsx"):
        lang = suffix
        outline = _outline_generic(source, suffix)
    else:
        lang = suffix or "text"
        outline = _outline_generic(source, suffix)

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


@llm.tool(
    description=(
        "Find every definition of a function or class by name across the codebase. "
        "Returns file path, line number, and a short source snippet for each match. "
        "Use this to 'go to definition' without knowing which file contains it."
    )
)
def find_symbol(
    name: str,
    directory: str = ".",
    file_glob: str = "**/*.py",
    include_calls: bool = False,
) -> str:
    """Use when: You want to jump to the definition of a specific function or class.

    Triggers: find function, where is defined, go to definition, locate class, find symbol,
              where does X live, which file defines, find implementation.
    Avoid when: You already know the file — use read_file or get_file_outline.
    Inputs:
      name (str, required): Exact function or class name to find.
      directory (str, optional): Root directory to search (default ".").
      file_glob (str, optional): File pattern (default "**/*.py").
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

    # Regex: match def/class at start of line (handles indented methods too)
    def_re = re.compile(
        rf"^([ \t]*)(?:(async\s+)?def|class)\s+({re.escape(name)})\s*[\(:]",
        re.MULTILINE,
    )
    call_re = (
        re.compile(rf"\b{re.escape(name)}\s*\(", re.MULTILINE)
        if include_calls
        else None
    )

    matches: list[dict] = []
    MAX = 60

    for fpath in sorted(root.glob(file_glob)):
        if not fpath.is_file():
            continue
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines = source.splitlines()

        for m in def_re.finditer(source):
            lineno = source[: m.start()].count("\n") + 1
            indent = len(m.group(1))
            kind = (
                "method" if indent > 0 else ("def" if "def" in m.group(0) else "class")
            )
            # grab signature: the line itself + up to 2 more for multi-line signatures
            sig_lines = []
            for i in range(lineno - 1, min(lineno + 2, len(lines))):
                sig_lines.append(lines[i])
                if ":" in lines[i] and not lines[i].rstrip().endswith(","):
                    break
            signature = " ".join(ln.strip() for ln in sig_lines)

            # 2-line context snippet
            ctx_start = max(0, lineno - 2)
            ctx_end = min(len(lines), lineno + 2)
            snippet = "\n".join(
                f"{ctx_start + i + 1}: {ln}"
                for i, ln in enumerate(lines[ctx_start:ctx_end])
            )

            matches.append(
                {
                    "file": str(fpath.relative_to(root)),
                    "line": lineno,
                    "kind": kind,
                    "signature": signature[:200],
                    "snippet": snippet,
                }
            )
            if len(matches) >= MAX:
                break

        if include_calls and call_re:
            for m in call_re.finditer(source):
                lineno = source[: m.start()].count("\n") + 1
                ctx_start = max(0, lineno - 2)
                ctx_end = min(len(lines), lineno + 1)
                snippet = "\n".join(
                    f"{ctx_start + i + 1}: {ln}"
                    for i, ln in enumerate(lines[ctx_start:ctx_end])
                )
                matches.append(
                    {
                        "file": str(fpath.relative_to(root)),
                        "line": lineno,
                        "kind": "call",
                        "signature": lines[lineno - 1].strip()[:200],
                        "snippet": snippet,
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

    return _safe_json(
        {
            "status": "ok",
            "name": name,
            "count": len(matches),
            "truncated": len(matches) >= MAX,
            "matches": matches,
        }
    )


@llm.tool(
    description=(
        "Scan every Python file in a directory and return a one-line summary of each: "
        "module docstring (or first comment), number of functions/classes. "
        "Use this to quickly understand the layout of a whole package."
    )
)
def get_project_map(directory: str = ".", max_depth: int = 3) -> str:
    """Use when: You need a bird's-eye view of a package or project.

    Triggers: map project, project structure, package overview, what files do what,
              understand repo, show all modules, codebase overview.
    Avoid when: You need function-level detail — use get_file_outline per file.
    Inputs:
      directory (str, optional): Root of the project to scan (default ".").
      max_depth (int, optional): Folder depth limit (default 3).
    Returns: JSON list of {path, docstring, n_functions, n_classes, lines}.
    Side effects: Read-only.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return _safe_json({"status": "error", "error": f"Not a directory: {directory}"})

    files: list[dict] = []

    for fpath in sorted(root.rglob("*.py")):
        # depth check
        try:
            rel = fpath.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) - 1 > max_depth:
            continue
        if any(part.startswith((".", "_")) for part in rel.parts[:-1]):
            continue

        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        total_lines = source.count("\n") + 1
        n_functions = len(
            re.findall(r"^\s*(?:async\s+)?def\s+\w+", source, re.MULTILINE)
        )
        n_classes = len(re.findall(r"^\s*class\s+\w+", source, re.MULTILINE))

        # Try to get module docstring or first meaningful comment
        first_doc = ""
        try:
            tree = ast.parse(source)
            doc = ast.get_docstring(tree)
            if doc:
                first_doc = doc.splitlines()[0][:100]
        except SyntaxError:
            pass
        if not first_doc:
            for line in source.splitlines()[:10]:
                stripped = line.strip()
                if stripped.startswith("#") and len(stripped) > 2:
                    first_doc = stripped[1:].strip()[:100]
                    break

        files.append(
            {
                "path": str(rel),
                "lines": total_lines,
                "n_functions": n_functions,
                "n_classes": n_classes,
                "summary": first_doc,
            }
        )

    return _safe_json(
        {
            "status": "ok",
            "root": str(root),
            "file_count": len(files),
            "files": files,
        }
    )


# ---------------------------------------------------------------------------
# Internal: run-command helper (reuse bootstrap's if available, else fallback)
# ---------------------------------------------------------------------------


def _explore_run(cmd: str, cwd: str | None = None, timeout: int = 30) -> dict:
    """Run a command using bootstrap's _run_cmd if available, else subprocess directly."""
    _rc = globals().get("_run_cmd")
    if _rc is not None:
        return _rc(cmd, cwd=cwd, timeout=timeout)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
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


@llm.tool(
    description=(
        "Search for a pattern across files using ripgrep (rg) — much faster than grep on large "
        "codebases. Falls back to pure-Python search if rg is not installed. "
        "Returns matching lines with file path, line number, and optional context."
    )
)
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

        truncated = len(matches) >= max_results
        return _safe_json(
            {
                "status": "ok",
                "tool_used": "rg",
                "pattern": pattern,
                "count": len([m for m in matches if m.get("match", True)]),
                "truncated": truncated,
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

    real_count = len([m for m in matches if m.get("match")])
    return _safe_json(
        {
            "status": "ok",
            "tool_used": "python",
            "pattern": pattern,
            "count": real_count,
            "truncated": real_count >= max_results,
            "matches": matches,
        }
    )


@llm.tool(
    description=(
        "Find files or directories by name pattern using fd (fast alternative to find). "
        "Falls back to pure-Python glob if fd is not installed. "
        "Use to locate files without knowing their exact path."
    )
)
def fd_find(
    pattern: str,
    directory: str = ".",
    file_type: str = "f",
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
                               "config", "test_.*\.py", "__init__".
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
