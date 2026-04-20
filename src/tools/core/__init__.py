"""Always-on core tools for the SOTA agent."""

import importlib as _importlib
import json
from pathlib import Path
from typing import Any

from ..compaction import (
    ContentReplacementState,
    compact_result,
)
from ..runtime import build_tool
from .BashTool.tool import bash
from .BashTool.tool import _validate_command as _validate_bash_command
from .FileEditTool.tool import (
    apply_edit_block,
    edit_file,
    preview_edit,
    smart_edit,
    write_file,
)
from .FilesystemTool.tool import delete_path, mkdir, move_path
from .FileReadTool.tool import list_dir, read_edit_context, read_file
from .GitTool.tool import (
    get_git_diff,
    get_git_status,
)
from .LspTool.tool import lsp_tool
from .MetaTool.tool import tool_search
from .NotebookTool.tool import notebook_edit
from .ProcessTool.tool import (
    get_process_output,
    install_packages,
    kill_process,
    list_processes,
    send_input_to_process,
    set_venv,
    set_working_directory,
    show_coding_config,
    start_background_process,
)
from .ProjectTool.tool import find_symbol, get_file_outline, get_project_map
from .PythonTool.tool import check_imports, list_installed_packages, run_python
from .RustTool.tool import (
    cargo_build,
    cargo_check,
    cargo_clippy,
    cargo_fmt,
    cargo_metadata,
    cargo_run,
    cargo_test,
    run_rust,
)
from .SearchTool.inspection import find_imports, get_symbol_info, read_line
from .SearchTool.tool import (
    fd_find,
    find_references,
    glob_files,
    grep_files,
    rg_search,
    search_code,
    search_file,
    search_symbols,
)
from .TaskTool.tool import think, todo
from .WebTool.tool import fetch_url, github_read_file, pypi_info, web_search

# ---------------------------------------------------------------------
# Optional LibCST support
# ---------------------------------------------------------------------
_LIBCST_AVAILABLE = False

_LIBCST_MODULE: Any = None  # loaded on first use
_LIBCST_CHECKED = False


def _check_libcst() -> bool:
    global _LIBCST_CHECKED, _LIBCST_MODULE, _LIBCST_AVAILABLE
    if _LIBCST_CHECKED:
        return _LIBCST_AVAILABLE
    _LIBCST_CHECKED = True
    try:
        import libcst  # noqa: F401
        _LIBCST_MODULE = _importlib.import_module(".FileEditTool.libcst", package=__name__)
        _LIBCST_AVAILABLE = True
    except ImportError:
        _LIBCST_AVAILABLE = False
    return _LIBCST_AVAILABLE


def _libcst_not_available(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {
        "status": "error",
        "error": "libcst is not installed. Install it with: pip install libcst",
    }


def _make_libcst_proxy(name: str):
    def _proxy(*args: Any, **kwargs: Any) -> Any:
        if _check_libcst() and _LIBCST_MODULE is not None:
            return getattr(_LIBCST_MODULE, name)(*args, **kwargs)
        return _libcst_not_available(*args, **kwargs)
    _proxy.__name__ = name
    return _proxy


edit_file_libcst = _make_libcst_proxy("edit_file_libcst")
replace_function_body = _make_libcst_proxy("replace_function_body")
replace_docstring = _make_libcst_proxy("replace_docstring")
replace_decorators = _make_libcst_proxy("replace_decorators")
replace_argument = _make_libcst_proxy("replace_argument")
insert_after_function = _make_libcst_proxy("insert_after_function")
delete_function = _make_libcst_proxy("delete_function")
find_function_by_name = _make_libcst_proxy("find_function_by_name")
find_class_by_name = _make_libcst_proxy("find_class_by_name")


_RUNTIME_META = {
    "read_file": {"read_only": True, "cacheable": True, "content_reader": True},
    "read_edit_context": {"read_only": True, "cacheable": True, "content_reader": True},
    "search_file": {"read_only": True, "cacheable": True, "content_reader": True},
    "list_dir": {"read_only": True, "cacheable": True},
    "preview_edit": {"read_only": True, "cacheable": False},
    "get_git_status": {"read_only": True, "cacheable": False},
    "get_git_diff": {"read_only": True, "cacheable": False},
    "get_symbol_info": {"read_only": True, "cacheable": True, "content_reader": True},
    "read_line": {"read_only": True, "cacheable": True, "content_reader": True},
    "find_imports": {"read_only": True, "cacheable": True, "content_reader": True},
    "get_file_outline": {"read_only": True, "cacheable": True, "content_reader": True},
    "find_symbol": {"read_only": True, "cacheable": True, "content_reader": True},
    "get_project_map": {"read_only": True, "cacheable": True},
    "fetch_url": {
        "read_only": True,
        "cacheable": False,
        "content_reader": True,
        "concurrency_safe": True,
    },
    "web_search": {"read_only": True, "cacheable": False, "concurrency_safe": True},
    "pypi_info": {"read_only": True, "cacheable": False, "concurrency_safe": True},
    "github_read_file": {
        "read_only": True,
        "cacheable": False,
        "content_reader": True,
        "concurrency_safe": True,
    },
    "set_venv": {"cacheable": False},
    "set_working_directory": {"cacheable": False},
    "install_packages": {"cacheable": False},
    "show_coding_config": {"read_only": True, "cacheable": False},
    "start_background_process": {"cacheable": False},
    "send_input_to_process": {"cacheable": False},
    "get_process_output": {"read_only": True, "cacheable": False},
    "kill_process": {"cacheable": False},
    "list_processes": {"read_only": True, "cacheable": False},
    "run_python": {"cacheable": False},
    "list_installed_packages": {"read_only": True, "cacheable": False},
    "check_imports": {"read_only": True, "cacheable": False},
    "run_rust": {"cacheable": False},
    "cargo_check": {"cacheable": False},
    "cargo_build": {"cacheable": False},
    "cargo_test": {"cacheable": False},
    "cargo_clippy": {"cacheable": False},
    "cargo_run": {"cacheable": False},
    "cargo_fmt": {"cacheable": False},
    "cargo_metadata": {"read_only": True, "cacheable": False},
    "lsp_tool": {
        "read_only": True,
        "cacheable": True,
        "content_reader": True,
        "concurrency_safe": True,
    },
    "glob_files": {"read_only": True, "cacheable": True},
    "grep_files": {"read_only": True, "cacheable": True},
    "search_code": {"read_only": True, "cacheable": True},
    "think": {"read_only": True, "cacheable": False},
    "write_file": {"writes_files": True, "cacheable": False},
    "edit_file": {"writes_files": True, "cacheable": False},
    "apply_edit_block": {"writes_files": True, "cacheable": False},
    "smart_edit": {"writes_files": True, "cacheable": False},
    "notebook_edit": {"writes_files": True, "cacheable": False},
    "mkdir": {"writes_files": True, "cacheable": False},
    "move_path": {"writes_files": True, "cacheable": False},
    "delete_path": {"writes_files": True, "cacheable": False},
    "edit_file_libcst": {"writes_files": True, "cacheable": False},
    "replace_function_body": {"writes_files": True, "cacheable": False},
    "replace_docstring": {"writes_files": True, "cacheable": False},
    "replace_decorators": {"writes_files": True, "cacheable": False},
    "replace_argument": {"writes_files": True, "cacheable": False},
    "insert_after_function": {"writes_files": True, "cacheable": False},
    "delete_function": {"writes_files": True, "cacheable": False},
    "find_function_by_name": {"read_only": True, "cacheable": True, "content_reader": True},
    "find_class_by_name": {"read_only": True, "cacheable": True, "content_reader": True},
    "tool_search": {"read_only": True, "cacheable": False, "concurrency_safe": True},
}

_TOOL_META = {
    "bash": {
        "description": "Execute a shell command and return structured stdout, stderr, and exit metadata.",
        "parameters": {
            "command": "Shell command string to execute.",
            "timeout": "Optional timeout in seconds.",
            "normalize_output": "Optional boolean to normalize stdout/stderr newlines to LF.",
            "require_read_only": "Optional boolean. When true, reject commands that are not classified as read-only inspection commands.",
        },
    },
    "fetch_url": {
        "description": "Fetch a webpage and return cleaned text content plus response metadata.",
        "parameters": {
            "url": "Full URL including scheme such as `https://docs.python.org/3/library/pathlib.html`.",
            "timeout": "Optional request timeout in seconds.",
            "max_chars": "Optional maximum number of text characters to return.",
        },
    },
    "web_search": {
        "description": "Search the web for documentation, API references, or current error explanations.",
        "parameters": {
            "query": "Search query string.",
            "n": "Optional maximum number of results to return.",
        },
    },
    "pypi_info": {
        "description": "Fetch package metadata from PyPI including latest version and project URLs.",
        "parameters": {
            "package_name": "PyPI package name such as `numpy` or `requests`.",
        },
    },
    "github_read_file": {
        "description": "Read a text file from a public GitHub repository via raw.githubusercontent.com.",
        "parameters": {
            "owner": "GitHub owner or organization.",
            "repo": "Repository name.",
            "path": "File path inside the repository.",
            "ref": "Optional branch, tag, or commit SHA. Defaults to `main`.",
        },
    },
    "set_venv": {
        "description": "Set the default virtualenv for subsequent shell and Python execution tools.",
        "parameters": {
            "venv_path": "Path to the virtualenv root, such as `/project/.venv`.",
        },
    },
    "set_working_directory": {
        "description": "Set the default working directory for subsequent shell and Python execution tools.",
        "parameters": {
            "path": "Absolute or relative path to the target directory.",
        },
    },
    "install_packages": {
        "description": "Install pip packages into the active or specified virtualenv.",
        "parameters": {
            "packages": "Space-separated package names.",
            "venv_path": "Optional virtualenv path overriding the configured default.",
            "upgrade": "Optional boolean. When true, pass `--upgrade` to pip.",
        },
    },
    "show_coding_config": {
        "description": "Show the currently configured virtualenv and working directory for execution tools.",
        "parameters": {},
    },
    "start_background_process": {
        "description": "Start a long-running command in the background and keep its buffered output in shared execution state.",
        "parameters": {
            "command": "Shell command to run in the background.",
            "name": "Unique process label used by the process-control tools.",
            "cwd": "Optional working directory for the process.",
            "venv_path": "Optional virtualenv path overriding the configured default.",
        },
    },
    "send_input_to_process": {
        "description": "Send text to a tracked background process stdin.",
        "parameters": {
            "name": "Tracked background process label.",
            "input_text": "Text to send to stdin. Real newlines or escaped `\\n` are both allowed.",
        },
    },
    "get_process_output": {
        "description": "Read buffered output from a tracked background process.",
        "parameters": {
            "name": "Tracked background process label.",
            "tail_lines": "Optional number of trailing lines to return. Use `0` for the full buffer.",
        },
    },
    "kill_process": {
        "description": "Stop a tracked background process.",
        "parameters": {
            "name": "Tracked background process label.",
            "force": "Optional boolean. When true, send SIGKILL instead of SIGTERM.",
        },
    },
    "list_processes": {
        "description": "List all tracked background processes in the shared execution runtime.",
        "parameters": {},
    },
    "run_python": {
        "description": "Execute a Python snippet in a fresh subprocess with transport-safe multiline normalization.",
        "parameters": {
            "code": "Python source code to execute.",
            "cwd": "Optional working directory for the subprocess.",
            "timeout": "Optional timeout in seconds.",
            "venv_path": "Optional virtualenv path to use for the subprocess.",
            "normalize_output": "Optional boolean to normalize stdout and stderr newlines to LF.",
        },
    },
    "list_installed_packages": {
        "description": "List installed pip packages for the active or specified Python environment.",
        "parameters": {
            "venv_path": "Optional virtualenv path overriding the configured default.",
        },
    },
    "check_imports": {
        "description": "Check whether one or more Python modules can be imported in the active or specified environment.",
        "parameters": {
            "modules": "Space-separated top-level module names such as `numpy pandas torch`.",
            "venv_path": "Optional virtualenv path overriding the configured default.",
        },
    },
    "run_rust": {
        "description": "Compile or run a standalone Rust snippet with bounded output.",
        "parameters": {
            "code": "Rust source code to compile.",
            "cwd": "Optional working directory.",
            "timeout": "Optional timeout in seconds.",
            "edition": "Rust edition such as `2021`.",
            "mode": "One of `run`, `check`, or `compile`.",
            "max_output_chars": "Optional maximum returned stdout/stderr characters.",
        },
    },
    "cargo_check": {
        "description": "Run `cargo check` with package and feature controls.",
        "parameters": {
            "cwd": "Optional Cargo workspace directory.",
            "package": "Optional package name.",
            "features": "Optional comma or space separated feature list.",
            "all_features": "Optional boolean.",
            "no_default_features": "Optional boolean.",
            "extra_args": "Optional extra cargo arguments.",
            "timeout": "Optional timeout in seconds.",
            "max_output_chars": "Optional maximum returned stdout/stderr characters.",
        },
    },
    "cargo_build": {
        "description": "Run `cargo build` with package, feature, and release controls.",
        "parameters": {
            "cwd": "Optional Cargo workspace directory.",
            "package": "Optional package name.",
            "features": "Optional feature list.",
            "all_features": "Optional boolean.",
            "no_default_features": "Optional boolean.",
            "release": "Optional boolean.",
            "extra_args": "Optional extra cargo arguments.",
            "timeout": "Optional timeout in seconds.",
            "max_output_chars": "Optional maximum returned stdout/stderr characters.",
        },
    },
    "cargo_test": {
        "description": "Run `cargo test`, optionally filtered to a specific test name.",
        "parameters": {
            "test_name": "Optional test name filter.",
            "cwd": "Optional Cargo workspace directory.",
            "package": "Optional package name.",
            "features": "Optional feature list.",
            "all_features": "Optional boolean.",
            "no_default_features": "Optional boolean.",
            "extra_args": "Optional extra cargo arguments.",
            "timeout": "Optional timeout in seconds.",
            "max_output_chars": "Optional maximum returned stdout/stderr characters.",
        },
    },
    "cargo_clippy": {
        "description": "Run `cargo clippy` with bounded output.",
        "parameters": {
            "cwd": "Optional Cargo workspace directory.",
            "package": "Optional package name.",
            "features": "Optional feature list.",
            "all_features": "Optional boolean.",
            "no_default_features": "Optional boolean.",
            "extra_args": "Optional extra cargo arguments.",
            "timeout": "Optional timeout in seconds.",
            "max_output_chars": "Optional maximum returned stdout/stderr characters.",
        },
    },
    "cargo_run": {
        "description": "Run `cargo run`, keeping cargo args and program args separate.",
        "parameters": {
            "cwd": "Optional Cargo workspace directory.",
            "package": "Optional package name.",
            "bin": "Optional binary target.",
            "example": "Optional example target.",
            "features": "Optional feature list.",
            "all_features": "Optional boolean.",
            "no_default_features": "Optional boolean.",
            "release": "Optional boolean.",
            "run_args": "Optional program arguments.",
            "extra_args": "Optional extra cargo arguments.",
            "timeout": "Optional timeout in seconds.",
            "max_output_chars": "Optional maximum returned stdout/stderr characters.",
        },
    },
    "cargo_fmt": {
        "description": "Run `cargo fmt`, defaulting to `--check` to avoid rewrites.",
        "parameters": {
            "cwd": "Optional Cargo workspace directory.",
            "check": "Optional boolean. When false, format files in place.",
            "extra_args": "Optional extra cargo fmt arguments.",
            "timeout": "Optional timeout in seconds.",
            "max_output_chars": "Optional maximum returned stdout/stderr characters.",
        },
    },
    "cargo_metadata": {
        "description": "Run `cargo metadata` and parse the JSON payload when possible.",
        "parameters": {
            "cwd": "Optional Cargo workspace directory.",
            "no_deps": "Optional boolean.",
            "timeout": "Optional timeout in seconds.",
            "max_output_chars": "Optional maximum returned stdout/stderr characters.",
        },
    },
    "glob_files": {
        "description": "Find files matching a glob pattern and return structured metadata for each match.",
        "parameters": {
            "pattern": "Glob pattern such as `**/*.py` or `*.md`.",
            "path": "Base directory to search.",
            "include_hidden": "Optional boolean. When true, include dotfiles and files under hidden directories.",
            "offset": "Optional result offset for pagination.",
            "max_results": "Optional maximum returned items for one page.",
            "output_mode": "Optional output mode: `entries`, `paths`, or `count`.",
        },
    },
    "grep_files": {
        "description": "Search recursively across files and return structured text matches with context.",
        "parameters": {
            "pattern": "Literal string or regex pattern to search for.",
            "path": "Base directory to search.",
            "glob": "Optional glob to restrict which files are searched.",
            "literal": "Optional boolean. When true, treat `pattern` as literal text.",
            "case_sensitive": "Optional boolean controlling case sensitivity.",
            "context_lines": "Optional number of context lines to include around each match.",
            "max_matches": "Optional maximum total matches to return.",
            "include_hidden": "Optional boolean. When true, include dotfiles and files under hidden directories.",
            "offset": "Optional result offset for pagination.",
            "output_mode": "Optional output mode: `matches`, `files_with_matches`, or `count`.",
        },
    },
    "lsp_tool": {
        "description": "Run Python-first LSP-style navigation queries such as document symbols, workspace symbols, definitions, references, and hover.",
        "parameters": {
            "operation": "Operation name: `document_symbols`, `workspace_symbols`, `go_to_definition`, `find_references`, or `hover`.",
            "path": "Target file path for file operations, or a directory/file root for workspace operations.",
            "line": "1-based line number for cursor-based operations such as definition, references, or hover.",
            "character": "1-based character column for cursor-based operations.",
            "query": "Optional symbol name or workspace query string. Required for `workspace_symbols`.",
            "glob": "Optional glob restricting workspace Python files. Defaults to `**/*.py`.",
            "include_hidden": "Optional boolean. When true, include dotfiles and files under hidden directories.",
            "max_results": "Optional maximum number of returned results.",
        },
        "doc": (
            "Run Python-first LSP-style source navigation without requiring an external editor integration.\n\n"
            "Supported operations:\n"
            "- `document_symbols`\n"
            "- `workspace_symbols`\n"
            "- `go_to_definition`\n"
            "- `find_references`\n"
            "- `hover`\n\n"
            "Example:\n"
            '{"tool_call":{"name":"lsp_tool","arguments":{"operation":"go_to_definition","path":"src/app.py","line":42,"character":13}}}'
        ),
    },
    "get_file_outline": {
        "description": "Return a structural outline for a source file including imports, classes, and functions with line numbers.",
        "parameters": {
            "path": "Absolute or relative path to the source file.",
        },
    },
    "find_symbol": {
        "description": "Search supported source files for definitions of a symbol name and optionally include call sites.",
        "parameters": {
            "name": "Exact function, class, struct, trait, or symbol name to locate.",
            "directory": "Root directory to search from.",
            "file_glob": "Optional glob restricting searched files. Defaults to `**/*` over supported source files.",
            "include_calls": "Optional boolean. When true, include call sites in addition to definitions.",
        },
    },
    "get_project_map": {
        "description": "Summarize the important source, config, and documentation files in a directory tree.",
        "parameters": {
            "directory": "Root directory to scan.",
            "max_depth": "Optional maximum directory depth.",
            "exclude": "Optional comma or newline separated relative subpaths to skip.",
        },
    },
    "search_code": {
        "description": "Search code using multiline literal/regex matching or Python symbol discovery across a file tree.",
        "parameters": {
            "query": "Literal text, regex, or symbol fragment to search for.",
            "path": "Base directory or file to search.",
            "glob": "Optional glob to restrict which files are searched.",
            "mode": "Search mode: `literal`, `regex`, or `symbol`.",
            "case_sensitive": "Optional boolean controlling case sensitivity.",
            "context_lines": "Optional number of surrounding lines for literal/regex matches.",
            "max_results": "Optional maximum number of results to return.",
            "include_hidden": "Optional boolean. When true, include dotfiles and files under hidden directories.",
            "offset": "Optional result offset for pagination.",
        },
    },
    "think": {
        "description": "Record an internal thought as a structured payload with a readable view.",
        "parameters": {
            "thought": "Reasoning text or scratchpad note.",
        },
    },
    "todo": {
        "description": "Inspect or update the structured task checklist for the current turn/session.",
        "parameters": {
            "todos": "Legacy full-list form, as a list of todo dicts or a JSON string.",
            "command": "Structured command such as `view`, `set`, `add`, `mark`, `note`, or `clear`.",
            "items": "Todo items for `set`/`update` commands.",
            "id": "Todo item id for `mark` or `note` commands.",
            "status": "New status for `add` or `mark` commands.",
            "title": "Title for `add` commands.",
            "note": "Note text for `add` or `note` commands.",
        },
    },
    "write_file": {
        "description": (
            "Create, replace, or append a text file. Pass the complete file body in the "
            "`content` field only. Existing files must be read first so writes can be "
            "validated against a fresh snapshot."
        ),
        "parameters": {
            "path": (
                "Destination file path as a single string. Example: `src/app.py` or "
                "`/tmp/sorting.py`."
            ),
            "content": (
                "Complete file contents as one string. Put ALL code/text inside this "
                "single field. Example JSON: "
                '{"path":"/tmp/sorting.py","content":"def bubble_sort(arr):\\n    return arr\\n"}. '
                "Do NOT pass extra keys such as `array`, `snippet`, or `code`."
            ),
            "mode": (
                "Optional write mode. Use `w` to replace the whole file or `a` to append. "
                "Usually omit this and use the default `w`."
            ),
            "normalize_newlines": (
                "Optional boolean. Usually omit this and keep the default `true`."
            ),
        },
        "doc": (
            "Write or append text to a file.\n\n"
            "Required call shape:\n"
            "- `path`: destination file path string\n"
            "- `content`: one string containing the ENTIRE file text\n\n"
            "Correct example:\n"
            '{"tool_call":{"name":"write_file","arguments":{"path":"/tmp/sorting.py","content":"def bubble_sort(arr):\\n    return arr\\n"}}}\n\n'
            "Rules:\n"
            "- Put all code inside `content`.\n"
            "- Do not pass extra keys like `array`, `text`, `snippet`, or variable names.\n"
            "- `content` may be a real multiline string or a single string with escaped `\\n`.\n"
            '- Use `mode="a"` only when appending.\n'
            "- If the file already exists, read it first. Writes are rejected when the file was not read or changed since it was read.\n"
            "- For existing files that need a surgical change, prefer `edit_file` over `write_file`."
        ),
    },
    "read_edit_context": {
        "description": (
            "Find an exact needle in a text file and return only a bounded context window "
            "around the match, including line offsets."
        ),
        "parameters": {
            "path": "File path as a single string.",
            "needle": "Exact text to search for. Real multiline text or escaped `\\n` are both allowed.",
            "context_lines": "Optional number of surrounding lines to include before and after the match.",
            "max_scan_bytes": "Optional scan cap in bytes. The tool stops after this limit instead of reading the whole file.",
        },
        "doc": (
            "Read a bounded edit-context slice around an exact needle without forcing a full-file read.\n\n"
            "Use this when you need the nearest surrounding chunk for a proposed edit or to understand why an exact edit failed.\n\n"
            "Correct example:\n"
            '{"tool_call":{"name":"read_edit_context","arguments":{"path":"src/app.py","needle":"def render():\\n    return old_value\\n","context_lines":2}}}'
        ),
    },
    "edit_file": {
        "description": (
            "Replace one exact, unique block inside an existing file. Pass the old text "
            "in `old_string` and the full replacement in `new_string`. The file must be "
            "read first so the edit can be checked against a fresh snapshot."
        ),
        "parameters": {
            "path": ("Existing file path as a single string. Example: `src/app.py`."),
            "old_string": (
                "Exact text to replace as one string. Include enough surrounding context "
                "to make the match unique. Real multiline text or escaped `\\n` are both allowed."
            ),
            "new_string": (
                "Replacement text as one string. Keep all replacement code inside this single field."
            ),
            "replace_all": (
                "Optional boolean. Set true to replace every matching occurrence of `old_string` in the file."
            ),
            "normalize_newlines": (
                "Optional boolean. Usually omit this and keep the default `true`."
            ),
        },
        "doc": (
            "Replace one unique exact substring in a file.\n\n"
            "Required call shape:\n"
            "- `path`: existing file path string\n"
            "- `old_string`: exact text to find\n"
            "- `new_string`: replacement text\n\n"
            "Correct example:\n"
            '{"tool_call":{"name":"edit_file","arguments":{"path":"src/app.py","old_string":"def f():\\n    return 1\\n","new_string":"def f():\\n    return 2\\n"}}}\n\n'
            "Rules:\n"
            "- Put the full search block in `old_string`.\n"
            "- Put the full replacement block in `new_string`.\n"
            "- Do not split the edit across extra keys like `before`, `after`, or `snippet`.\n"
            "- Both fields may use real multiline text or escaped `\\n`.\n"
            "- If the file does not exist yet, `old_string` may be empty to create it through this tool.\n"
            "- Read the file first. Edits are rejected when the file was not read, was only partially read, or changed since it was read.\n"
            "- If the search text appears multiple times, either add more context or set `replace_all=true`."
        ),
    },
    "notebook_edit": {
        "description": (
            "Apply a structured cell edit to an existing Jupyter notebook (`.ipynb`) "
            "without hand-editing raw notebook JSON."
        ),
        "parameters": {
            "path": "Existing notebook path ending in `.ipynb`.",
            "action": "One of `replace`, `insert`, `append`, `delete`, or `clear_outputs`.",
            "cell_index": "0-based cell index for `replace`, `delete`, `clear_outputs`, or `insert`.",
            "source": "Cell source text for `replace`, `insert`, or `append`.",
            "cell_type": "Optional cell type: `code`, `markdown`, or `raw`.",
            "new_index": "Optional alias for insert position.",
            "strip_outputs": "Optional boolean. When true, clear notebook outputs before saving.",
        },
    },
    "mkdir": {
        "description": "Create a directory in a structured way instead of shelling out to `mkdir`.",
        "parameters": {
            "path": "Directory path to create.",
            "parents": "Optional boolean. When true, create missing parent directories.",
            "exist_ok": "Optional boolean. When true, succeed if the directory already exists.",
        },
    },
    "move_path": {
        "description": "Move or rename a file or directory with optional destination-parent creation.",
        "parameters": {
            "src": "Existing source file or directory path.",
            "dst": "Destination file or directory path.",
            "overwrite": "Optional boolean. When true, replace an existing destination file or empty directory.",
            "create_parents": "Optional boolean. When true, create missing destination parent directories.",
        },
    },
    "delete_path": {
        "description": "Delete a file or directory with explicit recursion and safety guards.",
        "parameters": {
            "path": "File or directory path to delete.",
            "recursive": "Optional boolean. Required to remove non-empty directories.",
            "missing_ok": "Optional boolean. When true, missing paths return ok instead of error.",
        },
    },
    "edit_file_libcst": {
        "description": (
            "Replace Python code by structural AST matching. Use this for Python when "
            "exact text matching is too brittle."
        ),
        "parameters": {
            "path": "Python file path as a single string.",
            "old_pattern": (
                "Valid Python statement or expression to find. Pass one code snippet string."
            ),
            "new_code": ("Valid replacement Python statement(s) or expression as one string."),
            "case_sensitive": (
                "Compatibility flag. Usually omit this; structural matching is used."
            ),
        },
        "doc": (
            "Replace Python code by AST structure instead of exact bytes.\n\n"
            "Required call shape:\n"
            "- `path`: Python file path\n"
            "- `old_pattern`: valid Python snippet to find\n"
            "- `new_code`: valid Python snippet to insert\n\n"
            "Correct example:\n"
            '{"tool_call":{"name":"edit_file_libcst","arguments":{"path":"src/foo.py","old_pattern":"return x + y","new_code":"return x * 2"}}}\n\n'
            "Rules:\n"
            "- Use this only for Python files.\n"
            "- `old_pattern` and `new_code` must both be valid Python on their own.\n"
            "- Prefer targeted symbol tools like `replace_function_body` when editing a known function."
        ),
    },
    "replace_function_body": {
        "description": "Replace the body of a Python function by function name.",
        "parameters": {
            "path": "Python file path as a single string.",
            "function_name": "Exact function name to update.",
            "new_body": ("Replacement function body statements only. Do not include `def ...:`."),
        },
    },
    "replace_docstring": {
        "description": "Replace or insert the docstring of a Python function or class.",
        "parameters": {
            "path": "Python file path as a single string.",
            "function_name": "Exact function or class name whose docstring should change.",
            "new_docstring": ("Raw docstring text only. Do not include surrounding triple quotes."),
        },
    },
    "replace_decorators": {
        "description": "Replace all decorators on a Python function by function name.",
        "parameters": {
            "path": "Python file path as a single string.",
            "function_name": "Exact function name whose decorators should change.",
            "new_decorators": (
                "List of decorator strings such as `['classmethod', 'cache(ttl=30)']`."
            ),
        },
    },
    "replace_argument": {
        "description": "Replace a keyword argument value in calls to a Python function.",
        "parameters": {
            "path": "Python file path as a single string.",
            "function_name": "Called function name to match, like `request`.",
            "arg_name": "Keyword argument name to replace.",
            "new_value": "Valid Python expression for the new value.",
        },
    },
    "insert_after_function": {
        "description": "Insert top-level Python code immediately after a top-level function.",
        "parameters": {
            "path": "Python file path as a single string.",
            "function_name": "Exact top-level function name after which to insert code.",
            "new_code": "One or more valid top-level Python statements.",
        },
    },
    "delete_function": {
        "description": "Delete Python function definitions by exact function name.",
        "parameters": {
            "path": "Python file path as a single string.",
            "function_name": "Exact function name to delete.",
        },
    },
    "find_function_by_name": {
        "description": "Find a Python function by name and return its source plus location.",
        "parameters": {
            "path": "Python file path as a single string.",
            "function_name": "Exact function name to find.",
        },
    },
    "find_class_by_name": {
        "description": "Find a Python class by name and return its source plus location.",
        "parameters": {
            "path": "Python file path as a single string.",
            "class_name": "Exact class name to find.",
        },
    },
    "tool_search": {
        "description": "Search the currently registered tool surface by name, description, or parameter names.",
        "parameters": {
            "query": "Search query such as `python`, `git diff`, or `background process`.",
            "top_k": "Optional maximum number of matches to return.",
        },
    },
}


def _merge_tool_metadata(name: str, **extra: Any) -> None:
    current = dict(_TOOL_META.get(name, {}) or {})
    current.update(extra)
    _TOOL_META[name] = current


def _clip_summary(value: Any, *, limit: int = 80) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _scope_value(args: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(args.get(key) or "").strip()
        if value:
            return value
    return ""


def _directory_validator(args: dict[str, Any], key: str = "directory") -> bool | dict[str, Any]:
    directory = str(args.get(key) or ".").strip() or "."
    path = Path(directory).expanduser()
    if not path.exists():
        return {"result": False, "message": f"Directory does not exist: {directory}"}
    if not path.is_dir():
        return {"result": False, "message": f"Path is not a directory: {directory}"}
    return True


def _path_validator(
    args: dict[str, Any],
    *,
    key: str = "path",
    require_file: bool = False,
    default_value: str | None = None,
) -> bool | dict[str, Any]:
    path_text = str(args.get(key) or default_value or "").strip()
    if not path_text:
        return {"result": False, "message": f"{key} is required"}
    path = Path(path_text).expanduser()
    if not path.exists():
        return {"result": False, "message": f"Path does not exist: {path_text}"}
    if require_file and not path.is_file():
        return {"result": False, "message": f"Path is not a file: {path_text}"}
    return True


def _summary_from_query(
    query_key: str,
    *,
    scope_keys: tuple[str, ...] = ("path", "directory"),
    prefix: str = "",
) -> Any:
    def _summary(args: dict[str, Any]) -> str:
        query = _clip_summary(args.get(query_key))
        scope = _scope_value(args, *scope_keys)
        if query and scope and scope != ".":
            body = f"{query} in {scope}"
        else:
            body = query or scope or ""
        return f"{prefix}{body}".strip()

    return _summary


def _activity_from_query(prefix: str, query_key: str | None = None) -> Any:
    def _activity(args: dict[str, Any]) -> str:
        query = _clip_summary(args.get(query_key)) if query_key else ""
        return f"{prefix} {query}".strip() if query else prefix

    return _activity


def _search_flags(*, is_read: bool = False, is_list: bool = False) -> Any:
    def _flags(_args: dict[str, Any]) -> dict[str, bool]:
        return {"isSearch": True, "isRead": is_read, "isList": is_list}

    return _flags


def _todo_validate_input(args: dict[str, Any]) -> bool | dict[str, Any]:
    allowed_commands = {"view", "set", "update", "add", "mark", "note", "clear", "validate"}
    command = str(args.get("command") or "").strip().lower()
    if command:
        if command not in allowed_commands:
            return {
                "result": False,
                "message": (
                    f"Unknown command '{command}'. "
                    "Valid commands: view, set, update, add, mark, note, clear, validate"
                ),
            }
        if command in {"set", "update", "validate"}:
            items = args.get("items")
            if items is not None and not isinstance(items, list):
                return {"result": False, "message": "items must be a list of todo dicts"}
        if command == "add" and not str(args.get("title") or "").strip():
            return {"result": False, "message": "title is required for add"}
        if command in {"mark", "note"} and args.get("id") is None:
            return {"result": False, "message": f"id is required for {command}"}
        return True

    todos = args.get("todos")
    if todos is None:
        return True
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError as exc:
            return {"result": False, "message": f"invalid todos JSON: {exc}"}
    if not isinstance(todos, list):
        return {"result": False, "message": "todos must be a list"}
    if any(not isinstance(item, dict) for item in todos):
        return {"result": False, "message": "todos must contain only todo dicts"}
    return True


def _todo_tool_summary(args: dict[str, Any]) -> str:
    command = str(args.get("command") or "").strip().lower()
    if command in {"set", "update", "validate"}:
        count = len(args.get("items") or [])
        return f"{command} {count} task{'s' if count != 1 else ''}"
    if command == "add":
        title = _clip_summary(args.get("title"))
        return f"add {title}".strip()
    if command in {"mark", "note"}:
        item_id = args.get("id")
        return f"{command} task {item_id}".strip()
    if command:
        return command
    todos = args.get("todos")
    if isinstance(todos, list):
        count = len(todos)
        return f"{count} task{'s' if count != 1 else ''}"
    return "task list"


def _todo_activity(args: dict[str, Any]) -> str:
    command = str(args.get("command") or "").strip().lower()
    if command in {"view", "validate"}:
        return "Reviewing task list"
    if command in {"clear", "set", "update", "add", "mark", "note"}:
        return "Updating task list"
    return "Managing task list"


def _think_summary(args: dict[str, Any]) -> str:
    return _clip_summary(args.get("thought"))


def _required_text(args: dict[str, Any], key: str, *, message: str | None = None) -> bool | dict[str, Any]:
    value = str(args.get(key) or "").strip()
    if value:
        return True
    return {"result": False, "message": message or f"{key} is required"}


def _optional_directory(args: dict[str, Any], key: str = "cwd") -> bool | dict[str, Any]:
    value = str(args.get(key) or "").strip()
    if not value:
        return True
    return _directory_validator(args, key=key)


def _optional_file(args: dict[str, Any], key: str = "path") -> bool | dict[str, Any]:
    value = str(args.get(key) or "").strip()
    if not value:
        return True
    return _path_validator(args, key=key, require_file=True)


def _optional_venv(args: dict[str, Any], key: str = "venv_path") -> bool | dict[str, Any]:
    value = str(args.get(key) or "").strip()
    if not value:
        return True
    return _directory_validator(args, key=key)


def _positive_int_validator(
    args: dict[str, Any],
    key: str,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> bool | dict[str, Any]:
    value = args.get(key)
    if value in (None, ""):
        return True
    try:
        number = int(value)
    except Exception:
        return {"result": False, "message": f"{key} must be an integer"}
    if number < minimum:
        return {"result": False, "message": f"{key} must be >= {minimum}"}
    if maximum is not None and number > maximum:
        return {"result": False, "message": f"{key} must be <= {maximum}"}
    return True


def _non_negative_int_validator(args: dict[str, Any], key: str) -> bool | dict[str, Any]:
    return _positive_int_validator(args, key, minimum=0)


def _list_validator(args: dict[str, Any], key: str, *, required: bool = False) -> bool | dict[str, Any]:
    value = args.get(key)
    if value is None:
        return {"result": False, "message": f"{key} is required"} if required else True
    if not isinstance(value, list):
        return {"result": False, "message": f"{key} must be a list"}
    return True


def _http_url_validator(args: dict[str, Any], key: str = "url") -> bool | dict[str, Any]:
    from .WebTool.tool import _validate_http_url

    url = str(args.get(key) or "").strip()
    if not url:
        return {"result": False, "message": f"{key} is required"}
    error = _validate_http_url(url)
    if error:
        return {"result": False, "message": error}
    return True


def _combine_validators(*validators: Any) -> Any:
    def _validator(args: dict[str, Any]) -> bool | dict[str, Any]:
        for validator in validators:
            result = validator(args)
            if result is True or result is None:
                continue
            return result
        return True

    return _validator


def _summary_from_path(
    key: str = "path",
    *,
    prefix: str = "",
    fallback: str = "",
) -> Any:
    def _summary(args: dict[str, Any]) -> str:
        value = _clip_summary(args.get(key))
        body = value or fallback
        return f"{prefix}{body}".strip()

    return _summary


def _summary_from_name_and_path(name_key: str, path_key: str = "path", *, prefix: str = "") -> Any:
    def _summary(args: dict[str, Any]) -> str:
        name = _clip_summary(args.get(name_key))
        path = _clip_summary(args.get(path_key))
        if name and path:
            return f"{prefix}{name} in {path}".strip()
        return f"{prefix}{name or path}".strip()

    return _summary


def _activity(prefix: str) -> Any:
    return lambda _args: prefix


def _bash_summary(args: dict[str, Any]) -> str:
    return _clip_summary(args.get("command"))


def _bash_activity(_args: dict[str, Any]) -> str:
    return "Running shell command"


def _bash_flags(args: dict[str, Any]) -> dict[str, bool]:
    validation = _validate_bash_command(str(args.get("command") or ""), require_read_only=False)
    return {
        "isSearch": bool(validation.get("is_read_only")),
        "isRead": bool(validation.get("is_read_only")),
        "isList": False,
    }


def _bash_validate_input(args: dict[str, Any]) -> bool | dict[str, Any]:
    command = str(args.get("command") or "").strip()
    if not command:
        return {"result": False, "message": "command is required"}
    return _positive_int_validator(args, "timeout", minimum=1, maximum=3600)


_RUNTIME_META.update(
    {
        "read_file": {"read_only": True, "cacheable": True, "content_reader": True, "concurrency_safe": True},
        "read_edit_context": {"read_only": True, "cacheable": True, "content_reader": True, "concurrency_safe": True},
        "search_file": {"read_only": True, "cacheable": True, "content_reader": True, "concurrency_safe": True},
        "list_dir": {"read_only": True, "cacheable": True, "concurrency_safe": True},
        "get_git_status": {"read_only": True, "cacheable": False, "concurrency_safe": True},
        "get_git_diff": {"read_only": True, "cacheable": False, "content_reader": True, "concurrency_safe": True},
        "get_symbol_info": {"read_only": True, "cacheable": True, "content_reader": True, "concurrency_safe": True},
        "read_line": {"read_only": True, "cacheable": True, "content_reader": True, "concurrency_safe": True},
        "find_imports": {"read_only": True, "cacheable": True, "content_reader": True, "concurrency_safe": True},
        "get_file_outline": {"read_only": True, "cacheable": True, "content_reader": True, "concurrency_safe": True},
        "find_symbol": {"read_only": True, "cacheable": True, "content_reader": True, "concurrency_safe": True},
        "get_project_map": {"read_only": True, "cacheable": True, "concurrency_safe": True},
        "glob_files": {"read_only": True, "cacheable": True, "concurrency_safe": True},
        "grep_files": {"read_only": True, "cacheable": True, "content_reader": True, "concurrency_safe": True},
        "search_code": {"read_only": True, "cacheable": True, "content_reader": True, "concurrency_safe": True},
        "rg_search": {"read_only": True, "cacheable": True, "content_reader": True, "concurrency_safe": True},
        "fd_find": {"read_only": True, "cacheable": True, "concurrency_safe": True},
        "find_references": {"read_only": True, "cacheable": True, "content_reader": True, "concurrency_safe": True},
        "search_symbols": {"read_only": True, "cacheable": True, "content_reader": True, "concurrency_safe": True},
        "check_imports": {"read_only": True, "cacheable": False, "concurrency_safe": True},
        "list_installed_packages": {"read_only": True, "cacheable": False, "concurrency_safe": True},
        "cargo_metadata": {"read_only": True, "cacheable": False, "concurrency_safe": True},
        "todo": {"read_only": False, "cacheable": False},
    }
)

_merge_tool_metadata(
    "search_file",
    description="Search a single file for literal text or regex matches and return structured line context.",
    validate_input=lambda args: _path_validator(args, require_file=True),
    is_search_or_read_command=_search_flags(is_read=True),
    get_tool_use_summary=_summary_from_query("pattern", scope_keys=("path",)),
    get_activity_description=_activity_from_query("Searching file for", "pattern"),
    user_facing_name=lambda _args: "Search",
)
_merge_tool_metadata(
    "glob_files",
    validate_input=lambda args: _directory_validator(args, key="path"),
    is_search_or_read_command=_search_flags(is_list=True),
    get_tool_use_summary=_summary_from_query("pattern", scope_keys=("path",)),
    get_activity_description=_activity_from_query("Finding files matching", "pattern"),
    user_facing_name=lambda _args: "Find Files",
)
_merge_tool_metadata(
    "grep_files",
    validate_input=lambda args: _directory_validator(args, key="path"),
    is_search_or_read_command=_search_flags(),
    get_tool_use_summary=_summary_from_query("pattern", scope_keys=("path",)),
    get_activity_description=_activity_from_query("Searching for", "pattern"),
    user_facing_name=lambda _args: "Search",
)
_merge_tool_metadata(
    "search_code",
    validate_input=lambda args: _path_validator(args, key="path", default_value="."),
    is_search_or_read_command=_search_flags(),
    get_tool_use_summary=_summary_from_query("query", scope_keys=("path",)),
    get_activity_description=_activity_from_query("Searching code for", "query"),
    user_facing_name=lambda _args: "Search",
)
_merge_tool_metadata(
    "rg_search",
    description="Search a directory tree with ripgrep semantics and ranked structured matches.",
    validate_input=lambda args: _directory_validator(args, key="directory"),
    is_search_or_read_command=_search_flags(),
    get_tool_use_summary=_summary_from_query("pattern", scope_keys=("directory",)),
    get_activity_description=_activity_from_query("Searching for", "pattern"),
    user_facing_name=lambda _args: "Search",
)
_merge_tool_metadata(
    "fd_find",
    description="Find files or directories by name pattern and return the newest matches first.",
    validate_input=lambda args: _directory_validator(args, key="directory"),
    is_search_or_read_command=_search_flags(is_list=True),
    get_tool_use_summary=_summary_from_query("pattern", scope_keys=("directory",)),
    get_activity_description=_activity_from_query("Finding files matching", "pattern"),
    user_facing_name=lambda _args: "Find Files",
)
_merge_tool_metadata(
    "find_references",
    description="Find likely symbol references with a word-boundary text search and ranked results.",
    validate_input=lambda args: _directory_validator(args, key="directory"),
    is_search_or_read_command=_search_flags(),
    get_tool_use_summary=_summary_from_query("name", scope_keys=("directory",), prefix="references for "),
    get_activity_description=_activity_from_query("Finding references for", "name"),
    user_facing_name=lambda _args: "Search",
)
_merge_tool_metadata(
    "search_symbols",
    description="Search source trees for symbol definitions across supported languages using parsed syntax trees.",
    validate_input=lambda args: _directory_validator(args, key="directory"),
    is_search_or_read_command=_search_flags(),
    get_tool_use_summary=_summary_from_query("name", scope_keys=("directory",), prefix="symbols for "),
    get_activity_description=_activity_from_query("Searching symbols for", "name"),
    user_facing_name=lambda _args: "Search",
)
_merge_tool_metadata(
    "think",
    get_tool_use_summary=_think_summary,
    get_activity_description=lambda _args: "Thinking",
    user_facing_name=lambda _args: "",
)
_merge_tool_metadata(
    "todo",
    validate_input=_todo_validate_input,
    get_tool_use_summary=_todo_tool_summary,
    get_activity_description=_todo_activity,
    user_facing_name=lambda _args: "Tasks",
)
_merge_tool_metadata(
    "read_file",
    validate_input=lambda args: _path_validator(args, require_file=True),
    is_search_or_read_command=lambda _args: {"isSearch": False, "isRead": True, "isList": False},
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Reading file"),
    user_facing_name=lambda _args: "Read",
)
_merge_tool_metadata(
    "read_edit_context",
    validate_input=_combine_validators(
        lambda args: _path_validator(args, require_file=True),
        lambda args: _required_text(args, "needle"),
    ),
    is_search_or_read_command=lambda _args: {"isSearch": True, "isRead": True, "isList": False},
    get_tool_use_summary=_summary_from_query("needle", scope_keys=("path",)),
    get_activity_description=_activity("Reading edit context"),
    user_facing_name=lambda _args: "Read",
)
_merge_tool_metadata(
    "list_dir",
    validate_input=lambda args: _directory_validator(args, key="path"),
    is_search_or_read_command=lambda _args: {"isSearch": False, "isRead": True, "isList": True},
    get_tool_use_summary=_summary_from_path(fallback="."),
    get_activity_description=_activity("Listing directory"),
    user_facing_name=lambda _args: "List Files",
)
_merge_tool_metadata(
    "write_file",
    validate_input=_combine_validators(
        lambda args: _required_text(args, "path"),
        lambda args: _required_text(args, "content"),
    ),
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Writing file"),
    user_facing_name=lambda _args: "Write",
)
_merge_tool_metadata(
    "edit_file",
    validate_input=lambda args: _required_text(args, "path"),
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Editing file"),
    user_facing_name=lambda _args: "Edit",
)
_merge_tool_metadata(
    "apply_edit_block",
    validate_input=_combine_validators(
        lambda args: _required_text(args, "path"),
        lambda args: _required_text(args, "blocks"),
    ),
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Applying edit blocks"),
    user_facing_name=lambda _args: "Edit",
)
_merge_tool_metadata(
    "preview_edit",
    validate_input=_combine_validators(
        lambda args: _required_text(args, "path"),
        lambda args: _required_text(args, "blocks"),
    ),
    is_search_or_read_command=lambda _args: {"isSearch": False, "isRead": True, "isList": False},
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Previewing edit"),
    user_facing_name=lambda _args: "Edit",
)
_merge_tool_metadata(
    "smart_edit",
    validate_input=_combine_validators(
        lambda args: _required_text(args, "path"),
        lambda args: _list_validator(args, "edits", required=True),
    ),
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Applying smart edits"),
    user_facing_name=lambda _args: "Edit",
)
_merge_tool_metadata(
    "notebook_edit",
    validate_input=_combine_validators(
        lambda args: _path_validator(args, require_file=True),
        lambda args: _required_text(args, "action"),
    ),
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Editing notebook"),
    user_facing_name=lambda _args: "Edit",
)
_merge_tool_metadata(
    "mkdir",
    validate_input=lambda args: _required_text(args, "path"),
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Creating directory"),
    user_facing_name=lambda _args: "Files",
)
_merge_tool_metadata(
    "move_path",
    validate_input=_combine_validators(
        lambda args: _required_text(args, "src"),
        lambda args: _required_text(args, "dst"),
    ),
    get_tool_use_summary=lambda args: f"{_clip_summary(args.get('src'))} -> {_clip_summary(args.get('dst'))}",
    get_activity_description=_activity("Moving path"),
    user_facing_name=lambda _args: "Files",
)
_merge_tool_metadata(
    "delete_path",
    validate_input=lambda args: _required_text(args, "path"),
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Deleting path"),
    user_facing_name=lambda _args: "Files",
)
_merge_tool_metadata(
    "get_git_status",
    validate_input=lambda args: _path_validator(args, key="path", default_value="."),
    is_search_or_read_command=lambda _args: {"isSearch": False, "isRead": True, "isList": True},
    get_tool_use_summary=_summary_from_path(fallback="."),
    get_activity_description=_activity("Inspecting git status"),
    user_facing_name=lambda _args: "Git",
)
_merge_tool_metadata(
    "get_git_diff",
    validate_input=lambda args: _path_validator(args, key="path", default_value="."),
    is_search_or_read_command=lambda _args: {"isSearch": False, "isRead": True, "isList": False},
    get_tool_use_summary=_summary_from_path(fallback="."),
    get_activity_description=_activity("Reading git diff"),
    user_facing_name=lambda _args: "Git",
)
_merge_tool_metadata(
    "get_symbol_info",
    validate_input=_combine_validators(
        lambda args: _path_validator(args, require_file=True),
        lambda args: _required_text(args, "symbol"),
    ),
    is_search_or_read_command=lambda _args: {"isSearch": True, "isRead": True, "isList": False},
    get_tool_use_summary=_summary_from_name_and_path("symbol"),
    get_activity_description=_activity("Reading symbol info"),
    user_facing_name=lambda _args: "Inspect",
)
_merge_tool_metadata(
    "read_line",
    validate_input=_combine_validators(
        lambda args: _path_validator(args, require_file=True),
        lambda args: _positive_int_validator(args, "line_number", minimum=1),
    ),
    is_search_or_read_command=lambda _args: {"isSearch": False, "isRead": True, "isList": False},
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Reading line"),
    user_facing_name=lambda _args: "Read",
)
_merge_tool_metadata(
    "find_imports",
    validate_input=lambda args: _path_validator(args, require_file=True),
    is_search_or_read_command=lambda _args: {"isSearch": True, "isRead": True, "isList": True},
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Finding imports"),
    user_facing_name=lambda _args: "Inspect",
)
_merge_tool_metadata(
    "get_file_outline",
    validate_input=lambda args: _path_validator(args, require_file=True),
    is_search_or_read_command=lambda _args: {"isSearch": True, "isRead": True, "isList": True},
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Building file outline"),
    user_facing_name=lambda _args: "Inspect",
)
_merge_tool_metadata(
    "find_symbol",
    validate_input=_combine_validators(
        lambda args: _directory_validator(args, key="directory"),
        lambda args: _required_text(args, "name"),
    ),
    is_search_or_read_command=_search_flags(),
    get_tool_use_summary=_summary_from_query("name", scope_keys=("directory",)),
    get_activity_description=_activity_from_query("Finding symbol", "name"),
    user_facing_name=lambda _args: "Search",
)
_merge_tool_metadata(
    "get_project_map",
    validate_input=_combine_validators(
        lambda args: _directory_validator(args, key="directory"),
        lambda args: _non_negative_int_validator(args, "max_depth"),
    ),
    is_search_or_read_command=lambda _args: {"isSearch": True, "isRead": True, "isList": True},
    get_tool_use_summary=_summary_from_path("directory", fallback="."),
    get_activity_description=_activity("Mapping project"),
    user_facing_name=lambda _args: "Inspect",
)
_merge_tool_metadata(
    "fetch_url",
    validate_input=_combine_validators(
        lambda args: _http_url_validator(args, key="url"),
        lambda args: _positive_int_validator(args, "timeout", minimum=1, maximum=300),
        lambda args: _positive_int_validator(args, "max_chars", minimum=1, maximum=200000),
    ),
    is_search_or_read_command=lambda _args: {"isSearch": False, "isRead": True, "isList": False},
    get_tool_use_summary=_summary_from_path("url"),
    get_activity_description=_activity("Fetching URL"),
    user_facing_name=lambda _args: "Web",
)
_merge_tool_metadata(
    "web_search",
    validate_input=lambda args: _required_text(args, "query"),
    is_search_or_read_command=_search_flags(),
    get_tool_use_summary=_summary_from_query("query", scope_keys=()),
    get_activity_description=_activity_from_query("Searching web for", "query"),
    user_facing_name=lambda _args: "Web",
)
_merge_tool_metadata(
    "pypi_info",
    validate_input=lambda args: _required_text(args, "package_name"),
    is_search_or_read_command=lambda _args: {"isSearch": True, "isRead": True, "isList": False},
    get_tool_use_summary=lambda args: _clip_summary(args.get("package_name")),
    get_activity_description=_activity("Fetching PyPI metadata"),
    user_facing_name=lambda _args: "Web",
)
_merge_tool_metadata(
    "github_read_file",
    validate_input=_combine_validators(
        lambda args: _required_text(args, "owner"),
        lambda args: _required_text(args, "repo"),
        lambda args: _required_text(args, "path"),
    ),
    is_search_or_read_command=lambda _args: {"isSearch": False, "isRead": True, "isList": False},
    get_tool_use_summary=lambda args: _summary_from_name_and_path("repo", "path", prefix=f"{str(args.get('owner') or '').strip()}/")(args),
    get_activity_description=_activity("Reading GitHub file"),
    user_facing_name=lambda _args: "Web",
)
_merge_tool_metadata(
    "set_venv",
    validate_input=lambda args: _directory_validator(args, key="venv_path"),
    get_tool_use_summary=_summary_from_path("venv_path"),
    get_activity_description=_activity("Setting virtualenv"),
    user_facing_name=lambda _args: "Python",
)
_merge_tool_metadata(
    "set_working_directory",
    validate_input=lambda args: _directory_validator(args, key="path"),
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Setting working directory"),
    user_facing_name=lambda _args: "Terminal",
)
_merge_tool_metadata(
    "install_packages",
    validate_input=_combine_validators(
        lambda args: _required_text(args, "packages"),
        lambda args: _optional_venv(args),
    ),
    get_tool_use_summary=lambda args: _clip_summary(args.get("packages")),
    get_activity_description=_activity("Installing packages"),
    user_facing_name=lambda _args: "Python",
)
_merge_tool_metadata(
    "show_coding_config",
    is_search_or_read_command=lambda _args: {"isSearch": False, "isRead": True, "isList": False},
    get_tool_use_summary=lambda _args: "execution config",
    get_activity_description=_activity("Reading execution config"),
    user_facing_name=lambda _args: "Terminal",
)
_merge_tool_metadata(
    "start_background_process",
    validate_input=_combine_validators(
        lambda args: _required_text(args, "command"),
        lambda args: _required_text(args, "name"),
        lambda args: _optional_directory(args, "cwd"),
        lambda args: _optional_venv(args),
    ),
    get_tool_use_summary=lambda args: _clip_summary(args.get("name") or args.get("command")),
    get_activity_description=_activity("Starting background process"),
    user_facing_name=lambda _args: "Terminal",
)
_merge_tool_metadata(
    "send_input_to_process",
    validate_input=lambda args: _required_text(args, "name"),
    get_tool_use_summary=lambda args: _clip_summary(args.get("name")),
    get_activity_description=_activity("Sending process input"),
    user_facing_name=lambda _args: "Terminal",
)
_merge_tool_metadata(
    "get_process_output",
    validate_input=_combine_validators(
        lambda args: _required_text(args, "name"),
        lambda args: _non_negative_int_validator(args, "tail_lines"),
    ),
    is_search_or_read_command=lambda _args: {"isSearch": False, "isRead": True, "isList": False},
    get_tool_use_summary=lambda args: _clip_summary(args.get("name")),
    get_activity_description=_activity("Reading process output"),
    user_facing_name=lambda _args: "Terminal",
)
_merge_tool_metadata(
    "kill_process",
    validate_input=lambda args: _required_text(args, "name"),
    get_tool_use_summary=lambda args: _clip_summary(args.get("name")),
    get_activity_description=_activity("Stopping process"),
    user_facing_name=lambda _args: "Terminal",
)
_merge_tool_metadata(
    "list_processes",
    is_search_or_read_command=lambda _args: {"isSearch": False, "isRead": True, "isList": True},
    get_tool_use_summary=lambda _args: "background processes",
    get_activity_description=_activity("Listing processes"),
    user_facing_name=lambda _args: "Terminal",
)
_merge_tool_metadata(
    "run_python",
    validate_input=_combine_validators(
        lambda args: _required_text(args, "code"),
        lambda args: _positive_int_validator(args, "timeout", minimum=1, maximum=3600),
        lambda args: _optional_directory(args, "cwd"),
        lambda args: _optional_venv(args),
    ),
    get_tool_use_summary=lambda args: _clip_summary(str(args.get("code") or "").splitlines()[0] if str(args.get("code") or "").splitlines() else ""),
    get_activity_description=_activity("Running Python"),
    user_facing_name=lambda _args: "Python",
)
_merge_tool_metadata(
    "list_installed_packages",
    validate_input=lambda args: _optional_venv(args),
    is_search_or_read_command=lambda _args: {"isSearch": False, "isRead": True, "isList": True},
    get_tool_use_summary=lambda _args: "installed packages",
    get_activity_description=_activity("Listing installed packages"),
    user_facing_name=lambda _args: "Python",
)
_merge_tool_metadata(
    "check_imports",
    validate_input=_combine_validators(
        lambda args: _required_text(args, "modules"),
        lambda args: _optional_venv(args),
    ),
    is_search_or_read_command=lambda _args: {"isSearch": False, "isRead": True, "isList": False},
    get_tool_use_summary=lambda args: _clip_summary(args.get("modules")),
    get_activity_description=_activity("Checking imports"),
    user_facing_name=lambda _args: "Python",
)
_merge_tool_metadata(
    "lsp_tool",
    validate_input=_combine_validators(
        lambda args: _path_validator(args, key="path"),
        lambda args: _positive_int_validator(args, "max_results", minimum=1, maximum=500),
    ),
    is_search_or_read_command=lambda _args: {"isSearch": True, "isRead": True, "isList": False},
    get_tool_use_summary=lambda args: f"{str(args.get('operation') or '').strip()} {_clip_summary(args.get('path'))}".strip(),
    get_activity_description=_activity("Running source navigation"),
    user_facing_name=lambda _args: "Inspect",
)
_merge_tool_metadata(
    "bash",
    validate_input=_bash_validate_input,
    is_search_or_read_command=_bash_flags,
    get_tool_use_summary=_bash_summary,
    get_activity_description=_bash_activity,
    user_facing_name=lambda _args: "Terminal",
)
_merge_tool_metadata(
    "tool_search",
    validate_input=_combine_validators(
        lambda args: _required_text(args, "query"),
        lambda args: _positive_int_validator(args, "top_k", minimum=1, maximum=20),
    ),
    is_search_or_read_command=lambda _args: {"isSearch": True, "isRead": True, "isList": True},
    get_tool_use_summary=_summary_from_query("query", scope_keys=()),
    get_activity_description=_activity_from_query("Searching tools for", "query"),
    user_facing_name=lambda _args: "Inspect",
)
_merge_tool_metadata(
    "edit_file_libcst",
    validate_input=_combine_validators(
        lambda args: _path_validator(args, require_file=True),
        lambda args: _required_text(args, "old_pattern"),
        lambda args: _required_text(args, "new_code"),
    ),
    get_tool_use_summary=_summary_from_path(),
    get_activity_description=_activity("Editing Python structurally"),
    user_facing_name=lambda _args: "Edit",
)
for _tool_name, _field_name in (
    ("replace_function_body", "function_name"),
    ("replace_docstring", "function_name"),
    ("replace_decorators", "function_name"),
    ("replace_argument", "function_name"),
    ("insert_after_function", "function_name"),
    ("delete_function", "function_name"),
    ("find_function_by_name", "function_name"),
    ("find_class_by_name", "class_name"),
):
        _merge_tool_metadata(
            _tool_name,
            validate_input=_combine_validators(
                lambda args: _path_validator(args, require_file=True),
                lambda args, field_name=_field_name: _required_text(args, field_name),
        ),
        is_search_or_read_command=(
            (lambda _args: {"isSearch": True, "isRead": True, "isList": False})
            if _tool_name in {"find_function_by_name", "find_class_by_name"}
            else None
            ),
            get_tool_use_summary=_summary_from_name_and_path(_field_name),
            get_activity_description=_activity("Updating Python symbol" if "find_" not in _tool_name else "Reading Python symbol"),
            user_facing_name=lambda _args, tool_name=_tool_name: "Edit" if tool_name not in {"find_function_by_name", "find_class_by_name"} else "Inspect",
        )
for _tool_name in (
    "run_rust",
    "cargo_check",
    "cargo_build",
    "cargo_test",
    "cargo_clippy",
    "cargo_run",
    "cargo_fmt",
    "cargo_metadata",
):
    _merge_tool_metadata(
        _tool_name,
        validate_input=_combine_validators(
            lambda args: _optional_directory(args, "cwd"),
            lambda args: _positive_int_validator(args, "timeout", minimum=1, maximum=3600),
        ),
        get_tool_use_summary=lambda args, tool_name=_tool_name: _clip_summary(args.get("package") or args.get("test_name") or args.get("bin") or args.get("example") or args.get("cwd") or tool_name),
        get_activity_description=_activity("Running Rust tool"),
        user_facing_name=lambda _args: "Rust",
    )

_BASE_CORE_TOOL_ITEMS: tuple[tuple[str, Any], ...] = (
    ("read_file", read_file),
    ("read_edit_context", read_edit_context),
    ("write_file", write_file),
    ("edit_file", edit_file),
    ("search_file", search_file),
    ("list_dir", list_dir),
    ("apply_edit_block", apply_edit_block),
    ("preview_edit", preview_edit),
    ("smart_edit", smart_edit),
    ("notebook_edit", notebook_edit),
    ("mkdir", mkdir),
    ("move_path", move_path),
    ("delete_path", delete_path),
    ("get_git_status", get_git_status),
    ("get_git_diff", get_git_diff),
    ("get_symbol_info", get_symbol_info),
    ("read_line", read_line),
    ("find_imports", find_imports),
    ("get_file_outline", get_file_outline),
    ("find_symbol", find_symbol),
    ("get_project_map", get_project_map),
    ("fetch_url", fetch_url),
    ("web_search", web_search),
    ("pypi_info", pypi_info),
    ("github_read_file", github_read_file),
    ("set_venv", set_venv),
    ("set_working_directory", set_working_directory),
    ("install_packages", install_packages),
    ("show_coding_config", show_coding_config),
    ("start_background_process", start_background_process),
    ("send_input_to_process", send_input_to_process),
    ("get_process_output", get_process_output),
    ("kill_process", kill_process),
    ("list_processes", list_processes),
    ("run_python", run_python),
    ("list_installed_packages", list_installed_packages),
    ("check_imports", check_imports),
    ("run_rust", run_rust),
    ("cargo_check", cargo_check),
    ("cargo_build", cargo_build),
    ("cargo_test", cargo_test),
    ("cargo_clippy", cargo_clippy),
    ("cargo_run", cargo_run),
    ("cargo_fmt", cargo_fmt),
    ("cargo_metadata", cargo_metadata),
    ("lsp_tool", lsp_tool),
    ("bash", bash),
    ("glob_files", glob_files),
    ("grep_files", grep_files),
    ("search_code", search_code),
    ("rg_search", rg_search),
    ("fd_find", fd_find),
    ("find_references", find_references),
    ("search_symbols", search_symbols),
    ("tool_search", tool_search),
    ("think", think),
    ("todo", todo),
)

_LIBCST_TOOL_ITEMS: tuple[tuple[str, Any], ...] = (
    ("edit_file_libcst", edit_file_libcst),
    ("replace_function_body", replace_function_body),
    ("replace_docstring", replace_docstring),
    ("replace_decorators", replace_decorators),
    ("replace_argument", replace_argument),
    ("insert_after_function", insert_after_function),
    ("delete_function", delete_function),
    ("find_function_by_name", find_function_by_name),
    ("find_class_by_name", find_class_by_name),
)

_OPTIONAL_CORE_TOOL_ITEMS: tuple[tuple[str, Any], ...] = (
    _LIBCST_TOOL_ITEMS if _check_libcst() else ()
)

CORE_TOOL_ITEMS: tuple[tuple[str, Any], ...] = _BASE_CORE_TOOL_ITEMS + _OPTIONAL_CORE_TOOL_ITEMS
CORE_TOOL_FUNCTIONS: tuple[Any, ...] = tuple(fn for _, fn in CORE_TOOL_ITEMS)
CORE_TOOL_NAMES: tuple[str, ...] = tuple(name for name, _ in CORE_TOOL_ITEMS)

for name, _tool_fn in CORE_TOOL_ITEMS:
    tool_spec = dict(_TOOL_META.get(name, {}) or {})
    runtime = _RUNTIME_META.get(name)
    if runtime is not None:
        tool_spec["runtime"] = runtime
    if tool_spec:
        build_tool(_tool_fn, **tool_spec)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    *CORE_TOOL_NAMES,
    "CORE_TOOL_FUNCTIONS",
    "CORE_TOOL_NAMES",
    "ContentReplacementState",
    "compact_result",
]

# Optional: expose availability flag for users who want to check
__all__.append("_LIBCST_AVAILABLE")
