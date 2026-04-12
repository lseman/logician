"""Always-on core tools for the SOTA agent."""

from typing import Any

from ..runtime import build_tool
from .BashTool import bash
from .FileEditTool import (
    apply_edit_block,
    edit_file,
    preview_edit,
    smart_edit,
    write_file,
)
from .FileReadTool import list_dir, read_edit_context, read_file
from .GitTool import (
    get_git_diff,
    get_git_status,
)
from .LspTool import lsp_tool
from .ProcessTool import (
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
from .ProjectTool import find_symbol, get_file_outline, get_project_map
from .PythonTool import check_imports, list_installed_packages, run_python
from .SearchTool import (
    find_imports,
    get_symbol_info,
    glob_files,
    grep_files,
    read_line,
    search_code,
    search_file,
)
from .TaskTool import think, todo
from .WebTool import fetch_url, github_read_file, pypi_info, web_search

# ---------------------------------------------------------------------
# Optional LibCST support
# ---------------------------------------------------------------------
_LIBCST_AVAILABLE = False
_LIBCST_TOOLS: list[str] = []

try:
    import libcst  # noqa: F401  # just to check availability

    from .FileEditTool.libcst import (
        delete_function,
        edit_file_libcst,
        find_class_by_name,
        find_function_by_name,
        insert_after_function,
        replace_argument,
        replace_decorators,
        replace_docstring,
        replace_function_body,
    )

    _LIBCST_AVAILABLE = True
    _LIBCST_TOOLS = [
        "edit_file_libcst",
        "replace_function_body",
        "replace_docstring",
        "replace_decorators",
        "replace_argument",
        "insert_after_function",
        "delete_function",
        "find_function_by_name",
        "find_class_by_name",
    ]
except ImportError:
    # LibCST is optional — define no-op placeholders so the runtime meta loop
    # doesn't fail and __all__ stays consistent.
    def _libcst_not_available(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "error",
            "error": "libcst is not installed. Install it with: pip install libcst",
        }

    edit_file_libcst = _libcst_not_available
    replace_function_body = _libcst_not_available
    replace_docstring = _libcst_not_available
    replace_decorators = _libcst_not_available
    replace_argument = _libcst_not_available
    insert_after_function = _libcst_not_available
    delete_function = _libcst_not_available
    find_function_by_name = _libcst_not_available
    find_class_by_name = _libcst_not_available


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
    "edit_file_libcst": {"writes_files": True, "cacheable": False},
    "replace_function_body": {"writes_files": True, "cacheable": False},
    "replace_docstring": {"writes_files": True, "cacheable": False},
    "replace_decorators": {"writes_files": True, "cacheable": False},
    "replace_argument": {"writes_files": True, "cacheable": False},
    "insert_after_function": {"writes_files": True, "cacheable": False},
    "delete_function": {"writes_files": True, "cacheable": False},
    "find_function_by_name": {"read_only": True, "cacheable": True, "content_reader": True},
    "find_class_by_name": {"read_only": True, "cacheable": True, "content_reader": True},
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
}

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
    ("lsp_tool", lsp_tool),
    ("bash", bash),
    ("glob_files", glob_files),
    ("grep_files", grep_files),
    ("search_code", search_code),
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

CORE_TOOL_ITEMS: tuple[tuple[str, Any], ...] = _BASE_CORE_TOOL_ITEMS + _LIBCST_TOOL_ITEMS
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
]

# Optional: expose availability flag for users who want to check
__all__.append("_LIBCST_AVAILABLE")
