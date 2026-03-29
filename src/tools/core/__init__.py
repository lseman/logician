"""Always-on core tools for the SOTA agent."""

# import Any
from typing import Any

from .editing import (
    apply_edit_block,
    edit_file,
    preview_edit,
    smart_edit,
    write_file,
)
from .files import list_dir, read_file
from .git_tools import (
    get_git_diff,
    get_git_status,
)
from .inspection import (
    find_imports,
    get_symbol_info,
    read_line,
    search_file,
)
from .search import glob_files, grep_files
from .shell import bash
from .tasks import think, todo

# ---------------------------------------------------------------------
# Optional LibCST support
# ---------------------------------------------------------------------
_LIBCST_AVAILABLE = False
_LIBCST_TOOLS: list[str] = []

try:
    import libcst  # noqa: F401  # just to check availability

    from .files_libcst import (
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


# ---------------------------------------------------------------------
# Runtime metadata decoration
# ---------------------------------------------------------------------
def _apply_tool_meta(fn: Any, **tool_meta: Any) -> Any:
    meta = dict(getattr(fn, "__llm_tool_meta__", {}) or {})
    runtime_meta = dict(tool_meta.pop("runtime", {}) or {})
    if runtime_meta:
        runtime = dict(meta.get("runtime") or {})
        runtime.update(runtime_meta)
        meta["runtime"] = runtime
    for key, value in tool_meta.items():
        meta[key] = value
    setattr(fn, "__llm_tool_meta__", meta)
    return fn


_RUNTIME_META = {
    "read_file": {"read_only": True, "cacheable": True},
    "search_file": {"read_only": True, "cacheable": True},
    "list_dir": {"read_only": True, "cacheable": True},
    "preview_edit": {"read_only": True, "cacheable": False},
    "get_git_status": {"read_only": True, "cacheable": False},
    "get_git_diff": {"read_only": True, "cacheable": False},
    "get_symbol_info": {"read_only": True, "cacheable": True},
    "read_line": {"read_only": True, "cacheable": True},
    "find_imports": {"read_only": True, "cacheable": True},
    "glob_files": {"read_only": True, "cacheable": True},
    "grep_files": {"read_only": True, "cacheable": True},
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
    "find_function_by_name": {"read_only": True, "cacheable": True},
    "find_class_by_name": {"read_only": True, "cacheable": True},
}

_TOOL_META = {
    "bash": {
        "description": "Execute a shell command and return structured stdout, stderr, and exit metadata.",
        "parameters": {
            "command": "Shell command string to execute.",
            "timeout": "Optional timeout in seconds.",
            "normalize_output": "Optional boolean to normalize stdout/stderr newlines to LF.",
        },
    },
    "glob_files": {
        "description": "Find files matching a glob pattern and return structured metadata for each match.",
        "parameters": {
            "pattern": "Glob pattern such as `**/*.py` or `*.md`.",
            "path": "Base directory to search.",
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
            "`content` field only. Do not send extra keys like `array`, `text`, `code`, "
            "or per-variable fragments."
        ),
        "parameters": {
            "path": (
                "Destination file path as a single string. Example: `src/app.py` or "
                "`/tmp/sorting.py`."
            ),
            "content": (
                "Complete file contents as one string. Put ALL code/text inside this "
                "single field. Example JSON: "
                "{\"path\":\"/tmp/sorting.py\",\"content\":\"def bubble_sort(arr):\\n    return arr\\n\"}. "
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
            "{\"tool_call\":{\"name\":\"write_file\",\"arguments\":{\"path\":\"/tmp/sorting.py\",\"content\":\"def bubble_sort(arr):\\n    return arr\\n\"}}}\n\n"
            "Rules:\n"
            "- Put all code inside `content`.\n"
            "- Do not pass extra keys like `array`, `text`, `snippet`, or variable names.\n"
            "- `content` may be a real multiline string or a single string with escaped `\\n`.\n"
            "- Use `mode=\"a\"` only when appending.\n"
            "- For existing files that need a surgical change, prefer `edit_file` over `write_file`."
        ),
    },
    "edit_file": {
        "description": (
            "Replace one exact, unique block inside an existing file. Pass the old text "
            "in `old_string` and the full replacement in `new_string`."
        ),
        "parameters": {
            "path": (
                "Existing file path as a single string. Example: `src/app.py`."
            ),
            "old_string": (
                "Exact text to replace as one string. Include enough surrounding context "
                "to make the match unique. Real multiline text or escaped `\\n` are both allowed."
            ),
            "new_string": (
                "Replacement text as one string. Keep all replacement code inside this single field."
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
            "{\"tool_call\":{\"name\":\"edit_file\",\"arguments\":{\"path\":\"src/app.py\",\"old_string\":\"def f():\\n    return 1\\n\",\"new_string\":\"def f():\\n    return 2\\n\"}}}\n\n"
            "Rules:\n"
            "- Put the full search block in `old_string`.\n"
            "- Put the full replacement block in `new_string`.\n"
            "- Do not split the edit across extra keys like `before`, `after`, or `snippet`.\n"
            "- Both fields may use real multiline text or escaped `\\n`."
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
            "new_code": (
                "Valid replacement Python statement(s) or expression as one string."
            ),
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
            "{\"tool_call\":{\"name\":\"edit_file_libcst\",\"arguments\":{\"path\":\"src/foo.py\",\"old_pattern\":\"return x + y\",\"new_code\":\"return x * 2\"}}}\n\n"
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
            "new_body": (
                "Replacement function body statements only. Do not include `def ...:`."
            ),
        },
    },
    "replace_docstring": {
        "description": "Replace or insert the docstring of a Python function or class.",
        "parameters": {
            "path": "Python file path as a single string.",
            "function_name": "Exact function or class name whose docstring should change.",
            "new_docstring": (
                "Raw docstring text only. Do not include surrounding triple quotes."
            ),
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
    ("bash", bash),
    ("glob_files", glob_files),
    ("grep_files", grep_files),
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
    if name in _RUNTIME_META:
        _apply_tool_meta(_tool_fn, runtime=_RUNTIME_META[name])
    if name in _TOOL_META:
        _apply_tool_meta(_tool_fn, **_TOOL_META[name])


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
