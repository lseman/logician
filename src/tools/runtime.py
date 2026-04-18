from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _numpy() -> Any:
    try:
        import numpy as np

        return np
    except ImportError:

        class _NumpyFallback:
            generic = type("NPGenericFallback", (), {})
            ndarray = type("NPArrayFallback", (), {})

        return _NumpyFallback()


def _pandas() -> Any:
    try:
        import pandas as pd

        return pd
    except ImportError:

        class _PandasFallback:
            Timestamp = type("PDTimestampFallback", (), {})

            @staticmethod
            def isna(value: Any) -> bool:
                return value is None

        return _PandasFallback()


try:
    from toon_format import decode, encode

    HAS_TOON = True
except ImportError:
    encode = decode = None  # type: ignore
    HAS_TOON = False


# =============================================================================
# Permission Context (OpenClaude-style)
# =============================================================================


@dataclass
class PermissionRule:
    """Tool permission rule with pattern matching."""

    pattern: str
    mode: str = "auto"  # "auto", "plan", "bypass"

    def matches(self, tool_name: str, pattern_value: str | None = None) -> bool:
        """Check if this rule matches a tool/pattern."""
        pattern = str(pattern or "").strip()
        if not pattern:
            return False

        # Exact match
        if pattern == tool_name:
            return True

        # Prefix match (e.g., "review:*" matches "review-pr")
        if pattern.endswith(":*"):
            prefix = pattern[:-2]
            if tool_name.startswith(prefix):
                return True

        return False


@dataclass
class PermissionContext:
    """Centralized permission context for tool execution."""

    default_mode: str = "auto"
    always_allow_rules: dict[str, list[str]] = field(default_factory=dict)
    always_deny_rules: dict[str, list[str]] = field(default_factory=dict)
    always_ask_rules: dict[str, list[str]] = field(default_factory=dict)
    additional_working_dirs: dict[str, str] = field(default_factory=dict)
    is_bypass_available: bool = False

    def get_effective_mode(self, tool_name: str) -> str:
        """Get the effective permission mode for a tool."""
        if self.always_allow_rules.get(tool_name):
            return "bypass"
        if self.always_deny_rules.get(tool_name):
            return "deny"
        return self.default_mode


# =============================================================================
# Progress Tracking (OpenClaude-style)
# =============================================================================


@dataclass
class ToolProgressData:
    """Base type for tool progress data."""

    type: str


@dataclass
class BashProgress(ToolProgressData):
    """Progress data for bash tool."""

    type: str = "bash_progress"
    command: str = ""
    output: str = ""
    exit_code: int | None = None


@dataclass
class TaskOutputProgress(ToolProgressData):
    """Progress data for task output tool."""

    type: str = "task_output_progress"
    task_id: str = ""
    output: str = ""


@dataclass
class FileReadProgress(ToolProgressData):
    """Progress data for file read operations."""

    type: str = "file_read_progress"
    path: str = ""
    bytes_read: int = 0
    total_bytes: int = 0


# =============================================================================
# Tool Type (Unified OpenClaude-style)
# =============================================================================


class Tool(BaseModel):
    """
    Unified Tool type with comprehensive metadata and behavior methods.

    This replaces the simple function-based approach with a rich metadata system
    that supports validation, permissions, progress tracking, and UI rendering.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    name: str
    description: str
    function: Callable[..., Any]
    skill_id: str | None = None
    source_path: str | None = None
    skill_meta: dict[str, Any] | None = None
    runtime: ToolRuntimeMetadata = Field(default_factory=lambda: ToolRuntimeMetadata())

    # Permission methods
    validate_input: Callable[..., Any] | None = None
    check_permissions: Callable[..., Any] | None = None
    is_destructive: Callable[..., Any] | None = None

    # Behavior methods
    interrupt_behavior: Callable[..., Any] | None = None
    is_search_or_read_command: Callable[..., Any] | None = None
    is_concurrency_safe: Callable[..., Any] | None = None
    is_read_only: Callable[..., Any] | None = None

    # Metadata methods
    get_tool_use_summary: Callable[..., Any] | None = None
    get_activity_description: Callable[..., Any] | None = None
    user_facing_name: Callable[..., Any] | None = None
    to_auto_classifier_input: Callable[..., Any] | None = None

    # UI rendering methods
    render_tool_result_message: Callable[..., Any] | None = None
    render_tool_use_message: Callable[..., Any] | None = None
    render_tool_use_progress_message: Callable[..., Any] | None = None
    render_tool_use_queued_message: Callable[..., Any] | None = None
    render_tool_use_rejected_message: Callable[..., Any] | None = None
    render_tool_use_error_message: Callable[..., Any] | None = None
    render_grouped_tool_use: Callable[..., Any] | None = None

    # Execution control
    should_defer: bool = False
    always_load: bool = False

    # Result size limit
    max_result_size_chars: int = 100_000

    # Progress callback
    on_progress: Callable[..., Any] | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = str(value or "").strip()
        if not name:
            raise ValueError("Tool name must be non-empty")
        return name

    @field_validator("description", mode="before")
    @classmethod
    def _normalize_description(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("function")
    @classmethod
    def _validate_function(cls, value: Any) -> Any:
        if not callable(value):
            raise TypeError("Tool function must be callable")
        return value


@lru_cache(maxsize=1)
def check_optional_deps() -> dict[str, bool]:
    deps = {
        "scipy": False,
        "statsmodels": False,
        "ruptures": False,
        "sklearn": False,
        "neuralforecast": False,
    }

    try:
        import scipy  # noqa: F401

        deps["scipy"] = True
    except ImportError:
        pass

    try:
        import statsmodels  # noqa: F401

        deps["statsmodels"] = True
    except ImportError:
        pass

    try:
        import ruptures  # noqa: F401

        deps["ruptures"] = True
    except ImportError:
        pass

    try:
        import sklearn  # noqa: F401

        deps["sklearn"] = True
    except ImportError:
        pass

    try:
        import neuralforecast  # noqa: F401

        deps["neuralforecast"] = True
    except ImportError:
        pass

    return deps


class ToolPermissionRule(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    pattern: str
    mode: str = "auto"

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, value: str) -> str:
        pattern = str(value or "").strip()
        if not pattern:
            raise ValueError("Permission rule pattern must be non-empty")
        return pattern

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> str:
        mode = str(value or "auto").strip().lower()
        if mode not in {"plan", "auto", "bypass"}:
            raise ValueError("Permission rule mode must be one of: plan, auto, bypass")
        return mode


@dataclass
class AppState:
    data: pd.DataFrame | None = None
    original_data: pd.DataFrame | None = None
    data_name: str = ""
    freq_cache: str | None = None

    anomaly_store: dict[str, list[int]] = field(default_factory=dict)
    anomaly_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    todo_items: list[dict[str, Any]] = field(default_factory=list)
    file_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    mounted_paths: list[dict[str, Any]] = field(default_factory=list)
    rag_docs: list[dict[str, Any]] = field(default_factory=list)
    active_repos: list[dict[str, Any]] = field(default_factory=list)
    retrieval_insights: list[dict[str, Any]] = field(default_factory=list)
    permission_rules: list[ToolPermissionRule] = field(default_factory=list)
    permission_default_mode: str = "auto"

    plan_mode: bool = False
    tool_call_count: int = 0

    nf_best_model: str | None = None
    nf_cv_full: pd.DataFrame | None = None
    nf_pred_col: str | None = None

    @property
    def loaded(self) -> bool:
        return self.data is not None and len(self.data) > 0

    @property
    def value_columns(self) -> list[str]:
        if self.data is None:
            return []
        return [c for c in self.data.columns if c != "date"]

    @property
    def is_multivariate(self) -> bool:
        return len(self.value_columns) > 1

    def reset(self) -> None:
        self.data = None
        self.original_data = None
        self.data_name = ""
        self.freq_cache = None
        self.anomaly_store.clear()
        self.anomaly_meta.clear()
        self.todo_items.clear()
        self.file_snapshots.clear()
        self.mounted_paths.clear()
        self.rag_docs.clear()
        self.active_repos.clear()
        self.retrieval_insights.clear()
        self.permission_rules.clear()
        self.permission_default_mode = "auto"
        self.plan_mode = False
        self.tool_call_count = 0
        self.nf_best_model = None
        self.nf_cv_full = None
        self.nf_pred_col = None

    @staticmethod
    def _normalize_json_value(value: Any) -> Any:
        np = _numpy()
        pd = _pandas()
        if isinstance(value, BaseModel):
            return value.model_dump(exclude_none=True)
        if isinstance(value, dict):
            return {str(key): AppState._normalize_json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [AppState._normalize_json_value(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if pd.isna(value):
            return None
        return value

    @staticmethod
    def _serialize_frame(df: pd.DataFrame | None) -> str | None:
        if df is None:
            return None
        return df.to_json(orient="split", date_format="iso")

    @staticmethod
    def _deserialize_frame(payload: str | None) -> pd.DataFrame | None:
        if not payload:
            return None
        pd = _pandas()
        frame = pd.read_json(StringIO(payload), orient="split")
        if "date" in frame.columns:
            try:
                frame["date"] = pd.to_datetime(frame["date"])
            except Exception:
                pass
        return frame

    def to_state(self) -> dict[str, Any]:
        return {
            "data": self._serialize_frame(self.data),
            "original_data": self._serialize_frame(self.original_data),
            "data_name": self.data_name,
            "freq_cache": self.freq_cache,
            "anomaly_store": self._normalize_json_value(self.anomaly_store),
            "anomaly_meta": self._normalize_json_value(self.anomaly_meta),
            "todo_items": self._normalize_json_value(self.todo_items),
            "file_snapshots": self._normalize_json_value(self.file_snapshots),
            "mounted_paths": self._normalize_json_value(self.mounted_paths),
            "rag_docs": self._normalize_json_value(self.rag_docs),
            "active_repos": self._normalize_json_value(self.active_repos),
            "retrieval_insights": self._normalize_json_value(self.retrieval_insights),
            "permission_rules": self._normalize_json_value(self.permission_rules),
            "permission_default_mode": self.permission_default_mode,
            "plan_mode": self.plan_mode,
            "tool_call_count": self.tool_call_count,
            "nf_best_model": self.nf_best_model,
            "nf_cv_full": self._serialize_frame(self.nf_cv_full),
            "nf_pred_col": self.nf_pred_col,
        }

    def load_state(self, state: dict[str, Any] | None) -> None:
        self.reset()
        if not state:
            return

        self.data = self._deserialize_frame(state.get("data"))
        self.original_data = self._deserialize_frame(state.get("original_data"))
        self.data_name = str(state.get("data_name") or "")
        self.freq_cache = str(state["freq_cache"]) if state.get("freq_cache") is not None else None
        self.anomaly_store = {
            str(key): [int(v) for v in (values or [])]
            for key, values in dict(state.get("anomaly_store") or {}).items()
        }
        self.anomaly_meta = dict(state.get("anomaly_meta") or {})
        self.todo_items = [
            dict(item) for item in list(state.get("todo_items") or []) if isinstance(item, dict)
        ]
        self.file_snapshots = {
            str(path): dict(snapshot)
            for path, snapshot in dict(state.get("file_snapshots") or {}).items()
            if isinstance(snapshot, dict)
        }
        self.mounted_paths = [
            dict(item) for item in list(state.get("mounted_paths") or []) if isinstance(item, dict)
        ]
        self.rag_docs = [
            dict(item) for item in list(state.get("rag_docs") or []) if isinstance(item, dict)
        ]
        self.active_repos = [
            dict(item) for item in list(state.get("active_repos") or []) if isinstance(item, dict)
        ]
        self.retrieval_insights = [
            dict(item)
            for item in list(state.get("retrieval_insights") or [])
            if isinstance(item, dict)
        ]
        self.permission_rules = []
        for item in list(state.get("permission_rules") or []):
            if isinstance(item, ToolPermissionRule):
                self.permission_rules.append(item)
                continue
            if not isinstance(item, dict):
                continue
            try:
                self.permission_rules.append(ToolPermissionRule(**item))
            except Exception:
                continue
        mode = str(state.get("permission_default_mode") or "auto").strip().lower()
        self.permission_default_mode = mode if mode in {"plan", "auto", "bypass"} else "auto"
        self.plan_mode = bool(state.get("plan_mode", False))
        self.tool_call_count = int(state.get("tool_call_count", 0))
        self.nf_best_model = (
            str(state["nf_best_model"]) if state.get("nf_best_model") is not None else None
        )
        self.nf_cv_full = self._deserialize_frame(state.get("nf_cv_full"))
        self.nf_pred_col = (
            str(state["nf_pred_col"]) if state.get("nf_pred_col") is not None else None
        )


Context = AppState


def _safe_json_fallback(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": f"json failed: {e}"}, ensure_ascii=False)


class ToolParameter(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    name: str
    type: str
    description: str = ""
    required: bool = True
    enum: list[Any] | None = None

    def __init__(self, *args: Any, **data: Any) -> None:
        if args:
            fields = ("name", "type", "description", "required")
            if len(args) > len(fields):
                raise TypeError("ToolParameter() received too many positional arguments")
            for idx, value in enumerate(args):
                key = fields[idx]
                if key in data:
                    raise TypeError(f"ToolParameter() got multiple values for argument '{key}'")
                data[key] = value
        super().__init__(**data)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = str(value or "").strip()
        if not name:
            raise ValueError("Tool parameter name must be non-empty")
        return name

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, value: Any) -> str:
        raw = str(value or "").strip().lower()
        aliases = {
            "string": "string",
            "str": "string",
            "int": "int",
            "integer": "int",
            "float": "float",
            "number": "float",
            "bool": "bool",
            "boolean": "bool",
            "list": "list",
            "array": "list",
            "dict": "dict",
            "object": "dict",
            "map": "dict",
        }
        return aliases.get(raw, raw or "string")

    @field_validator("enum", mode="before")
    @classmethod
    def _normalize_enum(cls, value: Any) -> list[Any] | None:
        if value is None:
            return None
        np = _numpy()
        if isinstance(value, tuple):
            items = list(value)
        elif isinstance(value, list):
            items = value
        else:
            items = [value]

        out: list[Any] = []
        seen: set[str] = set()
        for item in items:
            norm = item.item() if isinstance(item, np.generic) else item
            if norm is not None and not isinstance(norm, (str, int, float, bool)):
                continue
            key = json.dumps(norm, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            out.append(norm)
        return out or None


class ToolRuntimeMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    read_only: bool | None = None
    writes_files: bool | None = None
    verifier: bool | None = None
    cacheable: bool | None = None
    content_reader: bool | None = None
    concurrency_safe: bool | None = None
    permission_mode: str | None = None

    @classmethod
    def from_tool_meta(cls, meta: dict[str, Any] | None) -> ToolRuntimeMetadata:
        payload = dict(meta or {})
        runtime_raw = payload.get("runtime")
        runtime = dict(runtime_raw) if isinstance(runtime_raw, dict) else {}
        for key in (
            "read_only",
            "writes_files",
            "verifier",
            "cacheable",
            "content_reader",
            "concurrency_safe",
            "permission_mode",
        ):
            if key in payload and key not in runtime:
                runtime[key] = payload[key]
        return cls(**runtime)


def build_tool(
    function: Any,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: list[ToolParameter] | None = None,
    doc: str | None = None,
    runtime: dict[str, Any] | ToolRuntimeMetadata | None = None,
    skill_id: str | None = None,
    source_path: str | None = None,
    skill_meta: dict[str, Any] | None = None,
    validate_input: Callable[..., Any] | None = None,
    check_permissions: Callable[..., Any] | None = None,
    is_destructive: Callable[..., Any] | None = None,
    interrupt_behavior: Callable[..., Any] | None = None,
    is_search_or_read_command: Callable[..., Any] | None = None,
    is_concurrency_safe: Callable[..., Any] | None = None,
    is_read_only: Callable[..., Any] | None = None,
    get_tool_use_summary: Callable[..., Any] | None = None,
    get_activity_description: Callable[..., Any] | None = None,
    user_facing_name: Callable[..., Any] | None = None,
    to_auto_classifier_input: Callable[..., Any] | None = None,
    render_tool_result_message: Callable[..., Any] | None = None,
    render_tool_use_message: Callable[..., Any] | None = None,
    render_tool_use_progress_message: Callable[..., Any] | None = None,
    render_tool_use_queued_message: Callable[..., Any] | None = None,
    render_tool_use_rejected_message: Callable[..., Any] | None = None,
    render_tool_use_error_message: Callable[..., Any] | None = None,
    render_grouped_tool_use: Callable[..., Any] | None = None,
    should_defer: bool = False,
    always_load: bool = False,
    max_result_size_chars: int = 100_000,
    **extra_meta: Any,
) -> Tool:
    """
    Build a Tool instance with metadata and behavior methods.

    This is a factory that creates a fully-featured Tool instance from a function.
    It merges runtime metadata with optional behavior methods for validation,
    permissions, and UI rendering.

    Args:
        function: The tool function to wrap
        name: Tool name (required if not found in function)
        description: Tool description
        parameters: List of ToolParameter for input schema
        doc: Documentation string
        runtime: Runtime metadata (read_only, cacheable, etc.)
        skill_id: Associated skill ID
        source_path: Source file path
        skill_meta: Additional skill metadata
        validate_input: Optional input validation function
        check_permissions: Optional permission check function
        is_destructive: Function returning True for destructive operations
        interrupt_behavior: How tool handles user interruption
        is_search_or_read_command: Returns {isSearch, isRead, isList}
        is_concurrency_safe: Returns True for safe concurrent execution
        is_read_only: Returns True for read-only operations
        get_tool_use_summary: Returns short summary for compact views
        get_activity_description: Returns spinner text for in-progress
        user_facing_name: Returns display name
        to_auto_classifier_input: Returns classifier input string
        render_*_message: UI rendering methods for different states
        should_defer: Whether to defer loading (for large schemas)
        always_load: Whether to load immediately (for essential tools)
        max_result_size_chars: Max result size before compaction

    Returns:
        A fully-configured Tool instance
    """
    target = getattr(function, "__func__", function)

    # Use existing metadata or create new
    existing_meta = dict(getattr(target, "__llm_tool_meta__", {}) or {})
    meta = dict(existing_meta) if existing_meta else {}

    # Set required fields
    if name is None:
        name = getattr(function, "__name__", f"unnamed_{id(function)}")

    meta["name"] = name
    if description is not None:
        meta["description"] = description
    if parameters is not None:
        if isinstance(parameters, dict):
            parsed_parameters: list[ToolParameter] = []
            for param_name, param_description in parameters.items():
                desc_text = str(param_description or "").strip()
                is_optional = desc_text.lower().startswith("optional")
                parsed_parameters.append(
                    ToolParameter(
                        name=str(param_name),
                        type="string",
                        description=desc_text,
                        required=not is_optional,
                    )
                )
            meta["parameters"] = [p.model_dump() for p in parsed_parameters]
        else:
            normalized_parameters = []
            for p in parameters:
                if isinstance(p, ToolParameter):
                    normalized_parameters.append(p.model_dump())
                elif isinstance(p, dict):
                    normalized_parameters.append(ToolParameter(**p).model_dump())
                else:
                    raise TypeError(
                        "Tool parameters must be a ToolParameter, dict, or dict-like iterable"
                    )
            meta["parameters"] = normalized_parameters
    if doc is not None:
        meta["doc"] = doc

    # Runtime metadata
    if runtime is not None:
        runtime_payload = (
            runtime.model_dump(exclude_none=True)
            if isinstance(runtime, ToolRuntimeMetadata)
            else dict(runtime)
        )
        meta["runtime"] = runtime_payload

    # Add optional behavior methods
    if validate_input is not None:
        meta["validate_input"] = validate_input
    if check_permissions is not None:
        meta["check_permissions"] = check_permissions
    if is_destructive is not None:
        meta["is_destructive"] = is_destructive
    if interrupt_behavior is not None:
        meta["interrupt_behavior"] = interrupt_behavior
    if is_search_or_read_command is not None:
        meta["is_search_or_read_command"] = is_search_or_read_command
    if is_concurrency_safe is not None:
        meta["is_concurrency_safe"] = is_concurrency_safe
    if is_read_only is not None:
        meta["is_read_only"] = is_read_only
    if get_tool_use_summary is not None:
        meta["get_tool_use_summary"] = get_tool_use_summary
    if get_activity_description is not None:
        meta["get_activity_description"] = get_activity_description
    if user_facing_name is not None:
        meta["user_facing_name"] = user_facing_name
    if to_auto_classifier_input is not None:
        meta["to_auto_classifier_input"] = to_auto_classifier_input
    if render_tool_result_message is not None:
        meta["render_tool_result_message"] = render_tool_result_message
    if render_tool_use_message is not None:
        meta["render_tool_use_message"] = render_tool_use_message
    if render_tool_use_progress_message is not None:
        meta["render_tool_use_progress_message"] = render_tool_use_progress_message
    if render_tool_use_queued_message is not None:
        meta["render_tool_use_queued_message"] = render_tool_use_queued_message
    if render_tool_use_rejected_message is not None:
        meta["render_tool_use_rejected_message"] = render_tool_use_rejected_message
    if render_tool_use_error_message is not None:
        meta["render_tool_use_error_message"] = render_tool_use_error_message
    if render_grouped_tool_use is not None:
        meta["render_grouped_tool_use"] = render_grouped_tool_use

    # Execution control flags
    meta["should_defer"] = should_defer
    meta["always_load"] = always_load
    meta["max_result_size_chars"] = max_result_size_chars

    # Extra metadata
    for key, value in extra_meta.items():
        meta[key] = value

    # Set metadata on function and return Tool instance
    setattr(target, "__llm_tool_meta__", meta)

    return Tool(
        name=name,
        description=str(meta.get("description") or ""),
        function=function,
        skill_id=skill_id,
        source_path=source_path,
        skill_meta=dict(skill_meta) if isinstance(skill_meta, dict) else skill_meta,
        runtime=ToolRuntimeMetadata.from_tool_meta(meta.get("runtime")),
        validate_input=meta.get("validate_input"),
        check_permissions=meta.get("check_permissions"),
        is_destructive=meta.get("is_destructive"),
        interrupt_behavior=meta.get("interrupt_behavior"),
        is_search_or_read_command=meta.get("is_search_or_read_command"),
        is_concurrency_safe=meta.get("is_concurrency_safe"),
        is_read_only=meta.get("is_read_only"),
        get_tool_use_summary=meta.get("get_tool_use_summary"),
        get_activity_description=meta.get("get_activity_description"),
        user_facing_name=meta.get("user_facing_name"),
        to_auto_classifier_input=meta.get("to_auto_classifier_input"),
        render_tool_result_message=meta.get("render_tool_result_message"),
        render_tool_use_message=meta.get("render_tool_use_message"),
        render_tool_use_progress_message=meta.get("render_tool_use_progress_message"),
        render_tool_use_queued_message=meta.get("render_tool_use_queued_message"),
        render_tool_use_rejected_message=meta.get("render_tool_use_rejected_message"),
        render_tool_use_error_message=meta.get("render_tool_use_error_message"),
        render_grouped_tool_use=meta.get("render_grouped_tool_use"),
        should_defer=bool(meta.get("should_defer", False)),
        always_load=bool(meta.get("always_load", False)),
        max_result_size_chars=int(meta.get("max_result_size_chars", 100_000)),
    )


def materialize_tool(
    function: Any,
    *,
    name: str,
    description: str,
    parameters: list["ToolParameter"] | None = None,
    runtime: dict[str, Any] | ToolRuntimeMetadata | None = None,
    doc: str | None = None,
    skill_id: str | None = None,
    source_path: str | None = None,
    skill_meta: dict[str, Any] | None = None,
    **extra_meta: Any,
) -> "Tool":
    """Build normalized callable metadata and return a Tool model instance."""
    target = getattr(function, "__func__", function)
    existing_meta = dict(getattr(target, "__llm_tool_meta__", {}) or {})
    merged_meta = dict(existing_meta)
    merged_meta.update(extra_meta)
    return build_tool(
        function,
        name=name,
        description=description,
        parameters=parameters,
        doc=doc if doc is not None else merged_meta.get("doc"),
        runtime=runtime if runtime is not None else merged_meta.get("runtime"),
        skill_id=skill_id,
        source_path=source_path,
        skill_meta=skill_meta,
        validate_input=merged_meta.get("validate_input"),
        check_permissions=merged_meta.get("check_permissions"),
        is_destructive=merged_meta.get("is_destructive"),
        interrupt_behavior=merged_meta.get("interrupt_behavior"),
        is_search_or_read_command=merged_meta.get("is_search_or_read_command"),
        is_concurrency_safe=merged_meta.get("is_concurrency_safe"),
        is_read_only=merged_meta.get("is_read_only"),
        get_tool_use_summary=merged_meta.get("get_tool_use_summary"),
        get_activity_description=merged_meta.get("get_activity_description"),
        user_facing_name=merged_meta.get("user_facing_name"),
        to_auto_classifier_input=merged_meta.get("to_auto_classifier_input"),
        render_tool_result_message=merged_meta.get("render_tool_result_message"),
        render_tool_use_message=merged_meta.get("render_tool_use_message"),
        render_tool_use_progress_message=merged_meta.get("render_tool_use_progress_message"),
        render_tool_use_queued_message=merged_meta.get("render_tool_use_queued_message"),
        render_tool_use_rejected_message=merged_meta.get("render_tool_use_rejected_message"),
        render_tool_use_error_message=merged_meta.get("render_tool_use_error_message"),
        render_grouped_tool_use=merged_meta.get("render_grouped_tool_use"),
        should_defer=bool(merged_meta.get("should_defer", False)),
        always_load=bool(merged_meta.get("always_load", False)),
        max_result_size_chars=int(merged_meta.get("max_result_size_chars", 100_000)),
        **{
            key: value
            for key, value in merged_meta.items()
            if key
            not in {
                "name",
                "description",
                "parameters",
                "doc",
                "runtime",
                "validate_input",
                "check_permissions",
                "is_destructive",
                "interrupt_behavior",
                "is_search_or_read_command",
                "is_concurrency_safe",
                "is_read_only",
                "get_tool_use_summary",
                "get_activity_description",
                "user_facing_name",
                "to_auto_classifier_input",
                "render_tool_result_message",
                "render_tool_use_message",
                "render_tool_use_progress_message",
                "render_tool_use_queued_message",
                "render_tool_use_rejected_message",
                "render_tool_use_error_message",
                "render_grouped_tool_use",
                "should_defer",
                "always_load",
                "max_result_size_chars",
            }
        },
    )


class Tool(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    function: Any
    skill_id: str | None = None
    source_path: str | None = None
    skill_meta: dict[str, Any] | None = None
    runtime: ToolRuntimeMetadata = Field(default_factory=lambda: ToolRuntimeMetadata())

    # Permission methods
    validate_input: Callable[..., Any] | None = None
    check_permissions: Callable[..., Any] | None = None
    is_destructive: Callable[..., Any] | None = None

    # Behavior methods
    interrupt_behavior: Callable[..., Any] | None = None
    is_search_or_read_command: Callable[..., Any] | None = None
    is_concurrency_safe: Callable[..., Any] | None = None
    is_read_only: Callable[..., Any] | None = None

    # Metadata methods
    get_tool_use_summary: Callable[..., Any] | None = None
    get_activity_description: Callable[..., Any] | None = None
    user_facing_name: Callable[..., Any] | None = None
    to_auto_classifier_input: Callable[..., Any] | None = None

    # UI rendering methods
    render_tool_result_message: Callable[..., Any] | None = None
    render_tool_use_message: Callable[..., Any] | None = None
    render_tool_use_progress_message: Callable[..., Any] | None = None
    render_tool_use_queued_message: Callable[..., Any] | None = None
    render_tool_use_rejected_message: Callable[..., Any] | None = None
    render_tool_use_error_message: Callable[..., Any] | None = None
    render_grouped_tool_use: Callable[..., Any] | None = None

    # Execution control
    should_defer: bool = False
    always_load: bool = False

    # Result size limit
    max_result_size_chars: int = 100_000

    # Progress callback
    on_progress: Callable[..., Any] | None = None

    def __init__(self, *args: Any, **data: Any) -> None:
        if args:
            fields = (
                "name",
                "description",
                "parameters",
                "function",
                "skill_id",
                "source_path",
                "skill_meta",
                "runtime",
            )
            if len(args) > len(fields):
                raise TypeError("Tool() received too many positional arguments")
            for idx, value in enumerate(args):
                key = fields[idx]
                if key in data:
                    raise TypeError(f"Tool() got multiple values for argument '{key}'")
                data[key] = value
        super().__init__(**data)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = str(value or "").strip()
        if not name:
            raise ValueError("Tool name must be non-empty")
        return name

    @field_validator("description", mode="before")
    @classmethod
    def _normalize_description(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("function")
    @classmethod
    def _validate_function(cls, value: Any) -> Any:
        if not callable(value):
            raise TypeError("Tool function must be callable")
        return value

    @field_validator("runtime", mode="before")
    @classmethod
    def _normalize_runtime(cls, value: Any) -> ToolRuntimeMetadata:
        if isinstance(value, ToolRuntimeMetadata):
            return value
        if isinstance(value, dict):
            return ToolRuntimeMetadata(**value)
        return ToolRuntimeMetadata()

    def __repr__(self) -> str:
        return (
            f"Tool(name={self.name!r}, parameters={len(self.parameters)}, "
            f"description={self.description[:48]!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


# =============================================================================
# Lazy Loading Proxy Pattern (OpenClaude-style)
# =============================================================================


def make_lazy_proxy(
    name: str,
    module: str,
    on_failure: Callable[..., Any] = None,
) -> Callable[..., Any]:
    """
    Create a lazy-loading proxy for optional dependencies.

    This prevents import errors from breaking tool registration when optional
    dependencies are not installed.

    Args:
        name: Name for the proxy function
        module: Module path to import (e.g., "libcst" or "package.submodule")
        on_failure: Optional function to call when import fails

    Returns:
        A proxy function that lazily imports and delegates calls
    """

    import importlib

    _instance: Callable[..., Any] | None = None
    _imported = False

    def _get_instance() -> Callable[..., Any]:
        global _instance, _imported
        if _instance is not None:
            return _instance
        if _imported:
            raise RuntimeError(
                f"Lazy proxy {name} was already attempted to be loaded. "
                "Call reset_lazy_proxy() to retry."
            )
        try:
            _instance = importlib.import_module(module)
            _imported = True
            return _instance
        except ImportError as e:
            if on_failure:
                on_failure(name, str(e))
            return None

    def _reset() -> None:
        global _instance, _imported
        _instance = None
        _imported = False

    def proxy(*args: Any, **kwargs: Any) -> Any:
        instance = _get_instance()
        if instance is None:
            raise ImportError(
                f"{name} requires {module} which is not installed. "
                "Install with: pip install {module}"
            )
        return instance(*args, **kwargs)

    proxy.__name__ = name
    proxy.__module__ = module
    proxy.__doc__ = f"Lazy-loading proxy for {name} (module: {module})"

    return proxy, _reset, _get_instance


@dataclass
class SkillDefinition:
    """
    Unified skill definition combining routing/metadata (SkillCard) and
    rendering capabilities (Skill) into a single dataclass.

    Replaces the previous dual-type system (SkillCard + Skill) with one
    type that supports both skill routing and prompt rendering.

    Fields from SkillCard (routing/metadata):
        id, name, summary, source_path, description, tool_names, playbooks,
        keywords, aliases, triggers, anti_triggers, preferred_tools,
        example_queries, when_not_to_use, next_skills, paths, and all
        OpenClaude-compatible fields (version, model, agent, context,
        effort, allowed_tools, argument_hint, argument_names, metadata).

    Fields from Skill (rendering):
        body, frontmatter, base_dir, _activated.
    """

    # Routing / metadata fields
    id: str
    name: str
    summary: str
    source_path: str
    description: str = ""
    tool_names: list[str] = field(default_factory=list)
    playbooks: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    anti_triggers: list[str] = field(default_factory=list)
    preferred_tools: list[str] = field(default_factory=list)
    example_queries: list[str] = field(default_factory=list)
    when_not_to_use: list[str] = field(default_factory=list)
    next_skills: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    # OpenClaude-compatible fields
    version: str = ""
    model: str = ""
    agent: str = ""
    context: str = ""
    effort: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    argument_hint: str = ""
    argument_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Rendering fields (from legacy Skill dataclass)
    body: str = ""
    frontmatter: dict[str, Any] = field(default_factory=dict)
    base_dir: str = ""
    _activated: bool = False

    # ------------------------------------------------------------------
    # Rendering helpers (moved from legacy Skill dataclass)
    # ------------------------------------------------------------------

    def is_callable(self) -> bool:
        """Whether this skill has executable scripts or tools."""
        fm = self.frontmatter
        if fm.get("callable"):
            return True
        if "scripts" in fm or "script" in fm:
            return True
        return bool(self.tool_names)

    def user_facing_name(self) -> str:
        """Display name for user-facing output."""
        fm_name = self.frontmatter.get("name")
        if fm_name:
            return str(fm_name)
        # Fall back to name field (usually slug-cased)
        return self.name

    def render_prompt(self, args: dict[str, Any], session_id: str = "") -> str:
        """
        Render the skill prompt with template substitution.

        Replaces patterns like ``${key}``, ``${CLAUDE_SKILL_DIR}``,
        ``${CLAUDE_SESSION_ID}`` with their values.
        """
        prompt = self.body
        if not prompt:
            return self.description

        # Build substitution map
        substitutions: dict[str, str] = {}
        substitutions["CLAUDE_SKILL_DIR"] = self.base_dir or str(Path(self.source_path).parent)
        substitutions["CLAUDE_SESSION_ID"] = session_id or ""

        # User-provided args
        substitutions.update(str(v) for v in args.values() if isinstance(v, str))

        # First pass: user args (most specific)
        for key, val in args.items():
            substitutions[key] = str(val)

        # Second pass: built-in substitutions (lower priority)
        substitutions.update(
            {
                "CLAUDE_SKILL_DIR": substitutions["CLAUDE_SKILL_DIR"],
                "CLAUDE_SESSION_ID": substitutions["CLAUDE_SESSION_ID"],
            }
        )

        # Apply substitutions using regex for multi-keyword patterns
        import re

        for key, val in substitutions.items():
            pattern = re.compile(re.escape(key))
            prompt = pattern.sub(val, prompt)

        return prompt


@dataclass
class SkillSelection:
    query: str
    selected_skills: list[SkillDefinition]
    selected_tools: list[str]
    fallback_tools: list[str]


# Backward-compatibility alias
SkillCard = SkillDefinition


__all__ = [
    "HAS_TOON",
    "AppState",
    "Context",
    "SkillCard",
    "SkillDefinition",
    "SkillSelection",
    "Tool",
    "ToolCall",
    "ToolPermissionRule",
    "ToolRuntimeMetadata",
    "ToolParameter",
    "build_tool",
    "materialize_tool",
    "_safe_json_fallback",
    "check_optional_deps",
    "decode",
    "encode",
    # Permission context
    "PermissionContext",
    "PermissionRule",
    # Progress types
    "ToolProgressData",
    "BashProgress",
    "TaskOutputProgress",
    "FileReadProgress",
    # Lazy loading
    "make_lazy_proxy",
]
