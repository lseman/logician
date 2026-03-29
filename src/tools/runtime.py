from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from io import StringIO
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from toon_format import decode, encode

    HAS_TOON = True
except ImportError:
    encode = decode = None  # type: ignore
    HAS_TOON = False


@lru_cache(maxsize=1)
def _numpy() -> Any:
    import numpy as np

    return np


@lru_cache(maxsize=1)
def _pandas() -> Any:
    import pandas as pd

    return pd


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


@dataclass
class Context:
    data: pd.DataFrame | None = None
    original_data: pd.DataFrame | None = None
    data_name: str = ""
    freq_cache: str | None = None

    anomaly_store: dict[str, list[int]] = field(default_factory=dict)
    anomaly_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    todo_items: list[dict[str, Any]] = field(default_factory=list)
    mounted_paths: list[dict[str, Any]] = field(default_factory=list)
    rag_docs: list[dict[str, Any]] = field(default_factory=list)
    active_repos: list[dict[str, Any]] = field(default_factory=list)
    retrieval_insights: list[dict[str, Any]] = field(default_factory=list)

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
        self.mounted_paths.clear()
        self.rag_docs.clear()
        self.active_repos.clear()
        self.retrieval_insights.clear()
        self.nf_best_model = None
        self.nf_cv_full = None
        self.nf_pred_col = None

    @staticmethod
    def _normalize_json_value(value: Any) -> Any:
        np = _numpy()
        pd = _pandas()
        if isinstance(value, dict):
            return {
                str(key): Context._normalize_json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [Context._normalize_json_value(item) for item in value]
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
            "mounted_paths": self._normalize_json_value(self.mounted_paths),
            "rag_docs": self._normalize_json_value(self.rag_docs),
            "active_repos": self._normalize_json_value(self.active_repos),
            "retrieval_insights": self._normalize_json_value(self.retrieval_insights),
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
        self.freq_cache = (
            str(state["freq_cache"]) if state.get("freq_cache") is not None else None
        )
        self.anomaly_store = {
            str(key): [int(v) for v in (values or [])]
            for key, values in dict(state.get("anomaly_store") or {}).items()
        }
        self.anomaly_meta = dict(state.get("anomaly_meta") or {})
        self.todo_items = [
            dict(item)
            for item in list(state.get("todo_items") or [])
            if isinstance(item, dict)
        ]
        self.mounted_paths = [
            dict(item)
            for item in list(state.get("mounted_paths") or [])
            if isinstance(item, dict)
        ]
        self.rag_docs = [
            dict(item)
            for item in list(state.get("rag_docs") or [])
            if isinstance(item, dict)
        ]
        self.active_repos = [
            dict(item)
            for item in list(state.get("active_repos") or [])
            if isinstance(item, dict)
        ]
        self.retrieval_insights = [
            dict(item)
            for item in list(state.get("retrieval_insights") or [])
            if isinstance(item, dict)
        ]
        self.nf_best_model = (
            str(state["nf_best_model"])
            if state.get("nf_best_model") is not None
            else None
        )
        self.nf_cv_full = self._deserialize_frame(state.get("nf_cv_full"))
        self.nf_pred_col = (
            str(state["nf_pred_col"]) if state.get("nf_pred_col") is not None else None
        )


def _safe_json_fallback(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps(
            {"status": "error", "error": f"json failed: {e}"}, ensure_ascii=False
        )


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
                    raise TypeError(
                        f"ToolParameter() got multiple values for argument '{key}'"
                    )
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

    @classmethod
    def from_tool_meta(cls, meta: dict[str, Any] | None) -> ToolRuntimeMetadata:
        payload = dict(meta or {})
        runtime_raw = payload.get("runtime")
        runtime = dict(runtime_raw) if isinstance(runtime_raw, dict) else {}
        for key in ("read_only", "writes_files", "verifier", "cacheable"):
            if key in payload and key not in runtime:
                runtime[key] = payload[key]
        return cls(**runtime)


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


@dataclass
class SkillCard:
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


@dataclass
class SkillSelection:
    query: str
    selected_skills: list[SkillCard]
    selected_tools: list[str]
    fallback_tools: list[str]


__all__ = [
    "HAS_TOON",
    "Context",
    "SkillCard",
    "SkillSelection",
    "Tool",
    "ToolCall",
    "ToolRuntimeMetadata",
    "ToolParameter",
    "_safe_json_fallback",
    "check_optional_deps",
    "decode",
    "encode",
]
