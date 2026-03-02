from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from io import StringIO
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from toon_format import decode, encode

    HAS_TOON = True
except ImportError:
    encode = decode = None  # type: ignore
    HAS_TOON = False


@lru_cache(maxsize=1)
def check_optional_deps() -> Dict[str, bool]:
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
    data: Optional[pd.DataFrame] = None
    original_data: Optional[pd.DataFrame] = None
    data_name: str = ""
    freq_cache: Optional[str] = None

    anomaly_store: Dict[str, List[int]] = field(default_factory=dict)
    anomaly_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    nf_best_model: Optional[str] = None
    nf_cv_full: Optional[pd.DataFrame] = None
    nf_pred_col: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.data is not None and len(self.data) > 0

    @property
    def value_columns(self) -> List[str]:
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
        self.nf_best_model = None
        self.nf_cv_full = None
        self.nf_pred_col = None

    @staticmethod
    def _normalize_json_value(value: Any) -> Any:
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
    def _serialize_frame(df: Optional[pd.DataFrame]) -> Optional[str]:
        if df is None:
            return None
        return df.to_json(orient="split", date_format="iso")

    @staticmethod
    def _deserialize_frame(payload: Optional[str]) -> Optional[pd.DataFrame]:
        if not payload:
            return None
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
            "nf_best_model": self.nf_best_model,
            "nf_cv_full": self._serialize_frame(self.nf_cv_full),
            "nf_pred_col": self.nf_pred_col,
        }

    def load_state(self, state: Optional[dict[str, Any]]) -> None:
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


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[ToolParameter]
    function: Callable[..., Any]
    skill_id: str | None = None
    source_path: str | None = None

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
    "ToolParameter",
    "_safe_json_fallback",
    "check_optional_deps",
    "decode",
    "encode",
]
