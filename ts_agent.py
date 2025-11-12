# agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import json
from typing import Any, Dict, List, Optional, get_type_hints

import numpy as np
import pandas as pd
from core import Agent
from core import ToolParameter
from ts_tools import (
    AnalysisTools,
    AnomalyTools,
    Context,
    DataTools,
    ForecastBaselineTools,
    HygieneTools,
    NeuralForecastTools,
    PlotTools,
    SeasonalityTools,
    StatsTools,
    TransformTools,
    _infer_freq_safe,
    _safe_json,
)

from helpers import *
# ─────────────────────────────────────────────────────────────────────────────
# Agent with improved tool mounting
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesAgent:
    """Slim façade around Agent + class-based tools. Keeps your previous interface."""
    def __init__(self, llm_url: str = "http://localhost:8080", chat_template: str = "chatml", use_chat_api: bool = False):
        self.ctx = Context()
        self.agent = Agent(
            llm_url=llm_url,
            system_prompt=self._get_system_prompt(),
            use_chat_api=use_chat_api,
            chat_template=chat_template
        )
        # De-dup state for tool registration
        self._seen_functions = set()   # function ids
        self._registered_tools: Dict[str, Any] = {}  # tool name -> method
        # NEW: automatic tool mounting
        self._mount_tools_auto()

    # ——— public convenience ———
    def set_numpy(self, arr: np.ndarray, start_date: str = "2018-01-01", freq: str = "D", name: str = "numpy_series"):
        return DataTools(self.ctx).set_numpy(arr, start_date, freq, name)

    def set_data(self, df: pd.DataFrame, name: str = "custom_data"):
        if "date" not in df.columns or "value" not in df.columns:
            raise ValueError("DataFrame must have 'date' and 'value' columns")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        self.ctx.data = df.sort_values("date")
        self.ctx.original_data = self.ctx.data.copy()
        self.ctx.data_name = name
        self.ctx.freq_cache = _infer_freq_safe(self.ctx.data["date"])
        return self

    def get_data(self) -> Optional[pd.DataFrame]:
        return self.ctx.data

    def reset(self):
        self.agent.reset()
        self.ctx = Context()
        # Also reset registration caches in case caller remounts
        self._seen_functions = set()
        self._registered_tools = {}
        return self

    # ——— agent passthrough ———
    def chat(self, message: str, verbose: bool = False) -> str:
        if self.ctx.data is not None and "data is already loaded" not in message.lower():
            info = DataTools(self.ctx).get_data_info()
            enhanced = f"Note: Time series data is already loaded: {info}\n\nUser request: {message}"
            return self.agent.chat(enhanced, verbose=verbose)
        return self.agent.chat(message, verbose=verbose)

    def run(self, message: str, verbose: bool = False):
        if self.ctx.data is not None and "data is already loaded" not in message.lower():
            info = DataTools(self.ctx).get_data_info()
            enhanced = f"Note: Time series data is already loaded: {info}\n\nUser request: {message}"
            return self.agent.run(enhanced, verbose=verbose)
        return self.agent.run(message, verbose=verbose)

    # ——— internals ———
    def _get_system_prompt(self) -> str:
        return (
            "You are a time series analysis expert with visualization capabilities. "
            "You help users prepare data for forecasting by cleaning, transforming, and analyzing series.\n\n"
            "CRITICAL:\n"
            "1) Prefer using already-loaded data; only call load tools if truly needed.\n"
            "2) Always return tool calls as a JSON object EXACTLY like:\n"
            "{\n"
            '  \"tool_call\": {\n'
            '    \"name\": \"tool_name\",\n'
            '    \"arguments\": { \"param1\": \"value1\" }\n'
            "  }\n"
            "}\n"
            "3) For tools without parameters: use an empty arguments object.\n"
            "4) Use plotting tools to visualize data and analysis results.\n"
            "5) When preparing data for forecasting, consider: regularization, outlier removal, "
            "stationarity (polynomial detrending), scaling, and seasonality.\n"
            "6) Do not call the same tool twice in a row with identical arguments. "
            "If it already ran, proceed to the next step or explain why another run is needed.\n"
        )

    # --- tool registration helpers ---
    def _register_tool(self, inst, tool_name: str, desc: str, method, params):
        """Register tool once. If a name collision occurs with a different method,
        namespace the name as ClassName.tool_name."""
        # function identity (bound method -> get underlying function)
        fn = getattr(method, "__func__", method)
        fid = id(fn)

        # 1) function-level de-dup
        if fid in self._seen_functions:
            return

        # 2) name-level de-dup (if same name but different function, namespace it)
        final_name = tool_name
        if final_name in self._registered_tools and self._registered_tools[final_name] is not method:
            final_name = f"{inst.__class__.__name__}.{tool_name}"

        # If still collides (extremely unlikely), bail
        if final_name in self._registered_tools:
            return

        # Register once
        self.agent.add_tool(final_name, desc, method, params)
        self._registered_tools[final_name] = method
        self._seen_functions.add(fid)

    def _mount_tools_auto(self):
        """
        Automatically discover tools from class-based tool instances.
        Exposure via @as_tool or Class.__tools__.
        Ensures each underlying method is registered ONCE and namespaced on collision.
        """
        # Instantiate once (shared Context)
        instances = [
            DataTools(self.ctx),
            HygieneTools(self.ctx),
            TransformTools(self.ctx),
            StatsTools(self.ctx),
            SeasonalityTools(self.ctx),
            AnomalyTools(self.ctx),
            ForecastBaselineTools(self.ctx),
            PlotTools(self.ctx),
            NeuralForecastTools(self.ctx),
            AnalysisTools(self.ctx),
        ]

        # Collect de-duplicated (by function id) first
        seen_fn_ids = set()
        collected = []  # list[(inst, tool_name, desc, method, params)]
        for inst in instances:
            for tool_name, desc, method, params in _iter_exposed_methods(inst):
                fn = getattr(method, "__func__", method)
                fid = id(fn)
                if fid in seen_fn_ids:
                    continue
                seen_fn_ids.add(fid)
                collected.append((inst, tool_name, desc, method, params))

        # Now register with collision-aware naming
        for inst, tool_name, desc, method, params in collected:
            self._register_tool(inst, tool_name, desc, method, params)

        # Optional: debug list
        # print("Mounted tools:", sorted(self._registered_tools.keys()))


# Factory (unchanged)
def create_timeseries_agent(
    llm_url: str = "http://localhost:8080",
    chat_template: str = "chatml",
    use_chat_api: bool = False
) -> TimeSeriesAgent:
    return TimeSeriesAgent(llm_url=llm_url, chat_template=chat_template, use_chat_api=use_chat_api)
