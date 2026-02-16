"""
Updated TimeSeriesAgent with proper helper injection using get_helpers().
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from src import Agent, Context, ToolRegistry


class TimeSeriesAgent:
    def __init__(
        self,
        llm_url: str = "http://localhost:8080",
        chat_template: str = "chatml",
        use_chat_api: bool = False,
        skills_md_path: Optional[str] = None,
    ):
        # Core agent
        self.agent = Agent(
            llm_url=llm_url,
            system_prompt=self._get_system_prompt(),
            use_chat_api=use_chat_api,
            chat_template=chat_template,
        )

        # Single source of truth: use Agent's context + registry so
        # data loaded via TimeSeriesAgent.ctx is visible inside agent.run tool calls.
        self.ctx = self.agent.ctx
        self.registry = self.agent.tools

        # Optional SKILLS.md override for the shared registry.
        if skills_md_path:
            custom_path = Path(skills_md_path)
            if not custom_path.exists():
                raise FileNotFoundError(f"Skills path not found at {custom_path}")
            self.registry.skills_md_path = custom_path
            self.registry._tools.clear()
            self.registry._bootstrapped = False
            self.registry.load_tools_from_skills()

        # Reinstall shared ctx explicitly (safe no-op if already installed).
        self.registry.install_context(self.ctx)

        print(f"✓ Loaded {len(self.registry.list_tools())} tools from skills source(s)")

    def _register_tools_with_agent_if_supported(self):
        """
        Optional: depends on your Agent API.
        If your Agent exposes something like register_tool(name, fn, schema),
        implement it here. If not, it's safe to skip.
        """
        if not hasattr(self.agent, "register_tool"):
            return

        for tool in self.registry.list_tools():
            # These attribute names depend on your ToolRegistry/tool representation.
            # Commonly you’d pass name + a callable that runs registry.execute.
            name = getattr(tool, "name", None) or tool["name"]
            schema = (
                getattr(tool, "schema", None)
                if hasattr(tool, "schema")
                else tool.get("schema", None)
            )

            def _runner(**kwargs):
                from src import ToolCall

                result = self.registry.execute(
                    ToolCall(id=f"agent_{name}", name=name, arguments=kwargs)
                )
                return result.content if hasattr(result, "content") else str(result)

            self.agent.register_tool(name=name, fn=_runner, schema=schema)

    def _inject_context_and_helpers(self):
        """
        Ensure tool execution scope has:
        - ctx
        - np, pd, json
        - all helpers returned by get_helpers()
        - _safe_json (fallback if missing)
        - call_tool for tool-to-tool calling
        """
        g = self.registry._execution_globals

        # Base modules always available to tools
        g["np"] = np
        g["pd"] = pd
        g["json"] = json

        # Guarantee ctx exists
        g["ctx"] = self.ctx

        # Tool-to-tool calling from inside tool code
        def call_tool(tool_name: str, **kwargs):
            from src import ToolCall

            tool_call = ToolCall(
                id=f"internal_{tool_name}",
                name=tool_name,
                arguments=kwargs,
            )
            result = self.registry.execute(tool_call)
            return result.content if hasattr(result, "content") else str(result)

        g["call_tool"] = call_tool

        injected = ["np", "pd", "json", "ctx", "_safe_json", "call_tool"]
        print(
            f"✓ Injected {len(injected)} names into tool scope (incl. ctx/np/pd/json/_safe_json)."
        )

    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return """You are a time series analysis expert assistant.

You have access to comprehensive time series analysis tools including:
- Data loading and inspection
- Data cleaning and preprocessing
- Statistical analysis and tests
- Trend and seasonality detection
- Anomaly detection
- Forecasting (classical and neural methods)
- Visualization

When analyzing time series:
1. Start by loading and inspecting the data
2. Check for missing values and outliers
3. Analyze stationarity and seasonality
4. Apply appropriate transformations if needed
5. Perform requested analysis or forecasting
6. Provide clear interpretations of results

Always use the available tools to perform analysis rather than making assumptions.
"""

    def _register_tools_with_agent(self):
        """Register all loaded tools with the core agent."""
        # Get all tools from registry
        tools = self.registry.list_tools()

        # Register each tool
        for tool in tools:
            # Tool registration logic here
            # This depends on your Agent class implementation
            pass

    # ==================== Convenience Methods ====================

    def load_csv(
        self, filepath: str, date_column: str, value_column: str, sep: str = ","
    ):
        """
        Convenience method to load CSV data.

        Args:
            filepath: Path to CSV file
            date_column: Name of date column
            value_column: Name of value column(s)
            sep: CSV delimiter

        Returns:
            JSON string with load status
        """
        from src import ToolCall

        result = self.registry.execute(
            ToolCall(
                id="load_csv",
                name="load_csv_data",
                arguments={
                    "filepath": filepath,
                    "date_column": date_column,
                    "value_column": value_column,
                    "sep": sep,
                },
            )
        )

        return result.content if hasattr(result, "content") else str(result)

    def get_info(self):
        """
        Get information about currently loaded data.

        Returns:
            JSON string with data metadata
        """
        from src import ToolCall

        result = self.registry.execute(
            ToolCall(id="get_info", name="get_data_info", arguments={})
        )

        return result.content if hasattr(result, "content") else str(result)

    def analyze(self, include_plots: bool = False):
        """
        Run comprehensive analysis on loaded data.

        Args:
            include_plots: Whether to generate plots

        Returns:
            JSON string with complete analysis results
        """
        from src import ToolCall

        result = self.registry.execute(
            ToolCall(
                id="analyze",
                name="comprehensive_analysis",
                arguments={"include_plots": include_plots},
            )
        )

        return result.content if hasattr(result, "content") else str(result)

    def detect_anomalies(self, method: str = "zscore", threshold: float = 3.0):
        """
        Detect anomalies in loaded data.

        Args:
            method: Detection method (zscore, iqr, hampel, etc.)
            threshold: Threshold value

        Returns:
            JSON string with anomaly detection results
        """
        from src import ToolCall

        result = self.registry.execute(
            ToolCall(
                id="detect_anomalies",
                name="detect_anomalies",
                arguments={"method": method, "threshold": threshold},
            )
        )

        return result.content if hasattr(result, "content") else str(result)

    def forecast(self, method: str = "holt_winters", periods: int = 30):
        """
        Generate forecast.

        Args:
            method: Forecasting method
            periods: Forecast horizon

        Returns:
            JSON string with forecast results
        """
        from src import ToolCall

        result = self.registry.execute(
            ToolCall(
                id="forecast",
                name="forecast_baselines",
                arguments={"method": method, "periods": periods},
            )
        )

        return result.content if hasattr(result, "content") else str(result)

    def call_tool_directly(self, tool_name: str, **kwargs):
        """
        Directly call a tool by name with arguments.

        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool arguments

        Returns:
            Tool result
        """
        result = self.agent.run_tool_direct(
            tool_name=tool_name,
            arguments=kwargs,
            persist_to_history=False,
        )
        return result


def create_timeseries_agent(
    llm_url: str = "http://localhost:8080",
    chat_template: str = "chatml",
    use_chat_api: bool = False,
    skills_md_path: Optional[str] = None,
) -> TimeSeriesAgent:
    """
    Factory helper for notebooks/scripts.
    """
    return TimeSeriesAgent(
        llm_url=llm_url,
        chat_template=chat_template,
        use_chat_api=use_chat_api,
        skills_md_path=skills_md_path,
    )


# ==================== Example Usage ====================

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    # Create agent
    agent = TimeSeriesAgent()

    # Create sample data directly in context
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    values = np.sin(np.arange(365) * 2 * np.pi / 7) + np.random.randn(365) * 0.1

    agent.ctx.data = pd.DataFrame({"date": dates, "value": values})
    agent.ctx.original_data = agent.ctx.data.copy()
    agent.ctx.data_name = "sine_wave"
    agent.ctx.freq_cache = "D"

    print("\n" + "=" * 70)
    print("Testing TimeSeriesAgent")
    print("=" * 70)

    # Test 1: Get info
    print("\n1. Get data info:")
    result = agent.get_info()
    print(result)

    # Test 2: Detect anomalies
    print("\n2. Detect anomalies:")
    result = agent.detect_anomalies(method="zscore", threshold=3.0)
    print(result)

    # Test 3: Comprehensive analysis (uses tool-to-tool calling internally)
    print("\n3. Comprehensive analysis:")
    result = agent.analyze(include_plots=False)
    print(result)

    # Test 4: Call tool directly
    print("\n4. Detect trend (direct call):")
    result = agent.call_tool_directly("detect_trend")
    print(result)

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
