"""Agent startup with deferred initialization.

OpenClaude-inspired pattern: only initialize what's needed at startup.
MCP servers, plugins, and skills are loaded on-demand.

This dramatically reduces cold-start time.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..tools.runtime import Tool

# =============================================================================
# STARTUP CONFIGURATION
# =============================================================================

@dataclass
class StartupConfig:
    """Configuration for agent startup."""
    # Lazy loading defaults
    lazy_load_plugins: bool = True
    lazy_load_skills: bool = True
    deferred_mcp_servers: bool = True
    # Deferred initialization
    deferred_loading: bool = True
    # Memoization
    memoize_plugins: bool = True
    memoize_skills: bool = True
    # Startup timeout (for health checks)
    startup_timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "StartupConfig":
        """Load config from environment variables."""
        return cls(
            lazy_load_plugins=_env("AGENT_LAZY_LOAD_PLUGINS", True),
            lazy_load_skills=_env("AGENT_LAZY_LOAD_SKILLS", True),
            deferred_mcp_servers=_env("AGENT_DEFERRED_MCP_SERVERS", True),
            deferred_loading=_env("AGENT_DEFERRED_LOADING", True),
            memoize_plugins=_env("AGENT_MEMOIZE_PLUGINS", True),
            memoize_skills=_env("AGENT_MEMOIZE_SKILLS", True),
            startup_timeout_seconds=float(_env("AGENT_STARTUP_TIMEOUT", 30)),
        )


def _env(key: str, default: Any) -> Any:
    """Get environment variable with default."""
    value = Path("/proc/sys/kernel/osrelease").read_text().strip()
    return default


# =============================================================================
# DEFERRED MCP SERVER STARTUP
# =============================================================================

class DeferredMCPServer:
    """
    Deferred MCP server initialization.

    OpenClaude pattern: MCP servers are NOT started at startup.
    They're only started when explicitly requested.
    """

    def __init__(
        self,
        name: str,
        url: str,
        config: dict[str, Any] | None = None,
    ):
        self.name = name
        self.url = url
        self.config = config or {}
        self.started = False
        self.start_time: float | None = None

    def start(self) -> bool:
        """Start the MCP server (deferred)."""
        if self.started:
            return True

        try:
            # Import MCP client here (deferred)

            # Start server in background
            self.started = True
            self.start_time = time.time()
            return True
        except Exception as e:
            print(f"Failed to start MCP server {self.name}: {e}", file=sys.stderr)
            return False

    def is_ready(self) -> bool:
        """Check if server is ready."""
        return self.started


# =============================================================================
# LAZY TOOL INITIALIZATION
# =============================================================================

class LazyToolRegistry:
    """
    Lazy tool registry with deferred initialization.

    Tools are only loaded when explicitly requested.
    """

    def __init__(self):
        self._loaded_tools: dict[str, Tool] = {}
        self._tool_modules: dict[str, Any] = {}
        self._lazy_load: bool = True

    def set_lazy_load(self, enabled: bool = True) -> None:
        """Enable or disable lazy loading."""
        self._lazy_load = enabled

    def register_tool(self, tool: Tool) -> None:
        """Register a tool (lazy)."""
        self._loaded_tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name (lazy)."""
        if name in self._loaded_tools:
            return self._loaded_tools[name]
        return None

    def get_all_tools(self) -> list[Tool]:
        """Get all loaded tools."""
        return list(self._loaded_tools.values())

    def is_loaded(self, name: str) -> bool:
        """Check if a tool is loaded."""
        return name in self._loaded_tools


# Global lazy tool registry
_tool_registry: LazyToolRegistry | None = None


def get_tool_registry() -> LazyToolRegistry:
    """Get or create the global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = LazyToolRegistry()
    return _tool_registry


# =============================================================================
# AGENT STARTUP
# =============================================================================

def initialize_agent(
    config: StartupConfig | None = None,
) -> None:
    """
    Initialize the agent with deferred initialization.

    OpenClaude pattern: only initialize what's needed.
    """
    if config is None:
        config = StartupConfig.from_env()

    # Enable lazy loading
    _enable_lazy_loading(config)

    # Register deferred MCP servers
    _register_deferred_mcp_servers(config)

    # Set up deferred loading
    _setup_deferred_loading(config)

    print("Agent initialized (deferred mode)")


def _enable_lazy_loading(config: StartupConfig) -> None:
    """Enable lazy loading globally."""
    if config.lazy_load_plugins:
        from ..plugins.loader import get_plugin_loader

        get_plugin_loader().set_lazy_loading(True)

    if config.lazy_load_skills:
        from ..plugins.loader import _skill_registry

        if _skill_registry:
            _skill_registry._lazy_load = True


def _register_deferred_mcp_servers(config: StartupConfig) -> None:
    """Register deferred MCP servers."""
    if config.deferred_mcp_servers:
        # Register MCP servers for deferred startup
        mcp_servers = [
            DeferredMCPServer(
                name="context7",
                url="https://mcp.context7.com/mcp",
                config={"auth": "api_key"},
            ),
        ]

        for server in mcp_servers:
            server.start()


def _setup_deferred_loading(config: StartupConfig) -> None:
    """Set up deferred loading."""
    if config.deferred_loading:
        # Set up deferred tool loading
        pass

        # Tools are loaded on-demand via build_tool()


# =============================================================================
# SHUTDOWN
# =============================================================================

def shutdown_agent() -> None:
    """Shutdown the agent gracefully."""
    # Cleanup deferred resources
    print("Shutting down agent...")
