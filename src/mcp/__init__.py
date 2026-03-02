"""
MCP (Model Context Protocol) client package.

Provides MCPClient for connecting to remote MCP servers over the
Streamable-HTTP transport and bridging their tools into the agent's
ToolRegistry.
"""

from .client import MCPClient, MCPToolDef

__all__ = ["MCPClient", "MCPToolDef"]
