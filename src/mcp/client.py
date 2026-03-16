"""
MCPClient — Streamable-HTTP transport client for the Model Context Protocol.

Supports remote MCP servers (e.g. Context7) using the JSON-RPC 2.0 HTTP
transport:
  - POST to a single endpoint for all requests
  - Optional session management via ``Mcp-Session-Id`` headers
  - Response can be plain JSON or an SSE stream; both are handled

No third-party dependencies — pure stdlib (urllib, json).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..logging_utils import get_logger

_MCP_PROTOCOL_VERSION = "2025-03-26"


@dataclass
class MCPToolParameter:
    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class MCPToolDef:
    """Normalised representation of an MCP tool after discovery."""

    name: str
    description: str
    parameters: list[MCPToolParameter] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    # Sanitised Python-safe name (hyphens → underscores, etc.)
    @property
    def safe_name(self) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", self.name)


class MCPClient:
    """
    Client for a single remote MCP server (Streamable-HTTP transport).

    Usage::

        client = MCPClient(
            name="context7",
            url="https://mcp.context7.com/mcp",
            headers={"CONTEXT7_API_KEY": "ctx7sk-..."},
        )
        client.initialize()
        tools = client.list_tools()
        result = client.call_tool("resolve-library-id", {"libraryName": "pandas"})
        client.close()

    The client is also a context manager::

        with MCPClient(...) as client:
            ...
    """

    def __init__(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.url = url
        self._extra_headers: dict[str, str] = headers or {}
        self.timeout = timeout
        self._session_id: str | None = None
        self._request_id: int = 0
        self._initialized: bool = False
        self._log = get_logger(f"agent.mcp.{name}")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def initialize(self) -> dict[str, Any]:
        """Perform the MCP initialize handshake.

        Must be called before ``list_tools`` or ``call_tool``.
        Captures any ``Mcp-Session-Id`` returned by the server.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "logician", "version": "1.0"},
            },
        }
        result = self._rpc(payload)
        self._initialized = True
        self._log.info(
            "MCP server '%s' initialised (protocol=%s server=%s)",
            self.name,
            result.get("protocolVersion", "?"),
            result.get("serverInfo", {}).get("name", "?"),
        )
        # Send initialized notification (fire-and-forget)
        try:
            self._rpc(
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                is_notification=True,
            )
        except Exception:
            pass
        return result

    def list_tools(self) -> list[MCPToolDef]:
        """Discover all tools advertised by the MCP server.

        Follows cursor-based pagination: keeps requesting the next page until
        the server returns no ``nextCursor``.
        """
        if not self._initialized:
            self.initialize()
        defs: list[MCPToolDef] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            result = self._rpc(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "tools/list",
                    **({"params": params} if params else {}),
                }
            )
            raw_tools: list[dict] = result.get("tools") or []
            for raw in raw_tools:
                defs.append(self._parse_tool_def(raw))
            cursor = result.get("nextCursor") or None
            if not cursor:
                break
        self._log.info(
            "MCP server '%s' advertises %d tool(s): %s",
            self.name,
            len(defs),
            [d.name for d in defs],
        )
        return defs

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool and return its content payload.

        Returns the first content item's ``text`` string when the result is a
        standard ``content`` list, otherwise returns the raw result dict.
        """
        if not self._initialized:
            self.initialize()
        result = self._rpc(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
        )
        # MCP tools/call result shape: {"content": [{"type":"text","text":"..."}], ...}
        content = result.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
        return result

    def close(self) -> None:
        """Send a graceful shutdown notification if we have a session."""
        if self._session_id:
            try:
                self._rpc(
                    {"jsonrpc": "2.0", "method": "notifications/cancelled"},
                    is_notification=True,
                )
            except Exception:
                pass
        self._initialized = False
        self._session_id = None

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> MCPClient:
        self.initialize()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        headers.update(self._extra_headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _rpc(
        self,
        payload: dict[str, Any],
        *,
        is_notification: bool = False,
    ) -> dict[str, Any]:
        """Execute a single JSON-RPC call and return the ``result`` field.

        Handles both plain-JSON and SSE-wrapped responses.
        For notifications (no ``id`` field) the response body is ignored.
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = self._build_headers()

        req = urllib.request.Request(
            self.url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                # Capture session ID from response headers
                sid = resp.headers.get("Mcp-Session-Id")
                if sid and not self._session_id:
                    self._session_id = sid
                    self._log.debug("MCP session established: %s", sid)

                content_type = (resp.headers.get("Content-Type") or "").lower()
                raw_body = resp.read().decode("utf-8", errors="replace")

        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")[:600]
            raise RuntimeError(
                f"MCP '{self.name}' HTTP {exc.code} {exc.reason}: {err_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"MCP '{self.name}' connection error: {exc.reason}"
            ) from exc

        if is_notification:
            return {}

        # Parse response — plain JSON or SSE
        if "text/event-stream" in content_type:
            return self._parse_sse_result(raw_body, payload.get("id"))
        return self._parse_json_result(raw_body, payload.get("id"))

    def _parse_json_result(self, raw: str, request_id: int | None) -> dict[str, Any]:
        raw = raw.strip()
        if not raw:
            return {}
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"MCP '{self.name}' malformed JSON: {exc}\nBody: {raw[:300]}"
            ) from exc
        return self._unwrap_envelope(envelope, request_id)

    def _parse_sse_result(self, raw: str, request_id: int | None) -> dict[str, Any]:
        """Extract the JSON-RPC response from an SSE stream body."""
        last_envelope: dict[str, Any] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                envelope = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "result" in envelope or "error" in envelope:
                last_envelope = envelope
                # Prefer the envelope that matches our request ID
                if envelope.get("id") == request_id:
                    break
        return self._unwrap_envelope(last_envelope, request_id)

    @staticmethod
    def _unwrap_envelope(
        envelope: dict[str, Any], request_id: int | None
    ) -> dict[str, Any]:
        if "error" in envelope:
            err = envelope["error"]
            code = err.get("code", "?")
            msg = err.get("message", str(err))
            raise RuntimeError(f"MCP error {code}: {msg}")
        return envelope.get("result") or {}

    @staticmethod
    def _parse_tool_def(raw: dict[str, Any]) -> MCPToolDef:
        name = raw.get("name", "unknown_tool")
        description = raw.get("description", "")
        input_schema = raw.get("inputSchema") or raw.get("input_schema") or {}
        properties = input_schema.get("properties") or {}
        required_set = set(input_schema.get("required") or [])

        params: list[MCPToolParameter] = []
        for pname, pdef in properties.items():
            ptype = pdef.get("type") or "string"
            # Handle anyOf / oneOf type unions — pick the first non-null
            if not isinstance(ptype, str):
                ptype = "string"
            pdesc = pdef.get("description") or ""
            params.append(
                MCPToolParameter(
                    name=pname,
                    type=ptype,
                    description=pdesc,
                    required=(pname in required_set),
                )
            )

        return MCPToolDef(
            name=name, description=description, parameters=params, raw=raw
        )
