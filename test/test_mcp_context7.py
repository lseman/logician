#!/usr/bin/env python3
"""
Integration test for the Context7 remote MCP server.

Checks:
  1. Handshake / initialize succeeds and returns protocol version + serverInfo.
  2. tools/list returns at least the two core tools
     (resolve-library-id, get-library-docs).
  3. resolve-library-id resolves a well-known library (numpy).
  4. get-library-docs fetches a doc snippet for the resolved ID.

Each test class reuses a single MCPClient connection (opened in setUpClass)
so we only pay TLS setup cost once per suite.

Run:
    python test/test_mcp_context7.py
    # or
    pytest test/test_mcp_context7.py -v

Requirements:
  - Network access to https://mcp.context7.com/mcp
  - Valid API key in agent_config.json  ->  mcp.context7.headers.Authorization
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src.mcp.client import MCPClient, MCPToolDef

# -- Config loader ------------------------------------------------------------

_CONFIG_PATH = AGENT_ROOT / "agent_config.json"


def _load_context7_cfg() -> dict:
    with _CONFIG_PATH.open() as fh:
        cfg = json.load(fh)
    block = cfg.get("mcp", {}).get("context7")
    if not block:
        raise RuntimeError("No mcp.context7 block found in agent_config.json")
    if not block.get("enabled", True):
        raise unittest.SkipTest("context7 MCP is disabled in agent_config.json")
    return block


def _make_client() -> MCPClient:
    cfg = _load_context7_cfg()
    return MCPClient(
        name="context7",
        url=cfg["url"],
        headers=cfg.get("headers", {}),
        timeout=30.0,
    )


# -- Shared client mixin ------------------------------------------------------


class _WithClient(unittest.TestCase):
    """Base class: opens one shared MCPClient for the entire test class."""

    _client: MCPClient
    _init_result: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls._client = _make_client()
        cls._init_result = cls._client.initialize()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._client.close()


# -- Test cases ---------------------------------------------------------------


class TestContext7Handshake(_WithClient):
    """Protocol-level initialization checks."""

    def test_client_is_initialized(self) -> None:
        self.assertTrue(
            self._client._initialized,
            "Client should be initialized after initialize()",
        )

    def test_server_info_present(self) -> None:
        result = self._init_result
        self.assertIn("protocolVersion", result)
        self.assertIn("serverInfo", result)
        server_name = result["serverInfo"].get("name", "")
        self.assertTrue(server_name, "serverInfo.name must be non-empty")
        print(f"\n  server={server_name!r}  protocol={result['protocolVersion']!r}")


class TestContext7ToolDiscovery(_WithClient):
    """Verify the tools advertised by the MCP server."""

    _REQUIRED = {"resolve-library-id", "query-docs"}
    # Accepted alternatives for the doc-fetching tool (Context7 renames it across versions)
    _DOC_TOOL_VARIANTS = {"query-docs", "get-library-docs"}
    _tools: list[MCPToolDef]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._tools = cls._client.list_tools()

    def test_required_tools_present(self) -> None:
        names = {t.name for t in self._tools}
        print(f"\n  Advertised tools: {sorted(names)}")
        # Print full parameter schema for each tool (aids debugging)
        for t in self._tools:
            params = [(p.name, p.type, p.required) for p in t.parameters]
            print(f"  {t.name}: {params}")
        self.assertIn(
            "resolve-library-id", names, "resolve-library-id must be advertised"
        )
        self.assertTrue(
            names & self._DOC_TOOL_VARIANTS,
            f"Expected one of {self._DOC_TOOL_VARIANTS} in advertised tools, got {names}",
        )

    def test_all_tools_have_descriptions(self) -> None:
        for t in self._tools:
            self.assertTrue(
                t.description.strip(), f"Tool '{t.name}' has an empty description"
            )

    def test_all_tools_have_safe_names(self) -> None:
        for t in self._tools:
            self.assertRegex(
                t.safe_name,
                r"^[a-zA-Z0-9_]+$",
                f"safe_name for '{t.name}' is not Python-safe",
            )


class TestContext7ToolCalls(_WithClient):
    """End-to-end tool invocation tests."""

    # Populated in setUpClass from the live tool schema
    _resolve_args: dict = {"query": "{library}", "libraryName": "{library}"}
    _doc_tool: str | None = None
    _docs_id_param: str = "libraryId"
    _docs_tools: set

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        tools = {t.name: t for t in cls._client.list_tools()}
        # Build the args dict for resolve-library-id from ALL required string params
        resolve_params = tools.get("resolve-library-id")
        if resolve_params:
            cls._resolve_args = {
                p.name: "{library}"
                for p in resolve_params.parameters
                if p.type == "string"
            }
        else:
            cls._resolve_args = {"query": "{library}", "libraryName": "{library}"}
        # Discover the doc-fetching tool and its ID parameter
        cls._docs_tools = {n for n in tools if "doc" in n or "query" in n} - {
            "resolve-library-id"
        }
        for doc_tool_name in ("query-docs", "get-library-docs"):
            if doc_tool_name in tools:
                t = tools[doc_tool_name]
                for candidate in (
                    "libraryId",
                    "context7CompatibleLibraryID",
                    "library_id",
                ):
                    if any(p.name == candidate for p in t.parameters):
                        cls._docs_id_param = candidate
                        break
                else:
                    cls._docs_id_param = (
                        t.parameters[0].name if t.parameters else "libraryId"
                    )
                cls._doc_tool = doc_tool_name
                break
        else:
            cls._doc_tool = next(iter(cls._docs_tools), None)
            cls._docs_id_param = "libraryId"
        print(
            f"\n  [setUpClass] resolve_args={cls._resolve_args}"
            f"  doc_tool={cls._doc_tool!r}  doc_id_param={cls._docs_id_param!r}"
        )

    def test_resolve_library_id_numpy(self) -> None:
        args = {
            k: v.replace("{library}", "numpy") for k, v in self._resolve_args.items()
        }
        result = self._client.call_tool("resolve-library-id", args)
        print(f"\n  resolve-library-id({args}) -> {str(result)[:300]}")
        self.assertTrue(result, "resolve-library-id should return a non-empty result")
        self.assertFalse(
            str(result).startswith("MCP error"),
            f"resolve-library-id returned an error: {str(result)[:200]}",
        )

    def test_get_library_docs_numpy(self) -> None:
        self.assertIsNotNone(self._doc_tool, "No doc-fetching tool discovered")
        args = {
            k: v.replace("{library}", "numpy") for k, v in self._resolve_args.items()
        }
        raw = self._client.call_tool("resolve-library-id", args)
        self.assertFalse(
            str(raw).startswith("MCP error"),
            f"resolve-library-id returned an error: {str(raw)[:200]}",
        )
        library_id = _extract_library_id(raw)
        self.assertTrue(
            library_id, f"Could not extract library ID from: {str(raw)[:200]}"
        )
        print(f"\n  Using library ID: {library_id!r}")

        docs = self._client.call_tool(
            self._doc_tool,
            {self._docs_id_param: library_id, "query": "array creation"},
        )
        docs_str = docs if isinstance(docs, str) else json.dumps(docs)
        print(f"  Docs snippet (first 300 chars): {docs_str[:300]}")
        self.assertFalse(
            docs_str.startswith("MCP error"),
            f"query-docs returned an error: {docs_str[:200]}",
        )
        self.assertGreater(
            len(docs_str), 50, "Documentation text should be substantial"
        )


# -- Helpers ------------------------------------------------------------------


def _extract_library_id(raw: object) -> str | None:
    """Parse a resolve-library-id result and return the first library ID.

    Context7 returns either:
      - A plain-text report ("Available Libraries:\\n- Context7-compatible library ID: /numpy/numpy")
      - A JSON object {"results": [{"id": "/numpy/numpy", ...}], ...}
      - A plain string that IS the ID (starts with "/")
    """
    text = raw if isinstance(raw, str) else json.dumps(raw)

    # 1) Plain-text format: "- Context7-compatible library ID: /some/id"
    m = re.search(r"Context7-compatible library ID:\s*(\S+)", text)
    if m:
        return m.group(1).strip()

    # 2) JSON format
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("results", "libraries", "items"):
                items = data.get(key)
                if items:
                    first = items[0]
                    for id_key in ("id", "libraryId", "library_id"):
                        val = first.get(id_key)
                        if val:
                            return val
    except (json.JSONDecodeError, TypeError, IndexError, AttributeError):
        pass

    # 3) First line is a bare ID
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if first_line.startswith("/"):
        return first_line

    return None


# -- CLI runner ----------------------------------------------------------------


def main() -> None:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestContext7Handshake,
        TestContext7ToolDiscovery,
        TestContext7ToolCalls,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
