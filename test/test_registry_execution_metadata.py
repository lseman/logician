from __future__ import annotations

import json
from pathlib import Path

from src.tools import Context, ToolCall, ToolParameter, ToolRegistry
from src.tools.runtime import build_tool


def _registry() -> tuple[ToolRegistry, Context]:
    registry = ToolRegistry(auto_load_from_skills=False)
    ctx = Context()
    registry.install_context(ctx)
    return registry, ctx


def test_execute_respects_validate_input_hook() -> None:
    calls: list[dict[str, str]] = []

    def guarded_tool(path: str) -> dict[str, str]:
        calls.append({"path": path})
        return {"status": "ok", "path": path}

    build_tool(
        guarded_tool,
        name="guarded_tool",
        description="Guarded tool.",
        parameters=[ToolParameter(name="path", type="string", required=True)],
        validate_input=lambda input, ctx: {
            "result": input.get("path") != "forbidden.txt",
            "message": "path is forbidden",
        },
    )

    registry, _ = _registry()
    registry.register(
        name="guarded_tool",
        description="Guarded tool.",
        parameters=[ToolParameter(name="path", type="string", required=True)],
        function=guarded_tool,
    )

    raw = registry.execute(
        call=ToolCall(
            id="validate_1",
            name="guarded_tool",
            arguments={"path": "forbidden.txt"},
        ),
        use_toon=False,
    )
    payload = json.loads(raw)

    assert payload["status"] == "error"
    assert payload["error_type"] == "invalid_arguments"
    assert "forbidden" in payload["error"]
    assert calls == []


def test_execute_respects_check_permissions_hook() -> None:
    def protected_tool(path: str) -> dict[str, str]:
        return {"status": "ok", "path": path}

    build_tool(
        protected_tool,
        name="protected_tool",
        description="Protected tool.",
        parameters=[ToolParameter(name="path", type="string", required=True)],
        check_permissions=lambda input, ctx: {
            "behavior": "deny" if input.get("path", "").startswith("/restricted") else "passthrough",
            "message": "restricted path",
        },
    )

    registry, _ = _registry()
    registry.register(
        name="protected_tool",
        description="Protected tool.",
        parameters=[ToolParameter(name="path", type="string", required=True)],
        function=protected_tool,
    )

    raw = registry.execute(
        call=ToolCall(
            id="perm_1",
            name="protected_tool",
            arguments={"path": "/restricted/file.txt"},
        ),
        use_toon=False,
    )
    payload = json.loads(raw)

    assert payload["status"] == "error"
    assert payload["error_type"] == "permission_blocked"
    assert "restricted path" in payload["error"]


def test_execute_compacts_large_results_and_tracks_state() -> None:
    large_text = "x" * 5000

    def verbose_tool() -> dict[str, str]:
        return {"status": "ok", "content": large_text}

    build_tool(
        verbose_tool,
        name="verbose_tool",
        description="Verbose tool.",
        parameters=[],
        max_result_size_chars=200,
    )

    registry, ctx = _registry()
    registry.register(
        name="verbose_tool",
        description="Verbose tool.",
        parameters=[],
        function=verbose_tool,
    )

    raw = registry.execute(
        call=ToolCall(
            id="compact_1",
            name="verbose_tool",
            arguments={},
        ),
        use_toon=False,
    )
    payload = json.loads(raw)

    assert payload["status"] == "ok"
    assert payload["compacted"] is True
    assert payload["result_path"]
    assert Path(payload["result_path"]).exists()
    assert hasattr(ctx, "content_replacement_state")
    state = ctx.content_replacement_state
    assert state.replacements
