from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from logician_bridge import BridgeServer


class _StubRunResponse:
    def __init__(self, final_response: str) -> None:
        self.final_response = final_response
        self.tool_calls = []
        self.messages = []
        self.iterations = 1


class _StubAgent:
    def run(self, *args, **kwargs):
        stream_callback = kwargs.get("stream_callback")
        if callable(stream_callback):
            for chunk in ["H", "e", "l", "l", "o"]:
                stream_callback(chunk)
        return _StubRunResponse("Hello")


class BridgeStreamBatchingTests(unittest.TestCase):
    def test_chat_batches_follow_on_stream_tokens(self) -> None:
        server = BridgeServer()
        server.active = "main"
        server.sessions = {"main": "sess-1"}
        server.agents = {"main": _StubAgent()}
        server.cfg = {"project_memory_enabled": False}

        emitted: list[tuple[str, dict[str, object]]] = []

        with (
            patch.object(BridgeServer, "_state_snapshot", return_value={}),
            patch(
                "src.agent.classify.classify_turn",
                return_value=SimpleNamespace(intent="social"),
            ),
        ):
            server._emit = lambda event, payload: emitted.append((event, payload))
            result = server.chat({"message": "hello"})

        self.assertEqual(result["final_response"], "Hello")
        token_events = [payload["token"] for event, payload in emitted if event == "token"]
        self.assertEqual(token_events, ["H", "ello"])


if __name__ == "__main__":
    unittest.main()
