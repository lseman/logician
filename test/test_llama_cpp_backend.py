import sys
import unittest
from pathlib import Path
from unittest.mock import patch

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src.backends import LlamaCppClient
from src.messages import Message, MessageRole


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _RecordingClient:
    last_request = None

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def request(self, method, url, **kwargs):
        type(self).last_request = {
            "method": method,
            "url": url,
            "kwargs": kwargs,
        }
        if url.endswith("/v1/chat/completions"):
            return _FakeResponse(
                {"choices": [{"message": {"content": "chat-ok"}, "finish_reason": "stop"}]}
            )
        return _FakeResponse({"content": "completion-ok", "stopped_limit": False})


class LlamaCppBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        _RecordingClient.last_request = None
        self.messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Return JSON only."),
        ]

    def test_completion_payload_forwards_grammar(self) -> None:
        llm = LlamaCppClient(
            base_url="http://localhost:8080",
            timeout=10.0,
            use_chat_api=False,
            chat_template="chatml",
            stop=[],
            retry_attempts=0,
        )

        with patch("src.backends.llama_cpp.httpx.Client", _RecordingClient):
            response = llm.generate(
                self.messages,
                temperature=0.1,
                max_tokens=64,
                grammar='root ::= "OK"',
            )

        self.assertEqual(response, "completion-ok")
        req = _RecordingClient.last_request
        assert req is not None
        self.assertEqual(req["url"], "http://localhost:8080/completion")
        self.assertEqual(req["kwargs"]["json"]["grammar"], 'root ::= "OK"')

    def test_chat_payload_forwards_grammar_without_tools(self) -> None:
        llm = LlamaCppClient(
            base_url="http://localhost:8080",
            timeout=10.0,
            use_chat_api=True,
            chat_template="chatml",
            stop=[],
            retry_attempts=0,
        )

        with patch("src.backends.llama_cpp.httpx.Client", _RecordingClient):
            response = llm.generate(
                self.messages,
                temperature=0.1,
                max_tokens=64,
                grammar='root ::= "OK"',
            )

        self.assertEqual(response, "chat-ok")
        req = _RecordingClient.last_request
        assert req is not None
        self.assertEqual(req["url"], "http://localhost:8080/v1/chat/completions")
        self.assertEqual(req["kwargs"]["json"]["grammar"], 'root ::= "OK"')

    def test_chat_payload_omits_grammar_when_using_tools(self) -> None:
        llm = LlamaCppClient(
            base_url="http://localhost:8080",
            timeout=10.0,
            use_chat_api=True,
            chat_template="chatml",
            stop=[],
            retry_attempts=0,
        )

        with patch("src.backends.llama_cpp.httpx.Client", _RecordingClient):
            response = llm.generate(
                self.messages,
                temperature=0.1,
                max_tokens=64,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                grammar='root ::= "OK"',
            )

        self.assertEqual(response, "chat-ok")
        req = _RecordingClient.last_request
        assert req is not None
        self.assertEqual(req["url"], "http://localhost:8080/v1/chat/completions")
        self.assertNotIn("grammar", req["kwargs"]["json"])
        self.assertIn("tools", req["kwargs"]["json"])


if __name__ == "__main__":
    unittest.main()
