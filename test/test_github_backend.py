import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src.backends import GitHubClient
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
    last_init = None

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        type(self).last_init = kwargs

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
        return _FakeResponse(
            {"choices": [{"message": {"content": "github-chat-ok"}, "finish_reason": "stop"}]}
        )


class GitHubBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        _RecordingClient.last_request = None
        self.messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Generate a short answer."),
        ]

    def test_chat_payload_includes_github_model(self) -> None:
        llm = GitHubClient(
            base_url="https://api.github.com",
            model="gpt-4o-mini",
            token="ghp_testtoken",
            api_version="2024-12-03",
            stop=[],
            retry_attempts=0,
        )

        with patch("src.backends.github.httpx.Client", _RecordingClient):
            response = llm.generate(
                self.messages,
                temperature=0.1,
                max_tokens=32,
                stream=False,
            )

        self.assertEqual(response, "github-chat-ok")
        req = _RecordingClient.last_request
        assert req is not None
        self.assertEqual(req["url"], "https://api.github.com/v1/chat/completions")
        self.assertEqual(req["kwargs"]["json"]["model"], "gpt-4o-mini")
        self.assertEqual(req["kwargs"]["json"]["temperature"], 0.1)
        self.assertEqual(req["kwargs"]["json"]["max_tokens"], 32)
        self.assertEqual(req["kwargs"]["json"]["stop"], [])

    def test_chat_headers_include_github_token(self) -> None:
        llm = GitHubClient(
            base_url="https://api.github.com",
            model="gpt-4o-mini",
            token="ghp_testtoken",
            api_version="2024-12-03",
            stop=["\n"],
            retry_attempts=0,
        )

        with patch("src.backends.github.httpx.Client", _RecordingClient):
            llm.generate(self.messages, temperature=0.2, max_tokens=10)

        init_kwargs = _RecordingClient.last_init
        assert init_kwargs is not None
        self.assertEqual(init_kwargs["headers"]["Authorization"], "Bearer ghp_testtoken")
        self.assertEqual(init_kwargs["headers"]["X-GitHub-Api-Version"], "2024-12-03")
        self.assertEqual(init_kwargs["headers"]["Accept"], "application/vnd.github+json")
        self.assertEqual(init_kwargs["headers"]["User-Agent"], "logician-github-backend/1.0")

    def test_chat_with_real_github_token(self) -> None:
        config_path = AGENT_ROOT / "agent_config.json"
        token = os.environ.get("GITHUB_TOKEN")
        model = os.environ.get("GITHUB_MODEL")
        base_url = os.environ.get("GITHUB_MODEL_URL")
        api_version = os.environ.get("GITHUB_API_VERSION")

        if config_path.is_file():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
            token = token or cfg.get("github_token")
            model = model or cfg.get("github_model")
            base_url = base_url or cfg.get("github_api_url")
            api_version = api_version or cfg.get("github_api_version")

        if not token:
            self.skipTest("GITHUB_TOKEN not set in environment or agent_config.json")

        model = model or "gpt-4o-mini"
        base_url = base_url or "https://api.github.com"
        api_version = api_version or "2024-12-03"

        llm = GitHubClient(
            base_url=base_url,
            model=model,
            token=token,
            api_version=api_version,
            stop=["\n"],
            retry_attempts=1,
        )

        try:
            response = llm.generate(self.messages, temperature=0.1, max_tokens=16)
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                self.skipTest(
                    "GitHub Models API endpoint returned 404. Token may not have AI access or the endpoint is unavailable."
                )
            raise

        self.assertIsInstance(response, str)
        self.assertTrue(response.strip())
