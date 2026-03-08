import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src.backends.common import normalize_chat_api_messages
from src.messages import Message, MessageRole


class ChatApiNormalizationTests(unittest.TestCase):
    def test_collapse_system_messages_to_single_leading_message(self) -> None:
        messages = [
            Message(role=MessageRole.SYSTEM, content="base system"),
            Message(role=MessageRole.USER, content="u1"),
            Message(role=MessageRole.SYSTEM, content="nudge one"),
            Message(role=MessageRole.ASSISTANT, content="a1"),
            Message(role=MessageRole.SYSTEM, content="nudge two"),
            Message(role=MessageRole.USER, content="u2"),
        ]

        out = normalize_chat_api_messages(messages, collapse_system_to_first=True)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[0].role, MessageRole.SYSTEM)
        self.assertEqual(
            out[0].content,
            "base system\n\nnudge one\n\nnudge two",
        )
        self.assertEqual([m.role for m in out[1:]], [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.USER,
        ])
        self.assertEqual([m.content for m in out[1:]], ["u1", "a1", "u2"])

    def test_keeps_messages_when_collapse_disabled(self) -> None:
        messages = [
            Message(role=MessageRole.SYSTEM, content="base"),
            Message(role=MessageRole.USER, content="u1"),
            Message(role=MessageRole.SYSTEM, content="nudge"),
        ]
        out = normalize_chat_api_messages(messages, collapse_system_to_first=False)
        self.assertEqual(len(out), len(messages))
        self.assertEqual([m.role for m in out], [m.role for m in messages])
        self.assertEqual([m.content for m in out], [m.content for m in messages])


if __name__ == "__main__":
    unittest.main()

