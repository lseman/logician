#!/usr/bin/env python3
"""
Minimal agent test - exactly as user requested.
"""

import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src import create_agent, ToolParameter

def ping_tool(text: str) -> str:
    return f"PONG: {text}"


def main() -> None:
    print("Creating agent...")
    agent = create_agent(
        embedding_model=None,
        config_overrides={"rag_enabled": False},
    )
    print("✅ Agent created.")

    agent.add_tool(
        name="ping",
        description="Simple echo ping.",
        function=ping_tool,
        parameters=[ToolParameter("text", "string", "Text to echo")],
    )
    print("✅ Tool added.")

    print("\n" + "=" * 60)
    print("Testing simple chat (no tools)...")
    print("=" * 60)

    response = agent.chat("Say hello (no tools).", fresh_session=True)

    print("\n" + "=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(response)
    print("\n" + "=" * 60)

    if len(response) < 200 and ("hello" in response.lower() or "hi" in response.lower()):
        print("✅ Response looks GOOD")
    else:
        print("❌ Response looks BAD (too long or irrelevant)")
        print(f"   Length: {len(response)} chars")
        print(f"   Contains 'hello': {'hello' in response.lower()}")


class MinimalAgentSmokeTests(unittest.TestCase):
    def test_ping_tool(self) -> None:
        self.assertEqual(ping_tool("hi"), "PONG: hi")


if __name__ == "__main__":
    main()
