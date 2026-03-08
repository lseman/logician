#!/usr/bin/env python3
"""
REPL demo runner for the core Agent.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import Agent
from src.config import Config


def main():
    print("Starting Agent REPL Demo")
    print("=" * 60)

    config = Config(
        llama_cpp_url="http://localhost:8080",
        use_chat_api=True,
        chat_template="chatml",
        max_iterations=5,
        pre_turn_thinking=True,
        temperature=0.7,
        max_tokens=2048,
    )

    agent = Agent(
        config=config,
        db_path="repl_demo_sessions.db",
        lazy_rag=True,
    )

    print("\nAgent initialized")
    print(f"Available tools: {len(agent.tools.list_tools())}")
    print(f"Memory module: {type(agent.memory).__name__}")
    print("\n" + "=" * 60)

    agent.repl()
    print("\nREPL session ended.")


if __name__ == "__main__":
    main()
