#!/usr/bin/env python3
"""
Runner to validate clean-session behavior.
"""

from src.agent import Agent
from src.config import Config


def main():
    print("Testing Agent with clean session")
    print("=" * 60)

    config = Config(
        llama_cpp_url="http://localhost:8080",
        use_chat_api=True,
        chat_template="chatml",
        max_iterations=3,
        temperature=0.7,
        max_tokens=512,
        rag_enabled=False,
    )

    agent = Agent(
        config=config,
        db_path="clean_test_sessions.db",
        lazy_rag=True,
    )

    print("\nAgent initialized")
    print(f"Tools: {len(agent.tools.list_tools())}")
    print(f"RAG: {'disabled' if not config.rag_enabled else 'enabled'}")
    print(f"Max tokens: {config.max_tokens}")
    print("\n" + "=" * 60)

    print("\nTest 1: introduction (fresh session)")
    response = agent.chat(
        "Hello! Can you introduce yourself?",
        fresh_session=True,
        verbose=True,
    )
    print("\n" + "=" * 60)
    print("Response:")
    print("=" * 60)
    print(response)
    print("\n" + "=" * 60)

    print("\nTest 2: follow-up")
    response2 = agent.chat(
        "What tools do you have for time series analysis?",
        verbose=False,
    )
    print("\nResponse:")
    print(response2)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

