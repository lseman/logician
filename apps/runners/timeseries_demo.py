#!/usr/bin/env python3
"""
Minimal time-series runner using the core Agent + tools.
"""

from src.agent import Agent
from src.config import Config


def main():
    config = Config(
        llama_cpp_url="http://localhost:8080",
        use_chat_api=True,
        chat_template="chatml",
        max_iterations=6,
        temperature=0.3,
        max_tokens=1200,
    )
    agent = Agent(config=config, db_path="run_sessions.db", lazy_rag=True)

    print("Creating sample seasonal series...")
    print(
        agent.run_tool_direct(
            "create_sample_data",
            {"pattern": "seasonal", "n_points": 365, "noise_level": 0.2},
            persist_to_history=False,
        )
    )

    query = (
        "Analyze the loaded time series: basic statistics, trend, weekly seasonality, "
        "anomalies, and a 30-day forecast. End with a concise business summary."
    )
    response = agent.chat(query, session_id="timeseries_demo", fresh_session=True)
    print("\n" + "=" * 60)
    print("Agent response")
    print("=" * 60)
    print(response)


if __name__ == "__main__":
    main()

