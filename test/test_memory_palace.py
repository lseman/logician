from __future__ import annotations

import json
from pathlib import Path

from src.config import Config
from src.memory import Memory
from src.messages import Message, MessageRole


def test_memory_palace_persists_raw_messages(tmp_path: Path) -> None:
    config = Config()
    config.memory_palace_enabled = True
    config.memory_palace_db_path = str(tmp_path / "memory_palace.db")
    config.memory_palace_apply_aaak = False
    config.rag_enabled = False

    storage_path = str(tmp_path / "agent_sessions.db")
    memory = Memory(config=config, db_path=storage_path, embedding_model=None)

    memory.save_message(
        "session-1", Message(role=MessageRole.USER, content="User prefers Postgres for durability.")
    )
    memory.save_message(
        "session-1",
        Message(
            role=MessageRole.ASSISTANT, content="We should keep the full conversation in chromadb."
        ),
    )

    results = memory.search_memory_palace("Postgres", n_results=4)
    assert results
    assert any("Postgres" in str(item.get("content", "")) for item in results)

    results2 = memory.search_memory_palace("chromadb", n_results=4)
    assert results2
    assert any("chromadb" in str(item.get("content", "")).lower() for item in results2)

    # Ensure the persistent store still exists after creating the Palace.
    assert Path(config.memory_palace_db_path).exists()
    metadata_json = json.dumps([item.get("metadata") for item in results], ensure_ascii=False)
    assert "session-1" in metadata_json


def test_message_history_stays_raw_without_loading_message_vectors(tmp_path: Path) -> None:
    config = Config()
    config.rag_enabled = False
    config.message_history_vector_enabled = False
    config.memory_palace_enabled = False

    memory = Memory(
        config=config,
        db_path=str(tmp_path / "agent_sessions.db"),
        embedding_model=None,
    )

    assert memory._db is not None
    assert memory._db.vector_enabled is False
    assert memory._db._collection is None

    memory.save_message("session-raw", Message(role=MessageRole.USER, content="keep this raw"))
    rows = memory.get_session_messages("session-raw")
    assert [msg.content for msg in rows] == ["keep this raw"]


def test_load_history_includes_cross_session_palace_context(tmp_path: Path) -> None:
    config = Config()
    config.rag_enabled = False
    config.memory_palace_enabled = True
    config.memory_palace_db_path = str(tmp_path / "memory_palace.db")
    config.memory_palace_apply_aaak = False
    config.memory_palace_context_enabled = True
    config.memory_palace_context_max_results = 3
    config.history_limit = 5

    memory = Memory(
        config=config,
        db_path=str(tmp_path / "agent_sessions.db"),
        embedding_model=None,
    )

    memory.save_message(
        "session-a",
        Message(role=MessageRole.USER, content="User prefers Postgres for durability."),
    )
    memory.save_message(
        "session-b",
        Message(role=MessageRole.USER, content="Current task is unrelated."),
    )
    memory.save_message(
        "session-b",
        Message(role=MessageRole.ASSISTANT, content="Working on the current request."),
    )

    history = memory.load_history(
        "session-b",
        message="postgres durability",
        use_semantic_retrieval=False,
    )

    assert history
    assert history[0].role == MessageRole.SYSTEM
    assert "Memory Palace Context" in history[0].content
    assert "session-a" in history[0].content
    assert "Postgres" in history[0].content
    assert any(msg.content == "Current task is unrelated." for msg in history[1:])
