from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from src.agent.core import Agent


class PromptRagRoutingTests(unittest.TestCase):
    def test_prompt_retrieval_prefers_routed_repo_paths_for_code_queries(self) -> None:
        agent = Agent.__new__(Agent)
        agent.config = SimpleNamespace(
            prompt_rag_context_enabled=True,
            prompt_rag_context_max_results=2,
            prompt_rag_context_neighbor_results=2,
            prompt_rag_context_max_chars=1200,
            prompt_wiki_context_enabled=False,
        )
        agent.ctx = SimpleNamespace(retrieval_insights=[])
        agent._runtime_context_snapshot = lambda: {
            "active_repos": [{"id": "repo-1"}],
            "rag_docs": [],
        }
        agent._compact_preview = lambda text, limit: str(text or "")[:limit]

        calls: list[dict | None] = []

        def search_rag(_query, *, where=None, n_results=None, event_cb=None):
            del n_results, event_cb
            calls.append(where)
            if where == {"$and": [{"repo_id": "repo-1"}, {"repo_rel_path": "pkg/service.py"}]}:
                return [
                    {
                        "content": "def register_repo(name): return name.strip()",
                        "metadata": {"repo_id": "repo-1", "repo_rel_path": "pkg/service.py"},
                        "distance": 0.1,
                    }
                ]
            return []

        agent.memory = SimpleNamespace(search_rag=search_rag)

        state = SimpleNamespace(user_query="where is register_repo called")
        with (
            mock.patch(
                "src.agent.core.load_repo_index",
                return_value=[{"id": "repo-1", "name": "Demo Repo"}],
            ),
            mock.patch(
                "src.agent.core.query_repo_context",
                return_value={
                    "candidate_paths": ["pkg/service.py"],
                    "matched_files": [{"rel_path": "pkg/service.py", "score": 96}],
                    "matched_symbols": [
                        {"name": "register_repo", "rel_path": "pkg/service.py", "line": 1}
                    ],
                },
            ),
            mock.patch(
                "src.agent.core.related_repo_context",
                return_value={"related_files": [], "related_symbols": []},
            ),
        ):
            context = Agent._prompt_retrieval_context(agent, state)

        self.assertTrue(calls)
        self.assertEqual(
            calls[0],
            {"$and": [{"repo_id": "repo-1"}, {"repo_rel_path": "pkg/service.py"}]},
        )
        self.assertIn("Focused files in repo repo-1: pkg/service.py", context)
        self.assertEqual(agent.ctx.retrieval_insights[0]["strategy"], "graph_lexical")

    def test_prompt_retrieval_falls_back_to_structured_wiki_context_when_no_rag_sources(self) -> None:
        agent = Agent.__new__(Agent)
        agent.config = SimpleNamespace(
            prompt_rag_context_enabled=True,
            prompt_rag_context_max_results=2,
            prompt_rag_context_neighbor_results=2,
            prompt_rag_context_max_chars=1200,
            prompt_wiki_context_enabled=True,
            prompt_wiki_context_max_results=3,
            prompt_wiki_context_max_chars=900,
        )
        agent.ctx = SimpleNamespace(retrieval_insights=[])
        agent._runtime_context_snapshot = lambda: {
            "active_repos": [],
            "rag_docs": [],
        }

        state = SimpleNamespace(user_query="how do outputs get filed")
        with mock.patch(
            "src.agent.core.build_wiki_context",
            return_value="Use this local wiki workspace if it helps answer the request:\nRelevant wiki pages:\n- wiki/articles/outputs-filed-outputs-md.md — Filed outputs are stored in wiki/outputs.",
        ) as mocked_wiki, mock.patch("src.agent.core.load_repo_index", return_value=[]):
            context = Agent._prompt_retrieval_context(agent, state)

        self.assertIn("Relevant wiki pages:", context)
        mocked_wiki.assert_called_once()


if __name__ == "__main__":
    unittest.main()
