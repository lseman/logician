from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_arxiv_skill_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "academic"
        / "arxiv"
        / "scripts"
        / "arxiv.py"
    )
    spec = importlib.util.spec_from_file_location("arxiv_skill", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["arxiv_skill"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_arxiv_result():
    return SimpleNamespace(
        title="Test Paper",
        published=SimpleNamespace(year=2025),
        entry_id="http://arxiv.org/abs/1234.5678",
        authors=[SimpleNamespace(name="Alice")],
        pdf_url="https://arxiv.org/pdf/1234.5678.pdf",
        summary="Abstract text.",
    )


def test_arxiv_search_falls_back_to_query_api(monkeypatch):
    module = _load_arxiv_skill_module()

    fake_arxiv = SimpleNamespace(
        query=lambda **kwargs: [_make_arxiv_result()],
    )

    monkeypatch.setattr(module, "arxiv", fake_arxiv)
    monkeypatch.setattr(module, "HAS_ARXIV", True)

    results = module.ArvixSource().search("test query", limit=1)

    assert len(results) == 1
    assert results[0].title == "Test Paper"
    assert results[0].arxiv_id == "1234.5678"
    assert results[0].venue == "arXiv"
    assert results[0].source == "arxiv"
