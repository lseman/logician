from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


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


ATOM = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/math.GT/0309136v2</id>
    <title>  Test   Paper </title>
    <summary>Abstract text.</summary>
    <published>2003-09-10T00:00:00Z</published>
    <author><name>Alice</name></author>
    <link rel="alternate" href="https://arxiv.org/abs/math.GT/0309136v2"/>
    <link rel="related" type="application/pdf"
          href="https://arxiv.org/pdf/math.GT/0309136v2"/>
    <category term="math.GT"/>
    <arxiv:primary_category term="math.GT"/>
    <arxiv:doi>10.1000/example</arxiv:doi>
  </entry>
</feed>"""


class FakeClient:
    def __init__(self, content: bytes = ATOM):
        self.content = content
        self.calls = []

    def get(self, url, params):
        self.calls.append((url, params))
        return SimpleNamespace(
            status_code=200,
            content=self.content,
            raise_for_status=lambda: None,
        )


def test_arxiv_search_uses_official_query_contract(monkeypatch):
    module = _load_arxiv_skill_module()
    monkeypatch.setattr(module, "_last_request_time", 0.0)
    source = module.ArxivSource()
    source._client = FakeClient()

    results = source.search("quantum computing", limit=1)

    params = source._client.calls[0][1]
    assert params["search_query"] == "all:quantum AND all:computing"
    assert params["sortBy"] == "submittedDate"
    assert params["sortOrder"] == "descending"
    assert results[0].arxiv_id == "math.gt/0309136"
    assert results[0].doi == "10.1000/example"
    assert results[0].title == "Test Paper"


def test_arxiv_preserves_explicit_query_syntax():
    module = _load_arxiv_skill_module()
    source = module.ArxivSource()
    assert (
        source._build_search_query('ti:"attention" AND au:vaswani')
        == 'ti:"attention" AND au:vaswani'
    )


def test_arxiv_get_uses_id_list(monkeypatch):
    module = _load_arxiv_skill_module()
    monkeypatch.setattr(module, "_last_request_time", 0.0)
    source = module.ArxivSource()
    source._client = FakeClient()

    paper = source.get_by_id("arXiv:math.GT/0309136")

    assert paper is not None
    params = source._client.calls[0][1]
    assert params["id_list"] == "math.gt/0309136"
    assert "search_query" not in params


def test_arxiv_atom_error_is_not_returned_as_a_paper(monkeypatch):
    module = _load_arxiv_skill_module()
    monkeypatch.setattr(module, "_last_request_time", 0.0)
    error_feed = b"""<feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>http://arxiv.org/api/errors#bad</id><title>Error</title>
      <summary>bad query</summary></entry></feed>"""
    source = module.ArxivSource()
    source._client = FakeClient(error_feed)

    with pytest.raises(ValueError, match="bad query"):
        source.search("test")
