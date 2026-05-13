from __future__ import annotations

import numpy as np
import pytest

from stage3.types import PaperMetadata, RetrievalHit


class StubCitationGraph:
    """In-memory citation graph. {"p1": ["p2"]} means p1 cites p2."""

    def __init__(self, adjacency: dict[str, list[str]]):
        self._adj = adjacency
        self._reverse: dict[str, list[str]] = {}
        for src, targets in adjacency.items():
            for t in targets:
                self._reverse.setdefault(t, []).append(src)

    def references(self, paper_id: str) -> list[str]:
        return self._adj.get(paper_id, [])

    def cited_by(self, paper_id: str) -> list[str]:
        return self._reverse.get(paper_id, [])


@pytest.fixture
def minimal_hits():
    return [
        RetrievalHit(paper_id="s1", faiss_score=0.9, rank=1),
        RetrievalHit(paper_id="s2", faiss_score=0.7, rank=2),
    ]


@pytest.fixture
def minimal_graph():
    # s1 cites g1, g2 cites s1
    return StubCitationGraph({"s1": ["g1"], "g2": ["s1"]})


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def minimal_embeddings(rng):
    return {
        "s1": rng.standard_normal(768).astype(np.float32),
        "s2": rng.standard_normal(768).astype(np.float32),
        "g1": rng.standard_normal(768).astype(np.float32),
        "g2": rng.standard_normal(768).astype(np.float32),
    }


@pytest.fixture
def query_embedding(rng):
    return rng.standard_normal(768).astype(np.float32)


@pytest.fixture
def minimal_bib_embeddings(minimal_embeddings):
    return [minimal_embeddings["s1"], minimal_embeddings["s2"]]


@pytest.fixture
def minimal_metadata():
    return {
        "s1": PaperMetadata(year=2020, citation_count=50),
        "s2": PaperMetadata(year=2018, citation_count=10),
        "g1": PaperMetadata(year=2022, citation_count=100),
        "g2": PaperMetadata(year=2015, citation_count=5),
    }
