from __future__ import annotations

import pytest

from stage3.graph_expansion import (
    compute_graph_scores,
    expand_candidates,
    select_graph_candidates,
)
from stage3.types import RetrievalHit
from tests.conftest import StubCitationGraph


def test_compute_graph_scores_ref_direction():
    # seed s1 cites g1 → graph_score[g1] == lambda_ref * s1.faiss_score
    seeds = [RetrievalHit(paper_id="s1", faiss_score=0.9, rank=1)]
    graph = StubCitationGraph({"s1": ["g1"]})
    scores = compute_graph_scores(seeds, graph, lambda_ref=1.0, lambda_cite=0.7)
    assert scores["g1"] == pytest.approx(1.0 * 0.9)


def test_compute_graph_scores_cite_direction():
    # g2 cites seed s1 → graph_score[g2] == lambda_cite * s1.faiss_score
    seeds = [RetrievalHit(paper_id="s1", faiss_score=0.9, rank=1)]
    graph = StubCitationGraph({"g2": ["s1"]})
    scores = compute_graph_scores(seeds, graph, lambda_ref=1.0, lambda_cite=0.7)
    assert scores["g2"] == pytest.approx(0.7 * 0.9)


def test_compute_graph_scores_accumulation():
    # g1 is neighbor of both s1 and s2
    seeds = [
        RetrievalHit(paper_id="s1", faiss_score=0.9, rank=1),
        RetrievalHit(paper_id="s2", faiss_score=0.7, rank=2),
    ]
    graph = StubCitationGraph({"s1": ["g1"], "s2": ["g1"]})
    scores = compute_graph_scores(seeds, graph)
    expected = 1.0 * 0.9 + 1.0 * 0.7
    assert scores["g1"] == pytest.approx(expected)


def test_compute_graph_scores_faiss_seed_as_neighbor():
    # s1 is referenced by s2 (s2 -> s1), so s1 also gets a graph_score
    seeds = [
        RetrievalHit(paper_id="s1", faiss_score=0.9, rank=1),
        RetrievalHit(paper_id="s2", faiss_score=0.7, rank=2),
    ]
    graph = StubCitationGraph({"s2": ["s1"]})
    scores = compute_graph_scores(seeds, graph)
    assert "s1" in scores
    assert scores["s1"] == pytest.approx(1.0 * 0.7)


def test_select_graph_candidates_excludes_faiss():
    graph_scores = {"s1": 0.9, "g1": 0.8, "g2": 0.6}
    faiss_ids = {"s1"}
    result = select_graph_candidates(graph_scores, faiss_ids, n_graph=5)
    ids = [r[0] for r in result]
    assert "s1" not in ids
    assert "g1" in ids
    assert "g2" in ids


def test_select_graph_candidates_top_n():
    graph_scores = {"g1": 0.9, "g2": 0.8, "g3": 0.5}
    result = select_graph_candidates(graph_scores, faiss_paper_ids=set(), n_graph=2)
    assert len(result) == 2
    assert result[0][0] == "g1"
    assert result[1][0] == "g2"


def test_select_graph_candidates_fewer_than_n():
    # Only 2 neighbors available, n_graph=10 → return all 2
    graph_scores = {"g1": 0.9, "g2": 0.5}
    result = select_graph_candidates(graph_scores, faiss_paper_ids=set(), n_graph=10)
    assert len(result) == 2


def test_expand_candidates_ratio():
    # 4 FAISS hits → int(4 * 0.5) = 2 graph candidates
    seeds = [RetrievalHit(paper_id=f"s{i}", faiss_score=0.9 - i * 0.1, rank=i) for i in range(4)]
    graph = StubCitationGraph({f"s{i}": [f"g{i}"] for i in range(4)})
    all_ids, _, faiss_set, graph_set = expand_candidates(seeds, graph, ratio=0.5)
    assert len(graph_set) <= 2


def test_expand_candidates_deduplication():
    # g1 appears both as FAISS seed reference and in faiss_hits
    seeds = [RetrievalHit(paper_id="s1", faiss_score=0.9, rank=1)]
    graph = StubCitationGraph({"s1": ["g1"]})
    all_ids, _, _, _ = expand_candidates(seeds, graph)
    assert len(all_ids) == len(set(all_ids))


def test_expand_candidates_source_flags_overlap():
    # g1 also happens to be a FAISS candidate
    seeds = [
        RetrievalHit(paper_id="s1", faiss_score=0.9, rank=1),
        RetrievalHit(paper_id="g1", faiss_score=0.6, rank=2),  # g1 is FAISS too
    ]
    graph = StubCitationGraph({"s1": ["g1", "g2"]})
    all_ids, graph_scores, faiss_set, graph_set = expand_candidates(seeds, graph)
    # g1 is in faiss_set (it was a hit), g2 is only in graph_set
    assert "g1" in faiss_set
    assert "g2" in graph_set


def test_expand_candidates_empty_graph():
    seeds = [RetrievalHit(paper_id="s1", faiss_score=0.9, rank=1)]
    graph = StubCitationGraph({})
    all_ids, graph_scores, faiss_set, graph_set = expand_candidates(seeds, graph)
    assert set(all_ids) == {"s1"}
    assert len(graph_set) == 0


def test_expand_candidates_empty_faiss_hits():
    graph = StubCitationGraph({"s1": ["g1"]})
    all_ids, graph_scores, faiss_set, graph_set = expand_candidates([], graph)
    assert all_ids == []
    assert graph_scores == {}
    assert faiss_set == set()
    assert graph_set == set()
