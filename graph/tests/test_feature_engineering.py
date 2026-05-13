from __future__ import annotations

import numpy as np
import pytest

from stage3.feature_engineering import (
    build_feature_dataframe,
    compute_bib_score,
    compute_recency_score,
    compute_semantic_score,
    cosine_similarity,
)
from stage3.types import PaperMetadata


# --- cosine_similarity ---

def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    assert cosine_similarity(a, b) == 0.0
    assert cosine_similarity(b, a) == 0.0


# --- compute_semantic_score ---

def test_semantic_score_faiss_candidate_uses_faiss_score():
    emb = {"p1": np.array([1.0, 0.0])}
    query = np.array([0.0, 1.0])
    score = compute_semantic_score(
        "p1",
        faiss_id_set={"p1"},
        faiss_score_map={"p1": 0.88},
        embeddings=emb,
        query_embedding=query,
    )
    assert score == pytest.approx(0.88)


def test_semantic_score_graph_candidate_uses_cosine():
    v = np.array([1.0, 0.0])
    emb = {"g1": v}
    query = np.array([1.0, 0.0])
    score = compute_semantic_score(
        "g1",
        faiss_id_set=set(),
        faiss_score_map={},
        embeddings=emb,
        query_embedding=query,
    )
    assert score == pytest.approx(1.0)


def test_semantic_score_missing_embedding_returns_zero():
    score = compute_semantic_score(
        "unknown",
        faiss_id_set=set(),
        faiss_score_map={},
        embeddings={},
        query_embedding=np.array([1.0, 0.0]),
    )
    assert score == 0.0


# --- compute_bib_score ---

def test_bib_score_is_top_k_mean_not_max():
    paper_emb = np.array([1.0, 0.0])
    # similarities: 1.0, 0.0, 0.5 → top-2 mean = (1.0 + 0.5) / 2 = 0.75
    bib = [
        np.array([1.0, 0.0]),   # sim=1.0
        np.array([0.0, 1.0]),   # sim=0.0
        np.array([1.0, 1.0]) / np.sqrt(2),  # sim≈0.707
    ]
    score = compute_bib_score("p1", {"p1": paper_emb}, bib, bib_top_k=2)
    expected = (1.0 + cosine_similarity(paper_emb, bib[2])) / 2
    assert score == pytest.approx(expected)


def test_bib_score_empty_bib_returns_zero():
    score = compute_bib_score("p1", {"p1": np.array([1.0, 0.0])}, [], bib_top_k=5)
    assert score == 0.0


def test_bib_score_missing_paper_returns_zero():
    score = compute_bib_score("unknown", {}, [np.array([1.0, 0.0])], bib_top_k=5)
    assert score == 0.0


def test_bib_score_fewer_than_top_k():
    # Only 2 bibs but top_k=5 → average all 2
    paper_emb = np.array([1.0, 0.0])
    bib = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    score = compute_bib_score("p1", {"p1": paper_emb}, bib, bib_top_k=5)
    expected = (1.0 + 0.0) / 2
    assert score == pytest.approx(expected)


# --- compute_recency_score ---

def test_recency_score_formula():
    # 1 / (1 + max(0, 2026 - 2020)) = 1/7
    assert compute_recency_score(2020, current_year=2026) == pytest.approx(1 / 7)


def test_recency_score_current_year():
    # year == current_year → 1 / (1 + 0) = 1.0
    assert compute_recency_score(2026, current_year=2026) == pytest.approx(1.0)


def test_recency_score_future_year():
    # year > current_year → max(0, ...) = 0 → 1.0
    assert compute_recency_score(2030, current_year=2026) == pytest.approx(1.0)


def test_recency_score_none_year():
    assert compute_recency_score(None, current_year=2026) == 0.0


# --- build_feature_dataframe ---

def test_build_feature_dataframe_columns(minimal_embeddings, query_embedding, minimal_bib_embeddings, minimal_metadata):
    from stage3.feature_engineering import build_feature_dataframe
    df = build_feature_dataframe(
        all_paper_ids=["s1", "g1"],
        graph_scores={"s1": 0.5, "g1": 0.3},
        faiss_id_set={"s1"},
        graph_id_set={"g1"},
        embeddings=minimal_embeddings,
        query_embedding=query_embedding,
        bib_embeddings=minimal_bib_embeddings,
        metadata=minimal_metadata,
        faiss_score_map={"s1": 0.9},
    )
    expected_cols = {"paper_id", "semantic_score", "bib_score", "graph_score",
                     "citation_count_log", "recency_score", "source_faiss", "source_graph"}
    assert set(df.columns) == expected_cols


def test_build_feature_dataframe_source_flags_dtype(minimal_embeddings, query_embedding, minimal_bib_embeddings, minimal_metadata):
    df = build_feature_dataframe(
        all_paper_ids=["s1"],
        graph_scores={},
        faiss_id_set={"s1"},
        graph_id_set=set(),
        embeddings=minimal_embeddings,
        query_embedding=query_embedding,
        bib_embeddings=minimal_bib_embeddings,
        metadata=minimal_metadata,
        faiss_score_map={"s1": 0.9},
    )
    assert df["source_faiss"].dtype == bool
    assert df["source_graph"].dtype == bool


def test_build_feature_dataframe_no_duplicates(minimal_embeddings, query_embedding, minimal_bib_embeddings, minimal_metadata):
    df = build_feature_dataframe(
        all_paper_ids=["s1", "g1"],
        graph_scores={},
        faiss_id_set={"s1"},
        graph_id_set={"g1"},
        embeddings=minimal_embeddings,
        query_embedding=query_embedding,
        bib_embeddings=minimal_bib_embeddings,
        metadata=minimal_metadata,
        faiss_score_map={"s1": 0.9},
    )
    assert df["paper_id"].nunique() == len(df)


def test_build_feature_dataframe_empty_input(query_embedding):
    df = build_feature_dataframe(
        all_paper_ids=[],
        graph_scores={},
        faiss_id_set=set(),
        graph_id_set=set(),
        embeddings={},
        query_embedding=query_embedding,
        bib_embeddings=[],
        metadata={},
        faiss_score_map={},
    )
    assert len(df) == 0
    assert "paper_id" in df.columns


def test_build_feature_dataframe_missing_metadata(minimal_embeddings, query_embedding, minimal_bib_embeddings):
    df = build_feature_dataframe(
        all_paper_ids=["s1"],
        graph_scores={},
        faiss_id_set={"s1"},
        graph_id_set=set(),
        embeddings=minimal_embeddings,
        query_embedding=query_embedding,
        bib_embeddings=minimal_bib_embeddings,
        metadata={},  # no metadata
        faiss_score_map={"s1": 0.9},
    )
    assert df.loc[0, "citation_count_log"] == pytest.approx(0.0)
    assert df.loc[0, "recency_score"] == pytest.approx(0.0)
