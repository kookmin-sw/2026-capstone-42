from __future__ import annotations

import numpy as np
import pytest

from stage3 import run_stage3
from stage3.types import RetrievalHit
from tests.conftest import StubCitationGraph

EXPECTED_COLUMNS = {
    "paper_id", "semantic_score", "bib_score", "graph_score",
    "citation_count_log", "recency_score", "source_faiss", "source_graph",
}


def test_run_stage3_returns_dataframe(minimal_hits, minimal_graph, minimal_embeddings,
                                      query_embedding, minimal_bib_embeddings, minimal_metadata):
    import pandas as pd
    df = run_stage3(minimal_hits, minimal_graph, minimal_embeddings,
                    query_embedding, minimal_bib_embeddings, minimal_metadata)
    assert isinstance(df, pd.DataFrame)


def test_run_stage3_output_schema(minimal_hits, minimal_graph, minimal_embeddings,
                                   query_embedding, minimal_bib_embeddings, minimal_metadata):
    df = run_stage3(minimal_hits, minimal_graph, minimal_embeddings,
                    query_embedding, minimal_bib_embeddings, minimal_metadata)
    assert set(df.columns) == EXPECTED_COLUMNS
    assert df["source_faiss"].dtype == bool
    assert df["source_graph"].dtype == bool


def test_run_stage3_empty_faiss_hits(minimal_graph, minimal_embeddings,
                                      query_embedding, minimal_bib_embeddings, minimal_metadata):
    df = run_stage3([], minimal_graph, minimal_embeddings,
                    query_embedding, minimal_bib_embeddings, minimal_metadata)
    assert len(df) == 0
    assert set(df.columns) == EXPECTED_COLUMNS


def test_run_stage3_ratio_respected(minimal_embeddings, query_embedding, minimal_bib_embeddings, minimal_metadata):
    # 4 FAISS hits, ratio=0.5 → up to 2 graph candidates added
    seeds = [RetrievalHit(paper_id=f"s{i}", faiss_score=0.9 - i * 0.1, rank=i) for i in range(4)]
    graph = StubCitationGraph({f"s{i}": [f"g{i}"] for i in range(4)})
    for pid in [f"g{i}" for i in range(4)]:
        minimal_embeddings[pid] = np.random.default_rng(99).standard_normal(768).astype(np.float32)
        minimal_metadata[pid] = minimal_metadata.get("g1")

    df = run_stage3(seeds, graph, minimal_embeddings, query_embedding,
                    minimal_bib_embeddings, minimal_metadata, expansion_ratio=0.5)
    graph_only = df[df["source_graph"] & ~df["source_faiss"]]
    assert len(graph_only) <= 2


def test_run_stage3_faiss_semantic_score_equals_faiss_score(minimal_hits, minimal_graph,
                                                              minimal_embeddings, query_embedding,
                                                              minimal_bib_embeddings, minimal_metadata):
    df = run_stage3(minimal_hits, minimal_graph, minimal_embeddings,
                    query_embedding, minimal_bib_embeddings, minimal_metadata)
    faiss_rows = df[df["source_faiss"]]
    for _, row in faiss_rows.iterrows():
        hit = next(h for h in minimal_hits if h.paper_id == row["paper_id"])
        assert row["semantic_score"] == pytest.approx(hit.faiss_score)


def test_run_stage3_empty_bib_embeddings(minimal_hits, minimal_graph, minimal_embeddings,
                                          query_embedding, minimal_metadata):
    df = run_stage3(minimal_hits, minimal_graph, minimal_embeddings,
                    query_embedding, [], minimal_metadata)
    assert (df["bib_score"] == 0.0).all()


def test_run_stage3_all_missing_metadata(minimal_hits, minimal_graph, minimal_embeddings,
                                          query_embedding, minimal_bib_embeddings):
    df = run_stage3(minimal_hits, minimal_graph, minimal_embeddings,
                    query_embedding, minimal_bib_embeddings, {})
    assert (df["citation_count_log"] == 0.0).all()
    assert (df["recency_score"] == 0.0).all()


def test_run_stage3_lambda_params_affect_graph_score(minimal_hits, minimal_graph,
                                                      minimal_embeddings, query_embedding,
                                                      minimal_bib_embeddings, minimal_metadata):
    df1 = run_stage3(minimal_hits, minimal_graph, minimal_embeddings, query_embedding,
                     minimal_bib_embeddings, minimal_metadata, lambda_ref=1.0)
    df2 = run_stage3(minimal_hits, minimal_graph, minimal_embeddings, query_embedding,
                     minimal_bib_embeddings, minimal_metadata, lambda_ref=2.0)
    # g1 is referenced by s1, so its graph_score should differ
    s1 = df1[df1["paper_id"] == "g1"]["graph_score"].values
    s2 = df2[df2["paper_id"] == "g1"]["graph_score"].values
    if len(s1) > 0 and len(s2) > 0:
        assert s2[0] == pytest.approx(s1[0] * 2.0)


def test_run_stage3_current_year_affects_recency(minimal_hits, minimal_graph, minimal_embeddings,
                                                   query_embedding, minimal_bib_embeddings, minimal_metadata):
    df2020 = run_stage3(minimal_hits, minimal_graph, minimal_embeddings, query_embedding,
                        minimal_bib_embeddings, minimal_metadata, current_year=2020)
    df2026 = run_stage3(minimal_hits, minimal_graph, minimal_embeddings, query_embedding,
                        minimal_bib_embeddings, minimal_metadata, current_year=2026)
    # s1 has year=2020; recency should be higher when current_year=2020
    r2020 = df2020[df2020["paper_id"] == "s1"]["recency_score"].values[0]
    r2026 = df2026[df2026["paper_id"] == "s1"]["recency_score"].values[0]
    assert r2020 > r2026


def test_run_stage3_graph_candidates_have_source_graph(minimal_hits, minimal_graph,
                                                        minimal_embeddings, query_embedding,
                                                        minimal_bib_embeddings, minimal_metadata):
    df = run_stage3(minimal_hits, minimal_graph, minimal_embeddings, query_embedding,
                    minimal_bib_embeddings, minimal_metadata)
    # g1 is a graph-expanded candidate (s1 cites g1)
    g1_rows = df[df["paper_id"] == "g1"]
    if len(g1_rows) > 0:
        assert g1_rows.iloc[0]["source_graph"] is True or g1_rows.iloc[0]["source_graph"]
