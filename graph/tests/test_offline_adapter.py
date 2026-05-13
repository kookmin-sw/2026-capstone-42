from __future__ import annotations

import json
import math

import pytest

from stage3.offline_adapter import (
    FEATURE_COLUMNS,
    build_fallback_dataframe,
    iter_fallback_dataframes,
    load_metadata,
)

# ---------------------------------------------------------------------------
# Dummy fixtures — no real data files used
# ---------------------------------------------------------------------------

_DUMMY_JSONL = [
    {"paper_id": "s2_aaa", "year": 2020, "citation_count": 10,
     "title": "A", "abstract": "", "fields": [], "all_references": []},
    {"paper_id": "s2_bbb", "year": 2018, "citation_count": 0,
     "title": "B", "abstract": "", "fields": [], "all_references": []},
    # s2_ccc intentionally absent to test missing-metadata fallback
]

_DUMMY_ITEM = {
    "query_id": "arxiv_1234.56789_00",
    "target_ids": ["s2_aaa"],
    "context": "some context text",
    "candidates": [
        {"paper_id": "s2_aaa", "sim": 0.9, "bib_score": 0.8},
        {"paper_id": "s2_bbb", "sim": 0.7, "bib_score": 0.5},
        {"paper_id": "s2_ccc", "sim": 0.6, "bib_score": 0.3},
    ],
}


@pytest.fixture
def metadata_file(tmp_path):
    p = tmp_path / "candidates.jsonl"
    p.write_text(
        "\n".join(json.dumps(r) for r in _DUMMY_JSONL),
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture
def offline_file(tmp_path):
    p = tmp_path / "offline_output.json"
    p.write_text(json.dumps([_DUMMY_ITEM, _DUMMY_ITEM]), encoding="utf-8")
    return str(p)


@pytest.fixture
def full_metadata(metadata_file):
    return load_metadata(metadata_file)


# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------

def test_load_metadata_keys(full_metadata):
    assert "s2_aaa" in full_metadata
    assert "s2_bbb" in full_metadata


def test_load_metadata_values(full_metadata):
    assert full_metadata["s2_aaa"].year == 2020
    assert full_metadata["s2_aaa"].citation_count == 10
    assert full_metadata["s2_bbb"].year == 2018
    assert full_metadata["s2_bbb"].citation_count == 0


# ---------------------------------------------------------------------------
# build_fallback_dataframe — schema
# ---------------------------------------------------------------------------

def test_columns_match_feature_columns():
    df = build_fallback_dataframe(_DUMMY_ITEM, {})
    assert list(df.columns) == FEATURE_COLUMNS


def test_row_count_equals_candidate_count():
    df = build_fallback_dataframe(_DUMMY_ITEM, {})
    assert len(df) == len(_DUMMY_ITEM["candidates"])


def test_empty_candidates_returns_empty_df_with_correct_columns():
    item = {**_DUMMY_ITEM, "candidates": []}
    df = build_fallback_dataframe(item, {})
    assert len(df) == 0
    assert list(df.columns) == FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# build_fallback_dataframe — scores from offline_output
# ---------------------------------------------------------------------------

def test_semantic_score_equals_sim():
    df = build_fallback_dataframe(_DUMMY_ITEM, {})
    assert df.loc[0, "semantic_score"] == pytest.approx(0.9)
    assert df.loc[1, "semantic_score"] == pytest.approx(0.7)


def test_bib_score_equals_offline_value():
    df = build_fallback_dataframe(_DUMMY_ITEM, {})
    assert df.loc[0, "bib_score"] == pytest.approx(0.8)
    assert df.loc[2, "bib_score"] == pytest.approx(0.3)


def test_graph_score_is_zero():
    df = build_fallback_dataframe(_DUMMY_ITEM, {})
    assert (df["graph_score"] == 0.0).all()


# ---------------------------------------------------------------------------
# build_fallback_dataframe — metadata-derived features
# ---------------------------------------------------------------------------

def test_citation_count_log_with_metadata(full_metadata):
    df = build_fallback_dataframe(_DUMMY_ITEM, full_metadata)
    row_aaa = df.loc[df["paper_id"] == "s2_aaa", "citation_count_log"].iloc[0]
    row_bbb = df.loc[df["paper_id"] == "s2_bbb", "citation_count_log"].iloc[0]
    row_ccc = df.loc[df["paper_id"] == "s2_ccc", "citation_count_log"].iloc[0]
    assert row_aaa == pytest.approx(math.log1p(10))
    assert row_bbb == pytest.approx(0.0)   # log1p(0)
    assert row_ccc == pytest.approx(0.0)   # missing metadata


def test_recency_score_with_metadata(full_metadata):
    df = build_fallback_dataframe(_DUMMY_ITEM, full_metadata, current_year=2026)
    row_aaa = df.loc[df["paper_id"] == "s2_aaa", "recency_score"].iloc[0]
    row_ccc = df.loc[df["paper_id"] == "s2_ccc", "recency_score"].iloc[0]
    assert row_aaa == pytest.approx(1 / 7)   # 1 / (1 + 2026 - 2020)
    assert row_ccc == pytest.approx(0.0)      # missing metadata


def test_missing_metadata_yields_zero_fallback():
    df = build_fallback_dataframe(_DUMMY_ITEM, {})
    assert (df["citation_count_log"] == 0.0).all()
    assert (df["recency_score"] == 0.0).all()


# ---------------------------------------------------------------------------
# build_fallback_dataframe — dtypes and flags
# ---------------------------------------------------------------------------

def test_source_faiss_is_all_true():
    df = build_fallback_dataframe(_DUMMY_ITEM, {})
    assert df["source_faiss"].dtype == bool
    assert df["source_faiss"].all()


def test_source_graph_is_all_false():
    df = build_fallback_dataframe(_DUMMY_ITEM, {})
    assert df["source_graph"].dtype == bool
    assert not df["source_graph"].any()


def test_graph_score_dtype_is_float():
    df = build_fallback_dataframe(_DUMMY_ITEM, {})
    assert df["graph_score"].dtype == float


# ---------------------------------------------------------------------------
# build_fallback_dataframe — eval metadata columns
# ---------------------------------------------------------------------------

def test_query_id_propagated_to_all_rows():
    df = build_fallback_dataframe(_DUMMY_ITEM, {})
    assert (df["query_id"] == "arxiv_1234.56789_00").all()


def test_target_ids_propagated_to_all_rows():
    df = build_fallback_dataframe(_DUMMY_ITEM, {})
    for val in df["target_ids"]:
        assert val == ["s2_aaa"]


# ---------------------------------------------------------------------------
# iter_fallback_dataframes
# ---------------------------------------------------------------------------

def test_iter_yields_one_df_per_query(offline_file):
    dfs = list(iter_fallback_dataframes(offline_file, {}))
    assert len(dfs) == 2


def test_iter_each_df_has_correct_row_count(offline_file):
    for df in iter_fallback_dataframes(offline_file, {}):
        assert len(df) == len(_DUMMY_ITEM["candidates"])


def test_iter_each_df_has_correct_columns(offline_file):
    for df in iter_fallback_dataframes(offline_file, {}):
        assert list(df.columns) == FEATURE_COLUMNS


def test_iter_query_id_correct(offline_file):
    for df in iter_fallback_dataframes(offline_file, {}):
        assert df["query_id"].iloc[0] == "arxiv_1234.56789_00"
