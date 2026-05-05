from __future__ import annotations

import json
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd

from .feature_engineering import compute_recency_score
from .types import PaperMetadata

# Column order matches build_feature_dataframe schema, with eval metadata prepended.
FEATURE_COLUMNS: List[str] = [
    "query_id",
    "target_ids",
    "paper_id",
    "semantic_score",
    "bib_score",
    "graph_score",
    "citation_count_log",
    "recency_score",
    "source_faiss",
    "source_graph",
]


def load_metadata(jsonl_path: str) -> Dict[str, PaperMetadata]:
    """Load candidates.jsonl into {paper_id: PaperMetadata}. Call once and reuse."""
    meta: Dict[str, PaperMetadata] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta[obj["paper_id"]] = PaperMetadata(
                year=obj.get("year"),
                citation_count=obj.get("citation_count"),
            )
    return meta


def build_fallback_dataframe(
    item: dict,
    metadata: Dict[str, PaperMetadata],
    current_year: int = 2026,
) -> pd.DataFrame:
    """Convert one offline_output.json item into a feature DataFrame.

    graph_score is fixed at 0.0 (no citation graph available).
    bib_score is taken directly from offline_output (pre-computed bibliographic coupling).
    """
    query_id = item["query_id"]
    target_ids = item["target_ids"]

    rows = []
    for cand in item["candidates"]:
        pid = cand["paper_id"]
        meta = metadata.get(pid)
        cc = meta.citation_count if (meta and meta.citation_count is not None) else 0
        year = meta.year if meta else None

        rows.append({
            "query_id": query_id,
            "target_ids": target_ids,
            "paper_id": pid,
            "semantic_score": float(cand["sim"]),
            "bib_score": float(cand["bib_score"]),
            "graph_score": 0.0,
            "citation_count_log": float(np.log1p(cc)),
            "recency_score": float(compute_recency_score(year, current_year)),
            "source_faiss": True,
            "source_graph": False,
        })

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS).astype({
            "semantic_score": float,
            "bib_score": float,
            "graph_score": float,
            "citation_count_log": float,
            "recency_score": float,
            "source_faiss": bool,
            "source_graph": bool,
        })

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    df["source_faiss"] = df["source_faiss"].astype(bool)
    df["source_graph"] = df["source_graph"].astype(bool)
    return df


def iter_fallback_dataframes(
    offline_path: str,
    metadata: Dict[str, PaperMetadata],
    current_year: int = 2026,
) -> Iterator[pd.DataFrame]:
    """Yield one feature DataFrame per query item in offline_output.json."""
    with open(offline_path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        yield build_fallback_dataframe(item, metadata, current_year)
