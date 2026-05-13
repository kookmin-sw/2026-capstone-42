from __future__ import annotations

import json
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from .feature_engineering import compute_recency_score
from .graph_expansion import compute_graph_scores
from .types import CitationGraph, PaperMetadata, RetrievalHit

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


def _normalize(vals: List[float]) -> List[float]:
    """Min-max normalization. All-equal inputs → all zeros."""
    arr = np.array(vals, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return [0.0] * len(vals)
    return ((arr - mn) / (mx - mn)).tolist()


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
    graph: Optional[CitationGraph] = None,
) -> pd.DataFrame:
    """Convert one offline_output.json item into a feature DataFrame.

    graph=None  → graph_score fixed at 0.0 (original fallback behaviour).
    graph given → graph_score from compute_graph_scores(), normalised within query pool.
    bib_score is taken directly from offline_output (pre-computed bibliographic coupling).
    """
    query_id = item["query_id"]
    target_ids = item["target_ids"]
    candidates = item["candidates"]

    # --- graph_score 계산 (graph 있을 때만) ---
    graph_score_map: Dict[str, float] = {}
    if graph is not None and candidates:
        seeds = [
            RetrievalHit(paper_id=c["paper_id"], faiss_score=float(c["sim"]), rank=i)
            for i, c in enumerate(candidates)
        ]
        raw_scores = compute_graph_scores(seeds, graph)
        pids = [c["paper_id"] for c in candidates]
        raw_vals = [raw_scores.get(pid, 0.0) for pid in pids]
        normed = _normalize(raw_vals)
        graph_score_map = dict(zip(pids, normed))

    rows = []
    for cand in candidates:
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
            "graph_score": graph_score_map.get(pid, 0.0),
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
    graph: Optional[CitationGraph] = None,
) -> Iterator[pd.DataFrame]:
    """Yield one feature DataFrame per query item in offline_output.json."""
    with open(offline_path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        yield build_fallback_dataframe(item, metadata, current_year, graph=graph)
