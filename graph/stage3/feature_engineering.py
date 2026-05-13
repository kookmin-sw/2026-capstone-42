from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .types import BibEmbeddings, Embeddings, Metadata


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_semantic_score(
    paper_id: str,
    faiss_id_set: Set[str],
    faiss_score_map: Dict[str, float],
    embeddings: Embeddings,
    query_embedding: np.ndarray,
) -> float:
    if paper_id in faiss_id_set:
        return faiss_score_map.get(paper_id, 0.0)
    if paper_id not in embeddings:
        return 0.0
    return cosine_similarity(query_embedding, embeddings[paper_id])


def compute_bib_score(
    paper_id: str,
    embeddings: Embeddings,
    bib_embeddings: BibEmbeddings,
    bib_top_k: int = 5,
) -> float:
    if not bib_embeddings or paper_id not in embeddings:
        return 0.0
    paper_emb = embeddings[paper_id]
    sims = [cosine_similarity(paper_emb, bib_emb) for bib_emb in bib_embeddings]
    sims.sort(reverse=True)
    top_k = sims[:bib_top_k]
    return float(np.mean(top_k))


def compute_recency_score(
    year: int | None,
    current_year: int = 2026,
) -> float:
    if year is None:
        return 0.0
    return 1.0 / (1.0 + max(0, current_year - year))


def build_feature_dataframe(
    all_paper_ids: List[str],
    graph_scores: Dict[str, float],
    faiss_id_set: Set[str],
    graph_id_set: Set[str],
    embeddings: Embeddings,
    query_embedding: np.ndarray,
    bib_embeddings: BibEmbeddings,
    metadata: Metadata,
    faiss_score_map: Dict[str, float],
    bib_top_k: int = 5,
    current_year: int = 2026,
) -> pd.DataFrame:
    rows = []
    for paper_id in all_paper_ids:
        meta = metadata.get(paper_id)

        semantic = compute_semantic_score(
            paper_id, faiss_id_set, faiss_score_map, embeddings, query_embedding
        )
        bib = compute_bib_score(paper_id, embeddings, bib_embeddings, bib_top_k)
        graph = graph_scores.get(paper_id, 0.0)
        citation_log = np.log1p(meta.citation_count if meta and meta.citation_count is not None else 0)
        recency = compute_recency_score(meta.year if meta else None, current_year)

        rows.append({
            "paper_id": paper_id,
            "semantic_score": float(semantic),
            "bib_score": float(bib),
            "graph_score": float(graph),
            "citation_count_log": float(citation_log),
            "recency_score": float(recency),
            "source_faiss": paper_id in faiss_id_set,
            "source_graph": paper_id in graph_id_set,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "paper_id", "semantic_score", "bib_score", "graph_score",
            "citation_count_log", "recency_score", "source_faiss", "source_graph",
        ]).astype({
            "paper_id": object,
            "semantic_score": float,
            "bib_score": float,
            "graph_score": float,
            "citation_count_log": float,
            "recency_score": float,
            "source_faiss": bool,
            "source_graph": bool,
        })

    df = pd.DataFrame(rows)
    df["source_faiss"] = df["source_faiss"].astype(bool)
    df["source_graph"] = df["source_graph"].astype(bool)
    return df
