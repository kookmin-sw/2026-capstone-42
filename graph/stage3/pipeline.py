from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .feature_engineering import build_feature_dataframe
from .graph_expansion import expand_candidates
from .types import BibEmbeddings, CitationGraph, Embeddings, Metadata, RetrievalHit


def run_stage3(
    faiss_hits: List[RetrievalHit],
    graph: CitationGraph,
    embeddings: Embeddings,
    query_embedding: np.ndarray,
    bib_embeddings: BibEmbeddings,
    metadata: Metadata,
    expansion_ratio: float = 0.5,
    lambda_ref: float = 1.0,
    lambda_cite: float = 0.7,
    bib_top_k: int = 5,
    current_year: int = 2026,
) -> pd.DataFrame:
    if not faiss_hits:
        return build_feature_dataframe(
            all_paper_ids=[],
            graph_scores={},
            faiss_id_set=set(),
            graph_id_set=set(),
            embeddings=embeddings,
            query_embedding=query_embedding,
            bib_embeddings=bib_embeddings,
            metadata=metadata,
            faiss_score_map={},
            bib_top_k=bib_top_k,
            current_year=current_year,
        )

    faiss_score_map = {hit.paper_id: hit.faiss_score for hit in faiss_hits}

    all_paper_ids, graph_scores, faiss_id_set, graph_id_set = expand_candidates(
        faiss_hits=faiss_hits,
        graph=graph,
        ratio=expansion_ratio,
        lambda_ref=lambda_ref,
        lambda_cite=lambda_cite,
    )

    return build_feature_dataframe(
        all_paper_ids=all_paper_ids,
        graph_scores=graph_scores,
        faiss_id_set=faiss_id_set,
        graph_id_set=graph_id_set,
        embeddings=embeddings,
        query_embedding=query_embedding,
        bib_embeddings=bib_embeddings,
        metadata=metadata,
        faiss_score_map=faiss_score_map,
        bib_top_k=bib_top_k,
        current_year=current_year,
    )
