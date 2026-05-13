from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .types import CitationGraph, RetrievalHit


def compute_graph_scores(
    seeds: List[RetrievalHit],
    graph: CitationGraph,
    lambda_ref: float = 1.0,
    lambda_cite: float = 0.7,
) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for seed in seeds:
        # seed -> candidate (seed가 candidate를 인용)
        for candidate in graph.references(seed.paper_id):
            scores[candidate] += lambda_ref * seed.faiss_score
        # candidate -> seed (candidate가 seed를 인용)
        for candidate in graph.cited_by(seed.paper_id):
            scores[candidate] += lambda_cite * seed.faiss_score
    return dict(scores)


def select_graph_candidates(
    graph_scores: Dict[str, float],
    faiss_paper_ids: Set[str],
    n_graph: int,
) -> List[Tuple[str, float]]:
    candidates = [
        (paper_id, score)
        for paper_id, score in graph_scores.items()
        if paper_id not in faiss_paper_ids
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n_graph]


def expand_candidates(
    faiss_hits: List[RetrievalHit],
    graph: CitationGraph,
    ratio: float = 0.5,
    lambda_ref: float = 1.0,
    lambda_cite: float = 0.7,
) -> Tuple[List[str], Dict[str, float], Set[str], Set[str]]:
    if not faiss_hits:
        return [], {}, set(), set()

    faiss_id_set: Set[str] = {hit.paper_id for hit in faiss_hits}
    n_graph = int(len(faiss_hits) * ratio)

    graph_scores = compute_graph_scores(faiss_hits, graph, lambda_ref, lambda_cite)

    # FAISS 후보에도 graph_score 계산 반영 (이미 graph_scores에 포함될 수 있음)
    # graph_scores에 없는 FAISS 후보는 0.0으로 처리 (pipeline에서 defaultdict 대신 get 사용)

    graph_candidates = select_graph_candidates(graph_scores, faiss_id_set, n_graph)
    graph_id_set: Set[str] = {paper_id for paper_id, _ in graph_candidates}

    # 중복 제거: FAISS 순서 유지 후 graph-only 추가
    all_paper_ids: List[str] = list(faiss_id_set)
    for paper_id in graph_id_set:
        if paper_id not in faiss_id_set:
            all_paper_ids.append(paper_id)

    return all_paper_ids, graph_scores, faiss_id_set, graph_id_set
