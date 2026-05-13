"""
Stage3 graph feature 적용 평가 스크립트.

결과물:
  output_graph_stage3.json    — query별 ranked candidates (with graph_score + coverage 필드)
  eval_baseline_vs_graph.csv  — baseline vs graph_applied 평가지표 비교
  coverage_summary.csv        — reference coverage 통계

실행:
  python run_stage3_eval.py            # 전체 실행 (n_queries=None)
  python run_stage3_eval.py --smoke    # smoke test: 앞 10개 query만 실행
  python run_stage3_eval.py --n 1000  # 앞 1000개 query만 실행
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from stage3.graph_loader import CitationGraphPKL
from stage3.offline_adapter import build_fallback_dataframe

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR             = Path(__file__).resolve().parent.parent
OFFLINE_OUTPUT_PATH  = BASE_DIR / "data" / "(삭제x)offline_output.json"
CITATION_GRAPH_ZIP   = BASE_DIR / "data" / "citation_graph.zip"
OUTPUT_PATH          = BASE_DIR / "output_graph_stage3.json"
EVAL_PATH            = BASE_DIR / "eval_baseline_vs_graph.csv"
COVERAGE_PATH        = BASE_DIR / "coverage_summary.csv"

# ─── final_score 가중치 ───────────────────────────────────────────────────────
W_SEMANTIC = 0.5
W_BIB      = 0.3
W_GRAPH    = 0.2

K_LIST = [10, 50, 100]


# ─── Coverage 누적 카운터 ─────────────────────────────────────────────────────
@dataclass
class CoverageStats:
    total_candidates:  int = 0
    fwd_count:         int = 0   # has_forward=True
    bwd_count:         int = 0   # has_backward=True
    graph_nonzero:     int = 0   # graph_score > 0
    total_targets:     int = 0   # is_target=True
    target_fwd:        int = 0   # is_target AND has_forward
    target_bwd:        int = 0   # is_target AND has_backward
    top10_total:       int = 0   # rank <= 10
    top10_zero_graph:  int = 0   # rank <= 10 AND graph_score == 0


# ─── 평가 함수 ────────────────────────────────────────────────────────────────
def compute_metrics(ranked_ids: List[str], target_ids: List[str]) -> Dict[str, float]:
    targets = set(target_ids)
    metrics: Dict[str, float] = {}

    if not targets:
        for k in K_LIST:
            metrics[f"Recall@{k}"] = 0.0
        metrics["MRR"] = 0.0
        return metrics

    for k in K_LIST:
        hits = len(set(ranked_ids[:k]) & targets)
        metrics[f"Recall@{k}"] = hits / len(targets)

    mrr = 0.0
    for rank, pid in enumerate(ranked_ids, 1):
        if pid in targets:
            mrr = 1.0 / rank
            break
    metrics["MRR"] = mrr
    return metrics


# ─── 메인 실행 ────────────────────────────────────────────────────────────────
def run(n_queries: Optional[int] = None) -> None:
    print(f"[1/4] citation graph 로딩: {CITATION_GRAPH_ZIP}")
    graph = CitationGraphPKL(str(CITATION_GRAPH_ZIP))
    print("      완료")

    print(f"[2/4] offline_output 로딩: {OFFLINE_OUTPUT_PATH}")
    print("      (파일이 크면 시간이 걸릴 수 있습니다)")
    with open(OFFLINE_OUTPUT_PATH, "r", encoding="utf-8") as f:
        offline_data = json.load(f)

    if n_queries is not None:
        offline_data = offline_data[:n_queries]

    total = len(offline_data)
    print(f"      처리 대상 query 수: {total}")

    # ── 누적 버퍼 ─────────────────────────────────────────────────────────────
    baseline_acc: Dict[str, List[float]] = {f"Recall@{k}": [] for k in K_LIST}
    baseline_acc["MRR"] = []
    graph_acc:    Dict[str, List[float]] = {f"Recall@{k}": [] for k in K_LIST}
    graph_acc["MRR"] = []

    cov = CoverageStats()
    output_results = []

    print("[3/4] query 처리 중 ...")
    log_interval = max(1, total // 10)

    for idx, item in enumerate(offline_data):
        if (idx + 1) % log_interval == 0 or idx == total - 1:
            print(f"      [{idx + 1}/{total}]")

        query_id   = item["query_id"]
        target_ids = item["target_ids"]
        candidates = item["candidates"]

        # ── baseline: sim 기준 내림차순 정렬 ──────────────────────────────────
        baseline_ids = [
            c["paper_id"]
            for c in sorted(candidates, key=lambda c: c["sim"], reverse=True)
        ]

        # ── graph 적용: feature DataFrame 생성 ────────────────────────────────
        # metadata={} → citation_count_log=0, recency_score=0 (candidates.jsonl 미사용)
        df = build_fallback_dataframe(item, metadata={}, graph=graph)

        if df.empty:
            for k in K_LIST:
                baseline_acc[f"Recall@{k}"].append(0.0)
                graph_acc[f"Recall@{k}"].append(0.0)
            baseline_acc["MRR"].append(0.0)
            graph_acc["MRR"].append(0.0)
            output_results.append({
                "query_id": query_id, "target_ids": target_ids, "candidates": []
            })
            continue

        # ── coverage 컬럼 추가 ────────────────────────────────────────────────
        fwd_counts = {c["paper_id"]: len(graph.references(c["paper_id"])) for c in candidates}
        bwd_counts = {c["paper_id"]: len(graph.cited_by(c["paper_id"]))   for c in candidates}

        df["forward_count"]  = df["paper_id"].map(fwd_counts).fillna(0).astype(int)
        df["backward_count"] = df["paper_id"].map(bwd_counts).fillna(0).astype(int)
        df["has_forward"]    = df["forward_count"]  > 0
        df["has_backward"]   = df["backward_count"] > 0
        df["graph_available"]= df["has_forward"] | df["has_backward"]

        # ── final_score 계산 및 정렬 ──────────────────────────────────────────
        df["final_score"] = (
            W_SEMANTIC * df["semantic_score"]
            + W_BIB    * df["bib_score"]
            + W_GRAPH  * df["graph_score"]
        )
        df_sorted = df.sort_values("final_score", ascending=False).reset_index(drop=True)
        graph_ids = df_sorted["paper_id"].tolist()

        # ── 지표 수집 ──────────────────────────────────────────────────────────
        b_m = compute_metrics(baseline_ids, target_ids)
        g_m = compute_metrics(graph_ids,    target_ids)
        for key in baseline_acc:
            baseline_acc[key].append(b_m[key])
            graph_acc[key].append(g_m[key])

        # ── output 구성 + coverage 누적 ───────────────────────────────────────
        target_set = set(target_ids)
        result_candidates = []

        for rank, row in enumerate(df_sorted.itertuples(index=False), 1):
            is_target     = row.paper_id in target_set
            has_fwd       = bool(row.has_forward)
            has_bwd       = bool(row.has_backward)
            graph_score   = float(row.graph_score)
            graph_avail   = bool(row.graph_available)

            result_candidates.append({
                "rank":           rank,
                "paper_id":       row.paper_id,
                "final_score":    round(float(row.final_score),    6),
                "semantic_score": round(float(row.semantic_score), 6),
                "bib_score":      round(float(row.bib_score),      6),
                "graph_score":    round(graph_score,               6),
                "is_target":      is_target,
                "has_forward":    has_fwd,
                "has_backward":   has_bwd,
                "forward_count":  int(row.forward_count),
                "backward_count": int(row.backward_count),
                "graph_available": graph_avail,
            })

            # coverage 누적
            cov.total_candidates += 1
            if has_fwd:            cov.fwd_count       += 1
            if has_bwd:            cov.bwd_count       += 1
            if graph_score > 0:    cov.graph_nonzero   += 1
            if is_target:
                cov.total_targets  += 1
                if has_fwd:        cov.target_fwd      += 1
                if has_bwd:        cov.target_bwd      += 1
            if rank <= 10:
                cov.top10_total    += 1
                if graph_score == 0:
                    cov.top10_zero_graph += 1

        output_results.append({
            "query_id":       query_id,
            "query_paper_id": "_".join(query_id.split("_")[:-1]),
            "context":        item["context"],
            "target_ids":     target_ids,
            "candidates":     result_candidates,
        })

    # ── 결과 저장 ──────────────────────────────────────────────────────────────
    print("[4/4] 결과 저장")

    # output_graph_stage3.json
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=2)
    print(f"      저장 완료: {OUTPUT_PATH}")

    # eval_baseline_vs_graph.csv
    eval_rows = []
    for metric in [f"Recall@{k}" for k in K_LIST] + ["MRR"]:
        b_val = float(np.mean(baseline_acc[metric]))
        g_val = float(np.mean(graph_acc[metric]))
        eval_rows.append({
            "metric":        metric,
            "baseline":      round(b_val,         6),
            "graph_applied": round(g_val,         6),
            "diff":          round(g_val - b_val, 6),
        })

    with open(EVAL_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "baseline", "graph_applied", "diff"])
        writer.writeheader()
        writer.writerows(eval_rows)
    print(f"      저장 완료: {EVAL_PATH}")

    # coverage_summary.csv
    def _ratio(numer: int, denom: int) -> str:
        return f"{numer / denom:.4f}" if denom > 0 else "N/A"

    cov_rows = [
        ("전체 candidate 중 has_forward 비율",
         _ratio(cov.fwd_count,      cov.total_candidates)),
        ("전체 candidate 중 has_backward 비율",
         _ratio(cov.bwd_count,      cov.total_candidates)),
        ("전체 candidate 중 graph_score > 0 비율",
         _ratio(cov.graph_nonzero,  cov.total_candidates)),
        ("target candidate 중 has_forward 비율",
         _ratio(cov.target_fwd,     cov.total_targets)),
        ("target candidate 중 has_backward 비율",
         _ratio(cov.target_bwd,     cov.total_targets)),
        ("top-10 추천 중 graph_score == 0 비율",
         _ratio(cov.top10_zero_graph, cov.top10_total)),
        ("전체 candidate 수 (raw)",   str(cov.total_candidates)),
        ("전체 target candidate 수",  str(cov.total_targets)),
        ("top-10 슬롯 수",            str(cov.top10_total)),
    ]

    with open(COVERAGE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stat", "value"])
        writer.writerows(cov_rows)
    print(f"      저장 완료: {COVERAGE_PATH}")

    # ── 콘솔 요약 ──────────────────────────────────────────────────────────────
    print("\n=== 평가 요약 ===")
    print(f"{'metric':<20} {'baseline':>10} {'graph':>10} {'diff':>10}")
    print("-" * 54)
    for row in eval_rows:
        print(f"{row['metric']:<20} {row['baseline']:>10.6f}"
              f" {row['graph_applied']:>10.6f} {row['diff']:>+10.6f}")

    print("\n=== Coverage 요약 ===")
    for stat, val in cov_rows[:6]:
        print(f"  {stat}: {val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="앞 10개 query만 실행 (smoke test)")
    parser.add_argument("--n",     type=int, default=None, help="처리할 query 수 (기본: 전체)")
    args = parser.parse_args()

    n = 10 if args.smoke else args.n
    run(n_queries=n)
