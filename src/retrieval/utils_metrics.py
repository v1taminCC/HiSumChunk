from __future__ import annotations

from typing import Dict, Iterable, Sequence, Set

EVAL_KS = (1, 3, 5, 10)


def hit_at_k(ranked_chunk_ids: Sequence[str], relevant_set: Set[str], k: int) -> float:
    if not relevant_set:
        return 0.0
    top = ranked_chunk_ids[:k]
    return 1.0 if any(chunk_id in relevant_set for chunk_id in top) else 0.0


def recall_at_k(ranked_chunk_ids: Sequence[str], relevant_set: Set[str], k: int) -> float:
    if not relevant_set:
        return 0.0
    top = ranked_chunk_ids[:k]
    return sum(1 for chunk_id in top if chunk_id in relevant_set) / len(relevant_set)


def mrr(ranked_chunk_ids: Sequence[str], relevant_set: Set[str]) -> float:
    if not relevant_set:
        return 0.0
    for idx, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / idx
    return 0.0


def evidence_recall_at_k(ranked_chunk_ids: Sequence[str], evidence_groups: Sequence[Sequence[str]], k: int) -> float:
    if not evidence_groups:
        return 0.0
    top = set(ranked_chunk_ids[:k])
    covered = sum(1 for group in evidence_groups if set(group) & top)
    return covered / len(evidence_groups)


def _build_metrics(prefix: str, ranked_chunk_ids: Sequence[str], relevant_set: Set[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in EVAL_KS:
        metrics[f"{prefix}_hit@{k}"] = hit_at_k(ranked_chunk_ids, relevant_set, k)
    metrics[f"{prefix}_mrr"] = mrr(ranked_chunk_ids, relevant_set)
    return metrics


def evaluate_query(
    ranked_chunk_ids: Sequence[str],
    strict_relevant: Set[str],
    relaxed_relevant: Set[str],
    evidence_groups: Sequence[Sequence[str]] | None = None,
) -> Dict[str, float]:
    evidence_groups = evidence_groups or []
    metrics = {}
    metrics.update(_build_metrics("strict", ranked_chunk_ids, strict_relevant))
    metrics.update(_build_metrics("relaxed", ranked_chunk_ids, relaxed_relevant))
    for k in EVAL_KS:
        metrics[f"evidence_recall@{k}"] = evidence_recall_at_k(ranked_chunk_ids, evidence_groups, k)
    return metrics


def aggregate_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(row[k] for row in rows) / len(rows) for k in keys}
