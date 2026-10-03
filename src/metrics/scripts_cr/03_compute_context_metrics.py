from __future__ import annotations

from datetime import datetime
from typing import Any

from config import (
    BERTSCORE_BATCH_SIZE,
    BERTSCORE_DEVICE,
    BERTSCORE_LANG,
    BERTSCORE_MODEL,
    BERTSCORE_RESCALE_WITH_BASELINE,
    CONTEXT_METRICS_PATH,
    MAX_CONTEXT_SENTENCES,
    MIN_SENTENCE_CHARS,
    PREPARED_RECORDS_PATH,
)
from utils.io_utils import ProgressLogger, ensure_directories, load_json, log_message, mean, safe_write_json, set_log_file, split_sentences
from utils.metric_utils import RougeLScorer, compute_bertscore_f1

SAVE_SOURCE_FIELDS = ["record_key", "query_id", "book_title", "scheme"]
SAVE_EVERY_RECORDS = 5
PROGRESS_EVERY_RECORDS = 1
ENABLE_RESUME = True
REQUIRED_METRIC_FIELDS = ["grounding_context_rougeL", "grounding_context_bertscore_f1"]


def load_existing_items() -> dict[str, dict[str, Any]]:
    if not ENABLE_RESUME or not CONTEXT_METRICS_PATH.exists():
        return {}
    payload = load_json(CONTEXT_METRICS_PATH)
    items = payload.get("items", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    existing: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = item.get("record_key")
        if not key:
            continue
        if all(field in item for field in REQUIRED_METRIC_FIELDS):
            existing[key] = item
    log_message(f"Resume enabled: loaded {len(existing)} completed context metric records from {CONTEXT_METRICS_PATH}")
    return existing


def save_checkpoint(items_by_key: dict[str, dict[str, Any]], metric_group: str = "context_metrics_partial") -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_group": metric_group,
        "bertscore_model": BERTSCORE_MODEL,
        "max_context_sentences": MAX_CONTEXT_SENTENCES,
        "resume_enabled": ENABLE_RESUME,
        "items": list(items_by_key.values()),
    }
    safe_write_json(payload, CONTEXT_METRICS_PATH)
    log_message(f"Checkpointed {len(items_by_key)} context metric records: {CONTEXT_METRICS_PATH}")


def compute_context_rouge_for_record(record: dict[str, Any], rouge: RougeLScorer) -> tuple[float | None, list[float | None], int, int]:
    answer_sentences = split_sentences(record.get("generated_answer", ""), min_chars=MIN_SENTENCE_CHARS)
    context_sentences = split_sentences(record.get("context_text", ""), min_chars=MIN_SENTENCE_CHARS)[:MAX_CONTEXT_SENTENCES]
    if not answer_sentences or not context_sentences:
        return None, [], len(answer_sentences), len(context_sentences)
    sentence_max_scores: list[float | None] = []
    for answer_sentence in answer_sentences:
        scores = [rouge.rouge_l_f1(context_sentence, answer_sentence) for context_sentence in context_sentences]
        valid = [score for score in scores if score is not None]
        sentence_max_scores.append(max(valid) if valid else None)
    return mean(sentence_max_scores), sentence_max_scores, len(answer_sentences), len(context_sentences)


def compute_context_bertscore_for_record(record: dict[str, Any], record_index: int, total_records: int) -> tuple[float | None, list[float | None], int]:
    answer_sentences = split_sentences(record.get("generated_answer", ""), min_chars=MIN_SENTENCE_CHARS)
    context_sentences = split_sentences(record.get("context_text", ""), min_chars=MIN_SENTENCE_CHARS)[:MAX_CONTEXT_SENTENCES]
    if not answer_sentences or not context_sentences:
        return None, [], 0

    predictions: list[str] = []
    references: list[str] = []
    pair_owner: list[int] = []
    for answer_idx, answer_sentence in enumerate(answer_sentences):
        for context_sentence in context_sentences:
            predictions.append(answer_sentence)
            references.append(context_sentence)
            pair_owner.append(answer_idx)

    pair_count = len(predictions)
    log_message(
        f"Context BERTScore record {record_index}/{total_records} | "
        f"scheme={record.get('scheme', '')} | answer_sentences={len(answer_sentences)} | "
        f"context_sentences={len(context_sentences)} | pairs={pair_count}"
    )
    pair_scores = compute_bertscore_f1(
        predictions=predictions,
        references=references,
        model_type=BERTSCORE_MODEL,
        lang=BERTSCORE_LANG,
        batch_size=BERTSCORE_BATCH_SIZE,
        rescale_with_baseline=BERTSCORE_RESCALE_WITH_BASELINE,
        device=BERTSCORE_DEVICE,
        progress_name=f"context BERTScore record {record_index}/{total_records}",
    )
    by_answer_sentence: list[list[float | None]] = [[] for _ in answer_sentences]
    for owner, score in zip(pair_owner, pair_scores):
        by_answer_sentence[owner].append(score)
    sentence_max_scores: list[float | None] = []
    for scores in by_answer_sentence:
        valid = [score for score in scores if score is not None]
        sentence_max_scores.append(max(valid) if valid else None)
    return mean(sentence_max_scores), sentence_max_scores, pair_count


def compute_context_metrics(records: list[dict[str, Any]], existing_by_key: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rouge = RougeLScorer()
    pending_records = [record for record in records if record.get("record_key") not in existing_by_key]
    log_message(f"Context resume status | total={len(records)} | done={len(existing_by_key)} | pending={len(pending_records)}")
    total_pairs = sum(int(item.get("context_bertscore_pair_count", 0) or 0) for item in existing_by_key.values())

    if pending_records:
        progress = ProgressLogger("context pending records", total=len(pending_records), every=PROGRESS_EVERY_RECORDS)
        for pending_index, record in enumerate(pending_records, start=1):
            original_index = records.index(record) + 1
            rouge_mean, rouge_sentence_scores, answer_sentence_count, context_sentence_count = compute_context_rouge_for_record(record, rouge)
            bert_mean, bert_sentence_scores, pair_count = compute_context_bertscore_for_record(record, original_index, len(records))
            total_pairs += pair_count
            item = {field: record.get(field, "") for field in SAVE_SOURCE_FIELDS}
            item.update(
                {
                    "valid_answer": record.get("valid_answer", False),
                    "answer_sentence_count": answer_sentence_count,
                    "context_sentence_count": context_sentence_count,
                    "context_bertscore_pair_count": pair_count,
                    "grounding_context_rougeL": rouge_mean,
                    "grounding_context_bertscore_f1": bert_mean,
                    "grounding_context_rougeL_sentence_max": rouge_sentence_scores,
                    "grounding_context_bertscore_sentence_max": bert_sentence_scores,
                }
            )
            existing_by_key[record["record_key"]] = item
            progress.update(
                pending_index,
                detail=f"scheme={record.get('scheme', '')}, total_done={len(existing_by_key)}, total_pairs={total_pairs}",
            )
            if SAVE_EVERY_RECORDS and pending_index % SAVE_EVERY_RECORDS == 0:
                save_checkpoint(existing_by_key)
        progress.finish(detail=f"new_rows={len(pending_records)}, total_done={len(existing_by_key)}, total_pairs={total_pairs}")
    else:
        log_message("No pending context metric records; all records are already completed")

    return [existing_by_key[record["record_key"]] for record in records if record.get("record_key") in existing_by_key]


def main() -> None:
    log_path = set_log_file(__file__)
    log_message(f"Log file: {log_path}")
    log_message("Step 03 started: compute context ROUGE-L and BERTScore")
    ensure_directories()
    records = load_json(PREPARED_RECORDS_PATH)
    if not isinstance(records, list):
        raise ValueError(f"Expected list JSON in {PREPARED_RECORDS_PATH}")
    log_message(
        f"Loaded prepared records: {len(records)} | max_context_sentences={MAX_CONTEXT_SENTENCES} | "
        f"BERTScore model={BERTSCORE_MODEL} | batch_size={BERTSCORE_BATCH_SIZE}"
    )
    existing_by_key = load_existing_items()
    results = compute_context_metrics(records, existing_by_key)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_group": "context_metrics",
        "bertscore_model": BERTSCORE_MODEL,
        "max_context_sentences": MAX_CONTEXT_SENTENCES,
        "resume_enabled": ENABLE_RESUME,
        "items": results,
    }
    safe_write_json(payload, CONTEXT_METRICS_PATH)
    log_message(f"Wrote context metrics for {len(results)} records: {CONTEXT_METRICS_PATH}")
    log_message("Step 03 finished")


if __name__ == "__main__":
    main()
