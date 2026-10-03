from __future__ import annotations

from datetime import datetime
from typing import Any

from config import (
    BERTSCORE_BATCH_SIZE,
    BERTSCORE_DEVICE,
    BERTSCORE_LANG,
    BERTSCORE_MODEL,
    BERTSCORE_RESCALE_WITH_BASELINE,
    PREPARED_RECORDS_PATH,
    REFERENCE_METRICS_PATH,
)
from utils.io_utils import ProgressLogger, ensure_directories, load_json, log_message, safe_write_json, set_log_file
from utils.metric_utils import RougeLScorer, compute_bertscore_f1

SAVE_SOURCE_FIELDS = ["record_key", "query_id", "book_title", "scheme"]
PROGRESS_EVERY_RECORDS = 50
ENABLE_RESUME = True
SAVE_EVERY_RECORDS = 50
REQUIRED_METRIC_FIELDS = ["correctness_reference_rougeL", "correctness_reference_bertscore_f1"]


def load_existing_items() -> dict[str, dict[str, Any]]:
    if not ENABLE_RESUME or not REFERENCE_METRICS_PATH.exists():
        return {}
    payload = load_json(REFERENCE_METRICS_PATH)
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
    log_message(f"Resume enabled: loaded {len(existing)} completed reference metric records from {REFERENCE_METRICS_PATH}")
    return existing


def save_checkpoint(items_by_key: dict[str, dict[str, Any]], metric_group: str = "reference_metrics_partial") -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_group": metric_group,
        "bertscore_model": BERTSCORE_MODEL,
        "resume_enabled": ENABLE_RESUME,
        "items": list(items_by_key.values()),
    }
    safe_write_json(payload, REFERENCE_METRICS_PATH)
    log_message(f"Checkpointed {len(items_by_key)} reference metric records: {REFERENCE_METRICS_PATH}")


def compute_reference_metrics(records: list[dict[str, Any]], existing_by_key: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rouge = RougeLScorer()
    pending_records = [record for record in records if record.get("record_key") not in existing_by_key]
    log_message(f"Reference resume status | total={len(records)} | done={len(existing_by_key)} | pending={len(pending_records)}")

    if pending_records:
        predictions = [record.get("generated_answer", "") for record in pending_records]
        references = [record.get("ground_truth_answer", "") for record in pending_records]
        log_message(
            f"Computing reference BERTScore | pending_records={len(pending_records)} | model={BERTSCORE_MODEL} | batch_size={BERTSCORE_BATCH_SIZE}"
        )
        bert_f1 = compute_bertscore_f1(
            predictions=predictions,
            references=references,
            model_type=BERTSCORE_MODEL,
            lang=BERTSCORE_LANG,
            batch_size=BERTSCORE_BATCH_SIZE,
            rescale_with_baseline=BERTSCORE_RESCALE_WITH_BASELINE,
            device=BERTSCORE_DEVICE,
            progress_name="reference BERTScore pending records",
        )
        log_message("Reference BERTScore finished; computing reference ROUGE-L for pending rows")

        progress = ProgressLogger("reference pending rows", total=len(pending_records), every=PROGRESS_EVERY_RECORDS)
        for idx, record in enumerate(pending_records, start=1):
            generated = record.get("generated_answer", "")
            reference = record.get("ground_truth_answer", "")
            item = {field: record.get(field, "") for field in SAVE_SOURCE_FIELDS}
            item.update(
                {
                    "valid_answer": record.get("valid_answer", False),
                    "correctness_reference_rougeL": rouge.rouge_l_f1(reference, generated),
                    "correctness_reference_bertscore_f1": bert_f1[idx - 1],
                }
            )
            existing_by_key[record["record_key"]] = item
            progress.update(idx, detail=f"scheme={record.get('scheme', '')}, total_done={len(existing_by_key)}")
            if SAVE_EVERY_RECORDS and idx % SAVE_EVERY_RECORDS == 0:
                save_checkpoint(existing_by_key)
        progress.finish(detail=f"new_rows={len(pending_records)}, total_done={len(existing_by_key)}")
    else:
        log_message("No pending reference metric records; all records are already completed")

    return [existing_by_key[record["record_key"]] for record in records if record.get("record_key") in existing_by_key]


def main() -> None:
    log_path = set_log_file(__file__)
    log_message(f"Log file: {log_path}")
    log_message("Step 02 started: compute reference ROUGE-L and BERTScore")
    ensure_directories()
    records = load_json(PREPARED_RECORDS_PATH)
    if not isinstance(records, list):
        raise ValueError(f"Expected list JSON in {PREPARED_RECORDS_PATH}")
    log_message(f"Loaded prepared records: {len(records)}")
    existing_by_key = load_existing_items()
    results = compute_reference_metrics(records, existing_by_key)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_group": "reference_metrics",
        "bertscore_model": BERTSCORE_MODEL,
        "resume_enabled": ENABLE_RESUME,
        "items": results,
    }
    safe_write_json(payload, REFERENCE_METRICS_PATH)
    log_message(f"Wrote reference metrics for {len(results)} records: {REFERENCE_METRICS_PATH}")
    log_message("Step 02 finished")


if __name__ == "__main__":
    main()
