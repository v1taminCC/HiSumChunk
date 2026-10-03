from __future__ import annotations

from datetime import datetime
from typing import Any

from config import ENTITY_METRICS_PATH, PREPARED_RECORDS_PATH, SPACY_MODEL
from utils.entity_utils import entity_overlap_metrics, entity_support_metrics, extract_entities, load_spacy_model
from utils.io_utils import ProgressLogger, ensure_directories, load_json, log_message, safe_write_json, set_log_file

SAVE_SOURCE_FIELDS = ["record_key", "query_id", "book_title", "scheme"]
PROGRESS_EVERY_RECORDS = 10
ENABLE_RESUME = True
SAVE_EVERY_RECORDS = 50
REQUIRED_METRIC_FIELDS = ["correctness_entity_f1", "hallucination_unsupported_entity_rate", "hallucination_entity_support_rate"]


def load_existing_items() -> dict[str, dict[str, Any]]:
    if not ENABLE_RESUME or not ENTITY_METRICS_PATH.exists():
        return {}
    payload = load_json(ENTITY_METRICS_PATH)
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
    log_message(f"Resume enabled: loaded {len(existing)} completed entity metric records from {ENTITY_METRICS_PATH}")
    return existing


def save_checkpoint(items_by_key: dict[str, dict[str, Any]], metric_group: str = "entity_metrics_partial") -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_group": metric_group,
        "spacy_model": SPACY_MODEL,
        "resume_enabled": ENABLE_RESUME,
        "items": list(items_by_key.values()),
    }
    safe_write_json(payload, ENTITY_METRICS_PATH)
    log_message(f"Checkpointed {len(items_by_key)} entity metric records: {ENTITY_METRICS_PATH}")


def compute_entity_metrics(records: list[dict[str, Any]], existing_by_key: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pending_records = [record for record in records if record.get("record_key") not in existing_by_key]
    log_message(f"Entity resume status | total={len(records)} | done={len(existing_by_key)} | pending={len(pending_records)}")
    if pending_records:
        log_message(f"Loading spaCy model: {SPACY_MODEL}")
        nlp = load_spacy_model(SPACY_MODEL)
        log_message(f"spaCy model loaded: {SPACY_MODEL}")
        progress = ProgressLogger("entity pending records", total=len(pending_records), every=PROGRESS_EVERY_RECORDS)
        for index, record in enumerate(pending_records, start=1):
            generated = record.get("generated_answer", "")
            reference = record.get("ground_truth_answer", "")
            context = record.get("context_text", "")

            answer_entities = extract_entities(generated, nlp)
            reference_entities = extract_entities(reference, nlp)
            context_entities = extract_entities(context, nlp)

            overlap = entity_overlap_metrics(answer_entities, reference_entities)
            support = entity_support_metrics(answer_entities, context_entities)

            item = {field: record.get(field, "") for field in SAVE_SOURCE_FIELDS}
            item.update(
                {
                    "valid_answer": record.get("valid_answer", False),
                    "answer_entities": answer_entities,
                    "reference_entities": reference_entities,
                    "context_entities": context_entities,
                    "answer_entity_count": len(answer_entities),
                    "reference_entity_count": len(reference_entities),
                    "context_entity_count": len(context_entities),
                    "correctness_entity_precision": overlap["precision"],
                    "correctness_entity_recall": overlap["recall"],
                    "correctness_entity_f1": overlap["f1"],
                    "correctness_entity_overlap_count": overlap["overlap_count"],
                    "hallucination_unsupported_entity_rate": support["unsupported_entity_rate"],
                    "hallucination_entity_support_rate": support["entity_support_rate"],
                    "hallucination_supported_entity_count": support["supported_entity_count"],
                    "hallucination_unsupported_entity_count": support["unsupported_entity_count"],
                }
            )
            existing_by_key[record["record_key"]] = item
            progress.update(
                index,
                detail=(
                    f"scheme={record.get('scheme', '')}, total_done={len(existing_by_key)}, "
                    f"answer_entities={len(answer_entities)}, reference_entities={len(reference_entities)}, "
                    f"context_entities={len(context_entities)}"
                ),
            )
            if SAVE_EVERY_RECORDS and index % SAVE_EVERY_RECORDS == 0:
                save_checkpoint(existing_by_key)
        progress.finish(detail=f"new_rows={len(pending_records)}, total_done={len(existing_by_key)}")
    else:
        log_message("No pending entity metric records; all records are already completed")
    return [existing_by_key[record["record_key"]] for record in records if record.get("record_key") in existing_by_key]


def main() -> None:
    log_path = set_log_file(__file__)
    log_message(f"Log file: {log_path}")
    log_message("Step 04 started: compute entity correctness and hallucination metrics")
    ensure_directories()
    records = load_json(PREPARED_RECORDS_PATH)
    if not isinstance(records, list):
        raise ValueError(f"Expected list JSON in {PREPARED_RECORDS_PATH}")
    log_message(f"Loaded prepared records: {len(records)}")
    existing_by_key = load_existing_items()
    results = compute_entity_metrics(records, existing_by_key)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_group": "entity_metrics",
        "spacy_model": SPACY_MODEL,
        "resume_enabled": ENABLE_RESUME,
        "items": results,
    }
    safe_write_json(payload, ENTITY_METRICS_PATH)
    log_message(f"Wrote entity metrics for {len(results)} records: {ENTITY_METRICS_PATH}")
    log_message("Step 04 finished")


if __name__ == "__main__":
    main()
