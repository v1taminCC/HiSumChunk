from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from config import (
    AUTO_METRICS_ALL_PATH,
    AUTO_METRICS_BY_SCHEME_CSV_PATH,
    AUTO_METRICS_BY_SCHEME_JSON_PATH,
    AUTO_METRICS_SUMMARY_PATH,
    CONTEXT_METRICS_PATH,
    CSV_FLOAT_PRECISION,
    ENTITY_METRICS_PATH,
    REFERENCE_METRICS_PATH,
    SCHEMES,
)
from utils.io_utils import ProgressLogger, ensure_directories, load_json, log_message, mean, safe_write_csv, safe_write_json, safe_write_text, set_log_file

METRIC_COLUMNS = [
    "grounding_context_rougeL",
    "grounding_context_bertscore_f1",
    "correctness_reference_rougeL",
    "correctness_reference_bertscore_f1",
    "correctness_entity_precision",
    "correctness_entity_recall",
    "correctness_entity_f1",
    "hallucination_unsupported_entity_rate",
    "hallucination_entity_support_rate",
]

LOWER_IS_BETTER = {"hallucination_unsupported_entity_rate"}
KEY_FIELDS = ["record_key", "query_id", "book_title", "scheme", "valid_answer"]
PROGRESS_EVERY_ITEMS = 100


def load_metric_items(path) -> list[dict[str, Any]]:
    log_message(f"Loading metric file: {path}")
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = payload["items"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unsupported metric file format: {path}")
    log_message(f"Loaded {len(items)} metric items from {path}")
    return items


def merge_items(metric_item_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    total = sum(len(items) for items in metric_item_lists)
    progress = ProgressLogger("merge metric items", total=total, every=PROGRESS_EVERY_ITEMS)
    processed = 0
    for items in metric_item_lists:
        for item in items:
            processed += 1
            key = item.get("record_key")
            if not key:
                progress.update(processed, detail="missing record_key")
                continue
            if key not in merged:
                merged[key] = {field: item.get(field, "") for field in KEY_FIELDS}
            for metric in METRIC_COLUMNS:
                if metric in item:
                    merged[key][metric] = item.get(metric)
            for extra in [
                "answer_sentence_count",
                "context_sentence_count",
                "context_bertscore_pair_count",
                "answer_entity_count",
                "reference_entity_count",
                "context_entity_count",
            ]:
                if extra in item:
                    merged[key][extra] = item.get(extra)
            progress.update(processed, detail=f"merged_records={len(merged)}")
    progress.finish(detail=f"merged_records={len(merged)}")
    return list(merged.values())


def round_float(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, CSV_FLOAT_PRECISION)
    return value


def aggregate_by_scheme(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    progress = ProgressLogger("group items by scheme", total=len(items), every=PROGRESS_EVERY_ITEMS)
    for index, item in enumerate(items, start=1):
        grouped[item.get("scheme", "unknown")].append(item)
        progress.update(index, detail=f"scheme={item.get('scheme', 'unknown')}")
    progress.finish(detail=f"schemes={len(grouped)}")

    rows: list[dict[str, Any]] = []
    scheme_progress = ProgressLogger("aggregate schemes", total=len(SCHEMES), every=1)
    for index, scheme in enumerate(SCHEMES, start=1):
        group = grouped.get(scheme, [])
        row: dict[str, Any] = {
            "scheme": scheme,
            "n_records": len(group),
            "n_valid_answer": sum(1 for item in group if item.get("valid_answer")),
        }
        for metric in METRIC_COLUMNS:
            row[metric] = mean(item.get(metric) for item in group)
        rows.append(row)
        scheme_progress.update(index, detail=f"scheme={scheme}, n={len(group)}")
    scheme_progress.finish(detail=f"rows={len(rows)}")
    return rows


def add_best_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    progress = ProgressLogger("mark best metrics", total=len(METRIC_COLUMNS), every=1)
    for index, metric in enumerate(METRIC_COLUMNS, start=1):
        valid_values = [(idx, row.get(metric)) for idx, row in enumerate(rows) if row.get(metric) is not None]
        if not valid_values:
            progress.update(index, detail=f"{metric}: no valid values")
            continue
        if metric in LOWER_IS_BETTER:
            best_value = min(value for _, value in valid_values)
        else:
            best_value = max(value for _, value in valid_values)
        for idx, value in valid_values:
            rows[idx][f"best_{metric}"] = bool(value == best_value)
        progress.update(index, detail=f"{metric}: best={best_value}")
    progress.finish(detail="best flags added")
    return rows


def format_summary(rows: list[dict[str, Any]]) -> str:
    lines = ["Automatic hard metrics by scheme", "=" * 40, ""]
    lines.append("Direction: all metrics are higher-is-better except hallucination_unsupported_entity_rate.")
    lines.append("")
    for row in rows:
        lines.append(f"[{row['scheme']}] n={row['n_records']}, valid={row['n_valid_answer']}")
        for metric in METRIC_COLUMNS:
            value = row.get(metric)
            suffix = " *best*" if row.get(f"best_{metric}") else ""
            if value is None:
                lines.append(f"  - {metric}: NA{suffix}")
            else:
                lines.append(f"  - {metric}: {value:.{CSV_FLOAT_PRECISION}f}{suffix}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    log_path = set_log_file(__file__)
    log_message(f"Log file: {log_path}")
    log_message("Step 05 started: aggregate hard metrics")
    ensure_directories()
    metric_items = [
        load_metric_items(REFERENCE_METRICS_PATH),
        load_metric_items(CONTEXT_METRICS_PATH),
        load_metric_items(ENTITY_METRICS_PATH),
    ]
    merged_items = merge_items(metric_items)
    rows = add_best_flags(aggregate_by_scheme(merged_items))

    safe_write_json(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "metric_columns": METRIC_COLUMNS,
            "items": merged_items,
        },
        AUTO_METRICS_ALL_PATH,
    )
    log_message(f"Wrote merged item metrics: {AUTO_METRICS_ALL_PATH}")

    safe_write_json(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "metric_columns": METRIC_COLUMNS,
            "rows": rows,
        },
        AUTO_METRICS_BY_SCHEME_JSON_PATH,
    )
    log_message(f"Wrote scheme JSON table: {AUTO_METRICS_BY_SCHEME_JSON_PATH}")

    csv_rows = [{key: round_float(value) for key, value in row.items()} for row in rows]
    safe_write_csv(csv_rows, AUTO_METRICS_BY_SCHEME_CSV_PATH)
    safe_write_text(format_summary(rows), AUTO_METRICS_SUMMARY_PATH)
    log_message(f"Wrote scheme CSV table: {AUTO_METRICS_BY_SCHEME_CSV_PATH}")
    log_message(f"Wrote summary: {AUTO_METRICS_SUMMARY_PATH}")
    log_message("Step 05 finished")


if __name__ == "__main__":
    main()
