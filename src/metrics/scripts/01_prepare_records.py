from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    ANSWER_FILE_PATTERN,
    ANSWERS_DIR,
    PREPARE_STATS_PATH,
    PREPARED_RECORDS_PATH,
    SCHEMES,
    TEXT_TRUNCATION_CHARS,
)
from utils.io_utils import (
    ProgressLogger,
    ensure_directories,
    get_context_text,
    load_json,
    log_message,
    normalize_space,
    record_key,
    safe_write_json,
    safe_write_text,
    set_log_file,
)

DEDUPLICATE_RECORDS = True
KEEP_QUESTION_TYPE_AND_DIFFICULTY = True
PROGRESS_EVERY_FILES = 5
PROGRESS_EVERY_RECORDS = 50


def iter_answer_files() -> list[Path]:
    files: list[Path] = []
    for scheme in SCHEMES:
        files.extend(sorted(ANSWERS_DIR.glob(ANSWER_FILE_PATTERN.format(scheme=scheme))))
    return files


def normalize_record(raw: dict[str, Any], source_file: Path, fallback_scheme: str | None = None) -> dict[str, Any]:
    context_chunks = raw.get("context_chunks", [])
    scheme = normalize_space(raw.get("scheme")) or fallback_scheme or "unknown"
    generated_answer = normalize_space(raw.get("generated_answer"))
    ground_truth_answer = normalize_space(raw.get("ground_truth_answer"))
    context_text = get_context_text(context_chunks, truncation_chars=TEXT_TRUNCATION_CHARS)
    record = {
        "query_id": normalize_space(raw.get("query_id")),
        "book_title": normalize_space(raw.get("book_title")),
        "scheme": scheme,
        "question": normalize_space(raw.get("question")),
        "ground_truth_answer": ground_truth_answer,
        "generated_answer": generated_answer,
        "context_chunks": context_chunks if isinstance(context_chunks, list) else [],
        "context_text": context_text,
        "generation_model": normalize_space(raw.get("generation_model")),
        "error": normalize_space(raw.get("error")),
        "valid_answer": bool(generated_answer and not normalize_space(raw.get("error"))),
        "source_file": str(source_file),
        "record_key": "",
    }
    if KEEP_QUESTION_TYPE_AND_DIFFICULTY:
        record["question_type"] = normalize_space(raw.get("question_type"))
        record["difficulty"] = normalize_space(raw.get("difficulty"))
    record["record_key"] = record_key(record)
    return record


def infer_scheme_from_filename(path: Path) -> str | None:
    name = path.name
    for scheme in SCHEMES:
        if f".in_book.{scheme}.top5.json" in name:
            return scheme
    return None


def count_raw_records(files: list[Path]) -> int:
    total = 0
    progress = ProgressLogger("count raw records", total=len(files), every=PROGRESS_EVERY_FILES)
    for index, path in enumerate(files, start=1):
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Expected list JSON in {path}")
        total += len(data)
        progress.update(index, detail=f"{path.name}: {len(data)} records")
    progress.finish(detail=f"raw_records={total}")
    return total


def build_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    duplicate_keys: list[str] = []
    seen: set[str] = set()
    files = iter_answer_files()
    raw_total = count_raw_records(files)
    file_progress = ProgressLogger("prepare files", total=len(files), every=PROGRESS_EVERY_FILES)
    record_progress = ProgressLogger("prepare records", total=raw_total, every=PROGRESS_EVERY_RECORDS)
    processed_raw = 0

    for file_index, path in enumerate(files, start=1):
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Expected list JSON in {path}")
        fallback_scheme = infer_scheme_from_filename(path)
        for raw in data:
            processed_raw += 1
            if not isinstance(raw, dict):
                record_progress.update(processed_raw, detail="skipped non-dict record")
                continue
            record = normalize_record(raw, path, fallback_scheme)
            key = record["record_key"]
            if DEDUPLICATE_RECORDS and key in seen:
                duplicate_keys.append(key)
                record_progress.update(processed_raw, detail=f"duplicate skipped | kept={len(records)}")
                continue
            seen.add(key)
            records.append(record)
            record_progress.update(processed_raw, detail=f"kept={len(records)}")
        file_progress.update(file_index, detail=path.name)

    file_progress.finish(detail=f"files={len(files)}")
    record_progress.finish(detail=f"kept={len(records)}, duplicates={len(duplicate_keys)}")

    scheme_counts = Counter(record["scheme"] for record in records)
    valid_counts = Counter(record["scheme"] for record in records if record["valid_answer"])
    empty_counts = Counter(record["scheme"] for record in records if not record["generated_answer"])
    book_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for scheme in SCHEMES:
        by_book = Counter(record["book_title"] for record in records if record["scheme"] == scheme)
        book_counts[scheme] = dict(sorted(by_book.items()))

    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_files": [str(path) for path in files],
        "total_records": len(records),
        "scheme_counts": dict(scheme_counts),
        "valid_answer_counts": dict(valid_counts),
        "empty_answer_counts": dict(empty_counts),
        "duplicate_records_skipped": len(duplicate_keys),
        "duplicate_keys_sample": duplicate_keys[:20],
        "book_counts_by_scheme": book_counts,
    }
    return records, stats


def format_stats(stats: dict[str, Any]) -> str:
    lines = ["Prepared evaluation records", "=" * 32, ""]
    lines.append(f"Created at: {stats['created_at']}")
    lines.append(f"Input files: {len(stats['input_files'])}")
    lines.append(f"Total records: {stats['total_records']}")
    lines.append(f"Duplicate records skipped: {stats['duplicate_records_skipped']}")
    lines.append("")
    lines.append("By scheme:")
    for scheme in SCHEMES:
        total = stats["scheme_counts"].get(scheme, 0)
        valid = stats["valid_answer_counts"].get(scheme, 0)
        empty = stats["empty_answer_counts"].get(scheme, 0)
        lines.append(f"- {scheme}: total={total}, valid={valid}, empty_answer={empty}")
    return "\n".join(lines) + "\n"


def main() -> None:
    log_path = set_log_file(__file__)
    log_message(f"Log file: {log_path}")
    log_message("Step 01 started: prepare unified evaluation records")
    ensure_directories()
    records, stats = build_records()
    safe_write_json(records, PREPARED_RECORDS_PATH)
    safe_write_text(format_stats(stats), PREPARE_STATS_PATH)
    log_message(f"Prepared {len(records)} records")
    log_message(f"Records: {PREPARED_RECORDS_PATH}")
    log_message(f"Stats: {PREPARE_STATS_PATH}")
    log_message("Step 01 finished")


if __name__ == "__main__":
    main()
