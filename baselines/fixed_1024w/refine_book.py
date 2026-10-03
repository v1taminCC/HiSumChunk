#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate BookSum JSON from split_output into one TXT per book, injecting chapter anchors.

Rules:
1) Only write chapter body text.
2) Preserve anchor before each chapter: <<<CHAPTER_ID:{book_id}>>>
3) No page markers injected.
4) Concatenate chapters in JSON appearance order (no deduplication).
5) Print duplicate book_id statistics at the end.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# ============================================================
# Configurable parameters
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
WORK_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))
INPUT_DIR = WORK_BOOK_DIR
OUTPUT_DIR = WORK_BOOK_DIR
JSON_GLOB = "booksum_raw_part_*_test.json"

# Which splits to read from records
READ_SPLITS = ["test"]

# Separator between chapters (written after each chapter body)
CHAPTER_SEPARATOR = "\n\n"

# Anchor template
CHAPTER_ANCHOR_TEMPLATE = "<<<CHAPTER_ID:{book_id}>>>"

# Whether to output anchors for empty chapters
KEEP_EMPTY_CHAPTER = False


# ============================================================
# Utility functions
# ============================================================
def extract_book_title(book_id: Any) -> str:
    """Extract book title from book_id for grouping into one TXT per book."""
    s = "" if book_id is None else str(book_id).strip()

    patterns = [
        r"\.book\s*[ivxlcdm\d]+\s*\.chapter\s*[ivxlcdm\d]+\s*-\s*chapter\s*[ivxlcdm\d]+.*$",
        r"\.book\s*[ivxlcdm\d]+\s*\.chapter\s*[ivxlcdm\d]+.*$",
        r"\.chapters?\s*[ivxlcdm\d]+\s*-\s*[ivxlcdm\d]+.*$",
        r"\.chapter\s*[ivxlcdm\d]+.*$",
        r"\.book\s*[ivxlcdm\d]+.*$",
        r"\.part\s*[ivxlcdm\d]+.*$",
        r"\.act\s*[ivxlcdm\d]+.*$",
        r"\.scene\s*[ivxlcdm\d]+.*$",
        r"\.epilogue.*$",
    ]
    for p in patterns:
        s = re.sub(p, "", s, flags=re.I)

    s = re.sub(r"\s+", " ", s).strip(" .")
    return s


def iter_records(data: Dict[str, Any], splits: Iterable[str]) -> Iterable[Dict[str, Any]]:
    records = data.get("records", {})
    if not isinstance(records, dict):
        return []
    out: List[Dict[str, Any]] = []
    for split in splits:
        rows = records.get(split, []) or []
        if isinstance(rows, list):
            out.extend(rows)
    return out


# ============================================================
# Main pipeline
# ============================================================
def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted([p for p in INPUT_DIR.glob(JSON_GLOB) if p.is_file()])
    if not json_files:
        raise FileNotFoundError(f"No JSON files matched: {INPUT_DIR / JSON_GLOB}")

    # book_title -> list[str] (append chapter texts in order)
    book_chapter_parts: Dict[str, List[str]] = defaultdict(list)

    # Track duplicate book_ids
    book_id_counter: Counter[str] = Counter()

    total_rows = 0
    total_kept_chapters = 0

    for fp in json_files:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for row in iter_records(data, READ_SPLITS):
            total_rows += 1

            book_id = row.get("book_id")
            chapter = row.get("chapter")

            if not isinstance(book_id, str) or not book_id.strip():
                continue

            book_id = book_id.strip()
            book_id_counter[book_id] += 1

            chapter_text = chapter if isinstance(chapter, str) else ""
            if not chapter_text and not KEEP_EMPTY_CHAPTER:
                continue

            book_title = extract_book_title(book_id)
            if not book_title:
                continue

            chapter_anchor = CHAPTER_ANCHOR_TEMPLATE.format(book_id=book_id)
            chapter_part = f"{chapter_anchor}\n{chapter_text.rstrip()}".rstrip()
            book_chapter_parts[book_title].append(chapter_part)
            total_kept_chapters += 1

    total_output_books = 0

    # Write one TXT per book (filename not sanitized, as requested)
    for book_title, chapter_parts in book_chapter_parts.items():
        full_book_text = CHAPTER_SEPARATOR.join(chapter_parts).rstrip() + "\n"

        out_file = OUTPUT_DIR / f"{book_title}.txt"
        with out_file.open("w", encoding="utf-8") as f:
            f.write(full_book_text)

        total_output_books += 1

    # Print statistics
    print("=== Done ===")
    print(f"JSON files: {len(json_files)}")
    print(f"Records scanned: {total_rows}")
    print(f"Chapters written: {total_kept_chapters}")
    print(f"Output books (txt): {total_output_books}")
    print(f"Output directory: {OUTPUT_DIR}")

    dup_items = [(bid, c) for bid, c in book_id_counter.items() if c > 1]
    dup_items.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Duplicate book_id statistics (only duplicates shown) ===")
    print(f"Duplicate book_id count: {len(dup_items)}")
    if not dup_items:
        print("No duplicates")
    else:
        for bid, c in dup_items:
            print(f"{bid}\tcount={c}")


if __name__ == "__main__":
    main()
