#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from config import BOOKSUM_RAW_DIR, SOURCE_CHAPTER_BY_BOOK_DIR, SOURCE_CHAPTER_DIR
from utils_io import ensure_dir, read_json, write_json

RAW_GLOB = "booksum_raw_part*.json"


BOOK_ID_CLEAN_PATTERNS = [
    r"\.book\s*[ivxlcdm\d]+\s*\.chapter\s*[ivxlcdm\d]+.*$",
    r"\.book\s*[ivxlcdm\d]+.*$",
    r"\.chapter\s*[ivxlcdm\d]+.*$",
    r"\.chapters?\s*[ivxlcdm\d]+\s*-\s*[ivxlcdm\d]+.*$",
    r"\.part\s*[ivxlcdm\d]+.*$",
    r"\.act\s*[ivxlcdm\d]+.*$",
    r"\.scene\s*[ivxlcdm\d]+.*$",
    r"\.epilogue.*$",
]


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    total = max(total, 1)
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix}: [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


def extract_book_title(book_id: Any) -> str:
    text = "" if book_id is None else str(book_id).strip()
    for pattern in BOOK_ID_CLEAN_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" .")


def iter_records(raw_data: dict[str, Any]) -> list[dict[str, Any]]:
    records = raw_data.get("records", {})
    if isinstance(records, list):
        return [row for row in records if isinstance(row, dict)]
    if not isinstance(records, dict):
        return []
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        part_rows = records.get(split_name, []) or []
        rows.extend(row for row in part_rows if isinstance(row, dict))
    return rows


def build_chapter_id(book_title: str, row: dict[str, Any], fallback_index: int) -> str:
    book_id = str(row.get("book_id", "") or "").strip()
    if book_id:
        return book_id
    summary_id = str(row.get("summary_id", "") or row.get("summary_name", "") or "").strip()
    if summary_id:
        return f"{book_title}.{summary_id}"
    return f"{book_title}.summary_{fallback_index}"


def build_rows(raw_dir: Path, glob_pattern: str, only_book: str | None = None) -> list[dict[str, Any]]:
    files = sorted(p for p in raw_dir.glob(glob_pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"BookSum raw part file not found: {raw_dir / glob_pattern}")

    rows: list[dict[str, Any]] = []
    seen_chapter_ids: set[str] = set()
    total_files = len(files)
    for file_idx, path in enumerate(files, start=1):
        raw_data = read_json(path)
        records = iter_records(raw_data)
        for record_idx, record in enumerate(records, start=1):
            summary_text = str(record.get("summary_text", "") or "").strip()
            book_title = extract_book_title(record.get("book_id"))
            if not book_title or not summary_text:
                continue
            if only_book and book_title != only_book:
                continue
            chapter_id = build_chapter_id(book_title, record, len(rows) + record_idx)
            if chapter_id in seen_chapter_ids:
                continue
            seen_chapter_ids.add(chapter_id)
            rows.append(
                {
                    "book_title": book_title,
                    "chapter_id": chapter_id,
                    "summary_id": str(record.get("summary_id", "") or ""),
                    "summary_name": str(record.get("summary_name", "") or ""),
                    "source": str(record.get("source", "") or ""),
                    "bid": record.get("bid"),
                    "is_aggregate": bool(record.get("is_aggregate", False)),
                    "text_source": "booksum_summary_text",
                    "source_file": path.name,
                    "text": summary_text,
                }
            )
        print_progress(file_idx, total_files, prefix="Reading BookSum raw")
    return rows


def write_by_book(rows: list[dict[str, Any]]) -> None:
    ensure_dir(SOURCE_CHAPTER_BY_BOOK_DIR)
    by_book: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_book.setdefault(row["book_title"], []).append(row)
    total_books = len(by_book)
    for book_idx, (book_title, book_rows) in enumerate(sorted(by_book.items()), start=1):
        write_json(SOURCE_CHAPTER_BY_BOOK_DIR / f"{book_title}.json", book_rows)
        print_progress(book_idx, total_books, prefix="Writing by_book")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QA generation input from BookSum raw part summary_text")
    parser.add_argument("--book", type=str, default=None, help="Book title; process all books if not specified")
    parser.add_argument("--raw_dir", type=str, default=str(BOOKSUM_RAW_DIR), help="Directory containing booksum_raw_part*.json")
    parser.add_argument("--glob", type=str, default=RAW_GLOB, help="Raw part file glob pattern")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    rows = build_rows(raw_dir, args.glob, args.book)
    ensure_dir(SOURCE_CHAPTER_DIR)
    write_json(SOURCE_CHAPTER_DIR / "source_chapters_all.json", rows)
    write_by_book(rows)
    print(f"Written summary_text input: {SOURCE_CHAPTER_DIR / 'source_chapters_all.json'}")
    print(f"Record count: {len(rows)}")
    print(f"Book count: {len({row['book_title'] for row in rows})}")


if __name__ == "__main__":
    main()
