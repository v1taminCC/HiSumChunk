#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy

from config import BOOK_QUERY_DIR, QA_DRAFT_BY_BOOK_DIR, QA_DRAFT_DIR, QA_FINAL_BY_BOOK_DIR, QA_FINAL_DIR, QUERY_DIR
from utils_book import list_book_titles
from utils_io import ensure_dir, read_json, write_json


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    total = max(total, 1)
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix}: [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


def normalize_draft_rows(rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    total_rows = len(rows)
    for row_idx, chapter_row in enumerate(rows, start=1):
        row = deepcopy(chapter_row)
        book_title = str(row.get("book_title", "") or "").strip()
        chapter_id = str(row.get("chapter_id", "") or "").strip()
        if not book_title or not chapter_id:
            print_progress(row_idx, total_rows, prefix="Organizing draft queries")
            continue
        qas = []
        for idx, qa in enumerate(row.get("qas", []) or [], start=1):
            qa = deepcopy(qa)
            if not qa.get("qa_id"):
                qa["qa_id"] = f"{chapter_id}.draft.qa_{idx}"
            if "parse_failed" not in qa:
                qa["parse_failed"] = False
            if "raw_model_output" not in qa:
                qa["raw_model_output"] = ""
            qas.append(qa)
        row["qas"] = qas
        normalized.append(row)
        print_progress(row_idx, total_rows, prefix="Organizing draft queries")
    return normalized


def build_queries_from_final(rows: list[dict]) -> list[dict]:
    queries: list[dict] = []
    total_rows = len(rows)
    for row_idx, row in enumerate(rows, start=1):
        queries.append(
            {
                "query_id": row["qa_id"],
                "book_title": row["book_title"],
                "chapter_id": row["chapter_id"],
                "question": row["question"],
                "answer": row["answer"],
                "question_type": row.get("question_type", "fact"),
                "difficulty": row.get("difficulty", "medium"),
                "evidence_units": row.get("evidence_units", []),
            }
        )
        print_progress(row_idx, total_rows, prefix="Building final queries")
    return queries


def load_source_rows(source: str, book: str | None) -> list[dict]:
    if source == "draft":
        if book:
            return read_json(QA_DRAFT_BY_BOOK_DIR / f"{book}.json")
        all_rows: list[dict] = []
        files = sorted(p for p in QA_DRAFT_BY_BOOK_DIR.glob("*.json") if p.is_file() and not p.name.endswith(".bak"))
        total_files = len(files)
        for file_idx, fp in enumerate(files, start=1):
            all_rows.extend(read_json(fp))
            print_progress(file_idx, total_files, prefix="Reading QA drafts")
        if all_rows:
            return all_rows
        return read_json(QA_DRAFT_DIR / "qa_drafts_all.json")

    if book:
        return read_json(QA_FINAL_BY_BOOK_DIR / f"{book}.json")
    return read_json(QA_FINAL_DIR / "qa_final_all.json")


def group_by_book(rows: list[dict]) -> dict[str, list[dict]]:
    by_book: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        book_title = str(row.get("book_title", "") or "").strip()
        if book_title:
            by_book[book_title].append(row)
    return by_book


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval query files from QA data; draft keeps chapter nesting format")
    parser.add_argument("--source", type=str, choices=["draft", "final"], default="draft")
    parser.add_argument("--book", type=str, default=None, help="Book title; process all QA files if not specified")
    args = parser.parse_args()

    ensure_dir(QUERY_DIR)
    ensure_dir(BOOK_QUERY_DIR)

    rows = load_source_rows(args.source, args.book)
    output_rows = normalize_draft_rows(rows) if args.source == "draft" else build_queries_from_final(rows)
    by_book = group_by_book(output_rows)

    all_rows = []
    target_books = [args.book] if args.book else list_book_titles()
    total_books = len(target_books) if target_books else len(by_book)
    written = 0
    for book_title in target_books:
        book_rows = by_book.get(book_title, [])
        if not book_rows:
            print(f"\nSkipping book with no QA: {book_title}", flush=True)
            continue
        write_json(BOOK_QUERY_DIR / f"{book_title}.json", book_rows)
        all_rows.extend(book_rows)
        written += 1
        print_progress(written, total_books, prefix="Writing queries")

    if not args.book:
        write_json(QUERY_DIR / "queries_all.json", all_rows)
    print(f"Query chapter/row count: {len(all_rows)}")


if __name__ == "__main__":
    main()
