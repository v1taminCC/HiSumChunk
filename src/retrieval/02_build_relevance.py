#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List

from config import BOOK_CHUNK_DIR, BOOK_QUERY_DIR, BOOK_RELEVANCE_DIR, CHUNK_DIR, QUERY_DIR, RELEVANCE_DIR, SUPPORTED_SCHEMES
from utils_io import ensure_dir, read_json, write_json


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    total = max(total, 1)
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix}: [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


def normalize_query_row(row: dict, row_idx: int) -> list[dict]:
    if "question" in row:
        question = str(row.get("question", "") or "").strip()
        if not question:
            return []
        return [
            {
                "query_id": str(row.get("query_id", "") or f"query_{row_idx}"),
                "book_title": str(row.get("book_title", "") or "").strip(),
                "chapter_id": str(row.get("chapter_id", "") or "").strip(),
                "question": question,
                "answer": str(row.get("answer", "") or ""),
                "question_type": str(row.get("question_type", "fact") or "fact"),
                "difficulty": str(row.get("difficulty", "medium") or "medium"),
                "evidence_units": list(row.get("evidence_units", []) or []),
            }
        ]

    book_title = str(row.get("book_title", "") or "").strip()
    chapter_id = str(row.get("chapter_id", "") or "").strip()
    queries = []
    for idx, qa in enumerate(row.get("qas", []) or [], start=1):
        if qa.get("parse_failed", False):
            continue
        question = str(qa.get("question", "") or "").strip()
        if not question:
            continue
        queries.append(
            {
                "query_id": str(qa.get("qa_id", "") or f"{chapter_id}.qa_{idx}"),
                "book_title": book_title,
                "chapter_id": chapter_id,
                "question": question,
                "answer": str(qa.get("answer", "") or ""),
                "question_type": str(qa.get("question_type", "fact") or "fact"),
                "difficulty": str(qa.get("difficulty", "medium") or "medium"),
                "evidence_units": list(qa.get("evidence_units", []) or []),
            }
        )
    return queries


def load_queries(book: str | None = None) -> List[dict]:
    query_path = BOOK_QUERY_DIR / f"{book}.json" if book else QUERY_DIR / "queries_all.json"
    rows = read_json(query_path)
    queries: List[dict] = []
    total_rows = len(rows)
    for row_idx, row in enumerate(rows, start=1):
        queries.extend(normalize_query_row(row, row_idx))
        print_progress(row_idx, total_rows, prefix="Reading queries")
    if book:
        queries = [row for row in queries if row["book_title"] == book]
    return queries


def load_chunks(scheme: str, book: str | None = None) -> List[dict]:
    chunk_path = BOOK_CHUNK_DIR / f"{book}.{scheme}.json" if book else CHUNK_DIR / f"chunks_all.{scheme}.json"
    rows = read_json(chunk_path)
    total_rows = len(rows)
    chunks: List[dict] = []
    for row_idx, row in enumerate(rows, start=1):
        chunks.append(row)
        print_progress(row_idx, total_rows, prefix="Reading chunks")
    return chunks


def build_evidence_groups(query: dict, scheme: str) -> List[List[str]]:
    groups: List[List[str]] = []
    for eu in query.get("evidence_units", []) or []:
        chunk_ids = eu.get("fixed_ids") if scheme == "fixed" else eu.get("chunk_ids")
        normalized_ids = [str(chunk_id) for chunk_id in (chunk_ids or []) if chunk_id]
        if normalized_ids:
            groups.append(normalized_ids)
            continue
        chunk_id = eu.get("fixed_id") if scheme == "fixed" else eu.get("chunk_id")
        if chunk_id:
            groups.append([str(chunk_id)])
    return groups


def build_relevance(
    queries: List[dict],
    chunks: List[dict],
    scheme: str,
    evidence_group_fallback: str = "relaxed",
) -> Dict[str, List[dict]]:
    chunks_by_book: Dict[str, List[dict]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_book[chunk["book_title"]].append(chunk)

    results_by_book: Dict[str, List[dict]] = defaultdict(list)
    strict_zero = 0
    relaxed_zero = 0
    empty_evidence_groups = 0
    total_queries = len(queries)

    for query_idx, query in enumerate(queries, start=1):
        book_title = query["book_title"]
        chapter_id = query["chapter_id"]
        strict_ids = []
        relaxed_ids = []
        for chunk in chunks_by_book.get(book_title, []):
            covered_ids = chunk.get("covered_chapter_ids") or []
            if scheme == "booksum_split":
                if chapter_id in covered_ids:
                    strict_ids.append(chunk["chunk_id"])
                    relaxed_ids.append(chunk["chunk_id"])
            else:
                if chunk.get("active_chapter_id") == chapter_id:
                    strict_ids.append(chunk["chunk_id"])
                if chapter_id in covered_ids:
                    relaxed_ids.append(chunk["chunk_id"])

        evidence_groups = build_evidence_groups(query, scheme)
        evidence_group_source = "matched_evidence"
        if not evidence_groups and evidence_group_fallback != "none":
            fallback_ids = strict_ids if evidence_group_fallback == "strict" else relaxed_ids
            evidence_groups = [[chunk_id] for chunk_id in fallback_ids]
            evidence_group_source = f"chapter_{evidence_group_fallback}_fallback"
        if not evidence_groups:
            empty_evidence_groups += 1
        if not strict_ids:
            strict_zero += 1
        if not relaxed_ids:
            relaxed_zero += 1

        results_by_book[book_title].append(
            {
                "query_id": query["query_id"],
                "book_title": book_title,
                "chapter_id": chapter_id,
                "strict_relevant_chunk_ids": strict_ids,
                "relaxed_relevant_chunk_ids": relaxed_ids,
                "evidence_groups": evidence_groups,
                "evidence_group_source": evidence_group_source,
            }
        )
        print_progress(query_idx, total_queries, prefix="Building relevance")

    print(f"Queries with empty strict relevant chunks: {strict_zero}")
    print(f"Queries with empty relaxed relevant chunks: {relaxed_zero}")
    print(f"Queries with empty evidence_groups: {empty_evidence_groups}")
    return results_by_book


def main() -> None:
    parser = argparse.ArgumentParser(description="Build query-to-chunk relevance definitions; process all books if --book not specified")
    parser.add_argument("--scheme", type=str, choices=SUPPORTED_SCHEMES, default="booksum_split")
    parser.add_argument("--book", type=str, default=None, help="Book title; process all books if not specified")
    parser.add_argument(
        "--evidence_group_fallback",
        type=str,
        choices=["none", "strict", "relaxed"],
        default="relaxed",
        help="When summary evidence cannot verbatim-match chunks, use chapter-relevant chunks to build evidence_groups; relaxed recommended for summary QA",
    )
    args = parser.parse_args()

    ensure_dir(RELEVANCE_DIR)
    ensure_dir(BOOK_RELEVANCE_DIR)
    queries = load_queries(args.book)
    chunks = load_chunks(args.scheme, args.book)
    by_book = build_relevance(
        queries,
        chunks,
        scheme=args.scheme,
        evidence_group_fallback=args.evidence_group_fallback,
    )

    all_rows = []
    total_books = len(by_book)
    for book_idx, (book_title, rows) in enumerate(sorted(by_book.items()), start=1):
        write_json(BOOK_RELEVANCE_DIR / f"{book_title}.{args.scheme}.json", rows)
        all_rows.extend(rows)
        print_progress(book_idx, total_books, prefix="Writing relevance")
    if not args.book:
        write_json(RELEVANCE_DIR / f"relevance_all.{args.scheme}.json", all_rows)


if __name__ == "__main__":
    main()
