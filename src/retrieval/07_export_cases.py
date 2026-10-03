#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

from config import BOOK_CHUNK_DIR, BOOK_QUERY_DIR, BOOK_RELEVANCE_DIR, CASES_DIR, GLOBAL_RUN_DIR, IN_BOOK_RUN_DIR, SUPPORTED_SCHEMES
from utils_book import list_book_titles
from utils_io import ensure_dir, read_json, write_json
from utils_query_consistency import SUPPORTED_QUERY_VARIANTS, query_variant_suffix


def export_one_book(scheme: str, book: str, top_k: int, limit: int, query_variant: str) -> int:
    suffix = query_variant_suffix(query_variant)
    query_rows = read_json(BOOK_QUERY_DIR / f"{book}.json")
    chunk_rows = read_json(BOOK_CHUNK_DIR / f"{book}.{scheme}.json")
    relevance_rows = read_json(BOOK_RELEVANCE_DIR / f"{book}.{scheme}.json")
    run_rows = read_json(IN_BOOK_RUN_DIR / f"{book}.{scheme}{suffix}.top{top_k}.json")
    output = CASES_DIR / f"{book}.{scheme}{suffix}.cases.json"
    rows = build_cases(query_rows, chunk_rows, relevance_rows, run_rows, limit)
    write_json(output, rows)
    print(f"Written: {output} | cases: {len(rows)}")
    return len(rows)


def flatten_queries(query_rows: list[dict]) -> dict[str, dict]:
    query_map: dict[str, dict] = {}
    for row in query_rows:
        if "query_id" in row:
            query_map[str(row["query_id"])] = row
            continue
        book_title = str(row.get("book_title", "") or "")
        chapter_id = str(row.get("chapter_id", "") or "")
        for qa in row.get("qas", []) or []:
            if qa.get("parse_failed", False):
                continue
            query_id = str(qa.get("qa_id", "") or "")
            if not query_id:
                continue
            query_map[query_id] = {
                "query_id": query_id,
                "book_title": book_title,
                "chapter_id": chapter_id,
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "question_type": qa.get("question_type", ""),
                "difficulty": qa.get("difficulty", ""),
                "evidence_units": qa.get("evidence_units", []),
                "retrieval_question_en": qa.get("retrieval_question_en", ""),
            }
    return query_map


def build_cases(query_rows: list[dict], chunk_rows: list[dict], relevance_rows: list[dict], run_rows: list[dict], limit: int) -> list[dict]:
    query_map = flatten_queries(query_rows)
    chunk_map = {x["chunk_id"]: x for x in chunk_rows}
    rel_map = {x["query_id"]: x for x in relevance_rows}
    rows = []
    for row in run_rows[:limit]:
        rel = rel_map.get(row["query_id"])
        if not rel:
            continue
        rows.append(
            {
                "query": query_map.get(row["query_id"], {}),
                "relevance": rel,
                "top_chunks": [
                    {
                        **item,
                        "chunk": chunk_map.get(item["chunk_id"], {}),
                    }
                    for item in row.get("top_k", [])[:5]
                ],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export QA retrieval cases; export all books if --book not specified")
    parser.add_argument("--scheme", type=str, choices=SUPPORTED_SCHEMES, required=True)
    parser.add_argument("--mode", type=str, choices=["in_book", "global"], required=True)
    parser.add_argument("--book", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--query_variant", type=str, choices=SUPPORTED_QUERY_VARIANTS, default="default", help="default=question field; en=retrieval_question_en; default_en=question + retrieval_question_en")
    args = parser.parse_args()
    suffix = query_variant_suffix(args.query_variant)

    ensure_dir(CASES_DIR)
    if args.mode == "in_book":
        books = [args.book] if args.book else list_book_titles()
        total = 0
        for book_idx, book in enumerate(books, start=1):
            print(f"\n[{book_idx}/{len(books)}] Exporting cases: {book}", flush=True)
            total += export_one_book(args.scheme, book, args.top_k, args.limit, args.query_variant)
        print(f"Total cases: {total}")
    else:
        from config import CHUNK_DIR, QUERY_DIR, RELEVANCE_DIR
        query_rows = read_json(QUERY_DIR / "queries_all.json")
        chunk_rows = read_json(CHUNK_DIR / f"chunks_all.{args.scheme}.json")
        relevance_rows = read_json(RELEVANCE_DIR / f"relevance_all.{args.scheme}.json")
        run_rows = read_json(GLOBAL_RUN_DIR / f"all_books.{args.scheme}{suffix}.top{args.top_k}.json")
        output = CASES_DIR / f"global_all.{args.scheme}{suffix}.cases.json"
        rows = build_cases(query_rows, chunk_rows, relevance_rows, run_rows, args.limit)
        write_json(output, rows)
        print(f"cases: {len(rows)}")


if __name__ == "__main__":
    main()
