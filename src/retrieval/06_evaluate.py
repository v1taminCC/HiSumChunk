#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

from config import BOOK_RELEVANCE_DIR, GLOBAL_RUN_DIR, IN_BOOK_RUN_DIR, TABLES_DIR, SUPPORTED_SCHEMES
from utils_book import list_book_titles
from utils_io import ensure_dir, read_json, write_json
from utils_metrics import EVAL_KS, evaluate_query
from utils_query_consistency import SUPPORTED_QUERY_VARIANTS, query_variant_suffix, validate_run_query_variant, validate_run_vs_relevance


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    total = max(total, 1)
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix}: [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


def summarize(rows: list[dict], key_prefix: str, num_queries: int) -> dict:
    summary = {"mrr": sum(row[f"{key_prefix}_mrr"] for row in rows) / num_queries if num_queries else 0.0}
    for k in EVAL_KS:
        summary[f"hit@{k}"] = sum(int(row[f"{key_prefix}_hit@{k}"] > 0) for row in rows) / num_queries if num_queries else 0.0
    for k in EVAL_KS:
        summary[f"evidence_recall@{k}"] = sum(row[f"evidence_recall@{k}"] for row in rows) / num_queries if num_queries else 0.0
    summary["num_queries"] = num_queries
    return summary


def evaluate_run(rows: list[dict], relevance_rows: list[dict]) -> dict:
    rel_map = {row["query_id"]: row for row in relevance_rows}
    flat_rows = []
    per_query = []
    missing = 0
    total_rows = len(rows)
    for row_idx, row in enumerate(rows, start=1):
        rel = rel_map.get(row["query_id"])
        if not rel:
            missing += 1
            print_progress(row_idx, total_rows, prefix="Evaluating queries")
            continue
        ranked_chunk_ids = [x["chunk_id"] for x in row.get("top_k", [])]
        metrics = evaluate_query(
            ranked_chunk_ids,
            set(rel.get("strict_relevant_chunk_ids", [])),
            set(rel.get("relaxed_relevant_chunk_ids", [])),
            rel.get("evidence_groups", []),
        )
        per_query.append({"query_id": row["query_id"], **metrics})
        flat_rows.append(metrics)
        print_progress(row_idx, total_rows, prefix="Evaluating queries")
    return {
        "summary": {"num_queries": len(rows), "evaluated_queries": len(flat_rows), "missing_relevance_queries": missing},
        "strict": summarize(flat_rows, "strict", len(flat_rows)),
        "relaxed": summarize(flat_rows, "relaxed", len(flat_rows)),
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA retrieval results; evaluate all books if --book not specified in in_book mode")
    parser.add_argument("--scheme", type=str, choices=SUPPORTED_SCHEMES, default="booksum_split")
    parser.add_argument("--mode", type=str, choices=["in_book", "global"], default="in_book")
    parser.add_argument("--book", type=str, default=None, help="Book title; evaluate all books if not specified in in_book mode")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--query_variant", type=str, choices=SUPPORTED_QUERY_VARIANTS, default="default", help="default=question field; en=retrieval_question_en; default_en=question + retrieval_question_en")
    args = parser.parse_args()
    suffix = query_variant_suffix(args.query_variant)

    ensure_dir(TABLES_DIR)
    if args.mode == "in_book":
        books = [args.book] if args.book else list_book_titles()
        total_evaluated = 0
        for book_idx, book in enumerate(books, start=1):
            print(f"\n[{book_idx}/{len(books)}] Evaluating: {book}", flush=True)
            run_rows = read_json(IN_BOOK_RUN_DIR / f"{book}.{args.scheme}{suffix}.top{args.top_k}.json")
            validate_run_query_variant(run_rows, args.query_variant, source=f"in_book/{book}/{args.scheme}")
            relevance_rows = read_json(BOOK_RELEVANCE_DIR / f"{book}.{args.scheme}.json")
            validate_run_vs_relevance(run_rows, relevance_rows, book=book, scheme=args.scheme)
            result = evaluate_run(run_rows, relevance_rows)
            result["query_variant"] = args.query_variant
            output = TABLES_DIR / f"in_book_{book}.{args.scheme}{suffix}.json"
            write_json(output, result)
            total_evaluated += result["summary"]["evaluated_queries"]
            print(f"Written: {output}")
        print(f"Evaluated queries: {total_evaluated}")
    else:
        from config import RELEVANCE_DIR
        run_rows = read_json(GLOBAL_RUN_DIR / f"all_books.{args.scheme}{suffix}.top{args.top_k}.json")
        validate_run_query_variant(run_rows, args.query_variant, source=f"global/{args.scheme}")
        relevance_rows = read_json(RELEVANCE_DIR / f"relevance_all.{args.scheme}.json")
        result = evaluate_run(run_rows, relevance_rows)
        result["query_variant"] = args.query_variant
        output = TABLES_DIR / f"global_all.{args.scheme}{suffix}.json"
        write_json(output, result)
        print(f"Written: {output}")


if __name__ == "__main__":
    main()
