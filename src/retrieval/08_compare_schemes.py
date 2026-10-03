#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

from config import REPORTS_DIR, TABLES_DIR
from utils_book import list_book_titles
from utils_io import ensure_dir, read_json, write_json
from utils_query_consistency import SUPPORTED_QUERY_VARIANTS, query_variant_suffix


KEYS = [
    "hit@1", "hit@3", "hit@5", "hit@10",
    "mrr",
    "evidence_recall@1", "evidence_recall@3", "evidence_recall@5", "evidence_recall@10",
]


def build_report(ours: dict, fixed: dict) -> dict:
    report = {"strict": {}, "relaxed": {}}
    for setting in ["strict", "relaxed"]:
        for key in KEYS:
            report[setting][key] = {
                "ours": ours[setting].get(key, 0.0),
                "Fixed": fixed[setting].get(key, 0.0),
                "delta": ours[setting].get(key, 0.0) - fixed[setting].get(key, 0.0),
            }
    return report


def average_reports(reports: list[dict]) -> dict:
    avg_report = {"strict": {}, "relaxed": {}}
    count = len(reports)
    for setting in ["strict", "relaxed"]:
        for key in KEYS:
            ours_avg = sum(report[setting][key]["ours"] for report in reports) / count if count else 0.0
            fixed_avg = sum(report[setting][key]["Fixed"] for report in reports) / count if count else 0.0
            avg_report[setting][key] = {
                "ours": ours_avg,
                "Fixed": fixed_avg,
                "delta": ours_avg - fixed_avg,
            }
    avg_report["num_books"] = count
    return avg_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ours vs fixed QA retrieval results")
    parser.add_argument("--scope", type=str, choices=["global", "in_book"], default="global")
    parser.add_argument("--book", type=str, default=None, help="Book title for in_book; compare all books if not specified")
    parser.add_argument("--query_variant", type=str, choices=SUPPORTED_QUERY_VARIANTS, default="default", help="default=question field; en=retrieval_question_en; default_en=question + retrieval_question_en")
    args = parser.parse_args()
    suffix = query_variant_suffix(args.query_variant)

    ensure_dir(REPORTS_DIR)
    if args.scope == "global":
        ours = read_json(TABLES_DIR / f"global_all.ours{suffix}.json")
        fixed = read_json(TABLES_DIR / f"global_all.fixed{suffix}.json")
        report = build_report(ours, fixed)
        report["query_variant"] = args.query_variant
        output = REPORTS_DIR / f"compare_ours_vs_fixed.global{suffix}.json"
        write_json(output, report)
        print(f"Comparison report written: {output}")
        return

    books = [args.book] if args.book else list_book_titles()
    all_reports = {}
    collected_reports = []
    for book_idx, book in enumerate(books, start=1):
        print(f"\n[{book_idx}/{len(books)}] Comparing: {book}", flush=True)
        ours = read_json(TABLES_DIR / f"in_book_{book}.ours{suffix}.json")
        fixed = read_json(TABLES_DIR / f"in_book_{book}.fixed{suffix}.json")
        report = build_report(ours, fixed)
        report["query_variant"] = args.query_variant
        output = REPORTS_DIR / f"compare_ours_vs_fixed.in_book_{book}{suffix}.json"
        write_json(output, report)
        all_reports[book] = report
        collected_reports.append(report)
        print(f"Comparison report written: {output}")

    if not args.book:
        output = REPORTS_DIR / f"compare_ours_vs_fixed.in_book_all_by_book{suffix}.json"
        write_json(output, all_reports)
        print(f"Per-book summary written: {output}")

        avg_output = REPORTS_DIR / f"compare_ours_vs_fixed.in_book_macro_avg{suffix}.json"
        avg_report = average_reports(collected_reports)
        write_json(avg_output, avg_report)
        print(f"Per-book macro average written: {avg_output}")


if __name__ == "__main__":
    main()
