#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stratified random sampling of chapters from all source chapters, ensuring each book gets at least some chapters.
Supports incremental sampling.

Usage:
  python3 00b_sample_chapters.py --count 100
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

try:
    from config import SOURCE_CHAPTER_DIR, QA_DRAFT_DIR
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import SOURCE_CHAPTER_DIR, QA_DRAFT_DIR

SAMPLING_STATE_FILE = QA_DRAFT_DIR / "qa_sampling_state.json"
CURRENT_SAMPLE_FILE = QA_DRAFT_DIR / "current_sampled_chapters.json"

DEFAULT_COUNT = 300
SEED = 42


def load_all_chapters() -> list[dict]:
    path = SOURCE_CHAPTER_DIR / "source_chapters_all.json"
    if not path.exists():
        raise FileNotFoundError(f"Please run 00_prepare_source_chapters_from_booksum_summary.py first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_state() -> dict:
    if SAMPLING_STATE_FILE.exists():
        return json.loads(SAMPLING_STATE_FILE.read_text(encoding="utf-8"))
    return {"sampled_ids": [], "generated_ids": [], "total_available": 0}


def save_state(state: dict) -> None:
    SAMPLING_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLING_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def stratified_sample(all_chapters: list[dict], sampled_set: set[str], count: int) -> list[dict]:
    """Stratified sampling: each book gets at least floor(count/n_books) chapters, remaining quota allocated proportionally."""
    by_book = defaultdict(list)
    for ch in all_chapters:
        if ch["chapter_id"] not in sampled_set:
            by_book[ch["book_title"]].append(ch)

    if not by_book:
        return []

    n_books = len(by_book)
    floor = max(1, count // n_books)
    actual_floor = min(floor, min(len(v) for v in by_book.values()))

    selected = []
    remainder_pool = []
    for book, chapters in by_book.items():
        shuffled = random.sample(chapters, len(chapters))
        # Take floor count per book
        for ch in shuffled[:actual_floor]:
            selected.append(ch)
        # Put remainder into candidate pool
        for ch in shuffled[actual_floor:]:
            remainder_pool.append(ch)

    # Randomly fill from candidate pool to count
    needed = count - len(selected)
    if needed > 0 and remainder_pool:
        extra = random.sample(remainder_pool, min(needed, len(remainder_pool)))
        selected.extend(extra)

    return selected


def main():
    parser = argparse.ArgumentParser(description="Stratified random chapter sampling for QA generation")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Sample count for this run")
    args = parser.parse_args()

    all_chapters = load_all_chapters()
    state = load_state()
    state["total_available"] = len(all_chapters)
    sampled_set = set(state["sampled_ids"])

    # Count total chapters per book
    by_book = defaultdict(list)
    for ch in all_chapters:
        by_book[ch["book_title"]].append(ch)
    print(f"Total: {len(all_chapters)} chapters, {len(by_book)} books")
    for bk, chs in sorted(by_book.items()):
        print(f"  {bk}: {len(chs)} chapters")

    random.seed(SEED)
    selected = stratified_sample(all_chapters, sampled_set, args.count)
    n_sample = len(selected)

    if n_sample == 0:
        print(f"\nAll chapters sampled. Sampled: {len(sampled_set)}")
        return

    # Write this sampling batch
    CURRENT_SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_SAMPLE_FILE.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    # Update state
    state["sampled_ids"] = sorted(sampled_set | {ch["chapter_id"] for ch in selected})
    state["last_sampled_at"] = datetime.now(timezone.utc).isoformat()
    state["last_count"] = n_sample
    save_state(state)

    # Statistics by book
    books_dist = defaultdict(int)
    for ch in selected:
        books_dist[ch["book_title"]] += 1
    print(f"\nSampled: {n_sample} chapters (target {args.count})")
    print(f"Total sampled: {len(state['sampled_ids'])} / {len(all_chapters)}")
    print(f"Across {len(books_dist)} books, distribution:")
    for bk, cnt in sorted(books_dist.items()):
        total = len(by_book[bk])
        print(f"  {bk}: {cnt}/{total} chapters")
    print(f"\nSampling output: {CURRENT_SAMPLE_FILE}")


if __name__ == "__main__":
    main()
