#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert variant splits to chunk format with assigned chunk_ids.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import CHUNKS_BY_BOOK_DIR, EXAMPLE_BOOK_DIR, VARIANT_SPLITS_DIR, VARIANTS
from utils_io import ensure_dir, read_json, write_json


def extract_book_title(chapter_id: str) -> str:
    """Extract book title from chapter_id"""
    if "::" in chapter_id:
        return chapter_id.split("::")[0]
    return ""


def infer_book_title_from_item(item: dict, source_file: Path) -> str:
    """Infer book title from item"""
    active = str(item.get("active_chapter_id", "") or "").strip()
    if active:
        title = extract_book_title(active)
        if title:
            return title
    covered = item.get("covered_chapter_ids", []) or []
    if covered:
        title = extract_book_title(covered[0])
        if title:
            return title
    stem = source_file.stem
    if stem.endswith("_split"):
        stem = stem[:-6]
    parts = stem.split("_MergeRule")
    return parts[0] if parts else stem


def build_chunks_for_variant(variant: str) -> dict[str, list[dict]]:
    """Build chunks for a given variant"""
    variant_dir = VARIANT_SPLITS_DIR / variant
    split_files = sorted(variant_dir.glob("*_split.json"))
    if not split_files:
        example_variant_dir = EXAMPLE_BOOK_DIR / "ablation" / "variant_splits" / variant
        split_files = sorted(example_variant_dir.glob("*_split.json"))

    by_book: dict[str, list[dict]] = {}
    chunk_prefix = f"{variant}_chunk"

    for fp in split_files:
        data = read_json(fp)
        items = data.get("items", [])

        book_count: dict[str, int] = {}
        for item in items:
            book_title = infer_book_title_from_item(item, fp)
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue

            book_count[book_title] = book_count.get(book_title, 0) + 1
            chunk = {
                "chunk_id": f"{book_title}::{chunk_prefix}::{book_count[book_title]}",
                "book_title": book_title,
                "text": text,
                "active_chapter_id": str(item.get("active_chapter_id", "") or "").strip(),
                "covered_chapter_ids": list(item.get("covered_chapter_ids", []) or []),
                "crosses_chapter": bool(item.get("crosses_chapter", False)),
            "is_outline": bool(item.get("is_region_summary", False)),
            "section_region": str(item.get("region_id", "") or ""),
                "info": str(item.get("info", "") or ""),
                "source_file": fp.name,
                "length": item.get("length"),
                "source_variant": variant,
            }
            by_book.setdefault(book_title, []).append(chunk)

    return by_book


def main():
    print("=" * 60)
    print("Ablation Experiment - Step 1: Build chunks")
    print("=" * 60)

    ensure_dir(CHUNKS_BY_BOOK_DIR)

    total_stats = {v: {"books": 0, "chunks": 0} for v in VARIANTS}

    for variant in VARIANTS:
        print(f"\nProcessing variant: {variant}")
        by_book = build_chunks_for_variant(variant)

        for book_title, chunks in sorted(by_book.items()):
            out_path = CHUNKS_BY_BOOK_DIR / f"{book_title}.{variant}.json"
            write_json(out_path, chunks)
            total_stats[variant]["books"] += 1
            total_stats[variant]["chunks"] += len(chunks)

        print(f"  {total_stats[variant]['books']} books, {total_stats[variant]['chunks']} chunks")

    print("\n" + "=" * 60)
    print("Chunk build summary:")
    for v in VARIANTS:
        print(f"  {v}: {total_stats[v]['books']} books, {total_stats[v]['chunks']} chunks")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
