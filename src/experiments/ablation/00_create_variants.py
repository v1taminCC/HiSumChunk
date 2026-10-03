#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 3 ablation variants from _split.json and _L1Outline.json:
  - wo_h1:   remove H1 region summary slices, keep H2-prefixed content slices
  - wo_h2:   keep H1 region summary slices, strip H2 prefix from content slices
  - wo_h1h2: remove both H1 region summaries and H2 prefixes

Also copies queries, qa_drafts, and query embeddings to local directory.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    DATA_DIR,
    EMBEDDINGS_BY_BOOK_DIR,
    OURS_SPLIT_JSON_DIR,
    QA_DRAFTS_DIR,
    QUERIES_BY_BOOK_DIR,
    SOURCE_EMBEDDINGS_BY_BOOK_DIR,
    SOURCE_QA_DRAFTS_ALL,
    SOURCE_QUERIES_BY_BOOK_DIR,
    SOURCE_SPLITS_DIR,
    VARIANT_SPLITS_DIR,
    VARIANTS,
)

INFO_PATTERN = re.compile(r"Region\s+(\d+)---Block\s+(\d+)---Subchunk\s+(\d+)")
CHUNK_INFO_PATTERN = re.compile(r"Region\s+(\d+)---Block\s+(\d+)---Subchunk\s+(\d+)")
PUNCTUATIONS = ".!?"


def find_l1outline_file(split_file: Path) -> Path | None:
    stem = split_file.stem
    if stem.endswith("_split"):
        l1_stem = stem[:-6]  # remove "_split"
    else:
        return None
    l1_path = split_file.parent / f"{l1_stem}.json"
    return l1_path if l1_path.exists() else None


def build_h2_map(l1_data: dict) -> dict[tuple[int, int], str]:
    """Build (region_idx, chunk_local_index) -> h2_outline mapping from L1Outline.json"""
    h2_map = {}
    for region_idx, region in enumerate(l1_data.get("h2_regions", []), start=1):
        for chunk_idx, chunk in enumerate(region.get("chunks", []), start=1):
            h2 = chunk.get("h2_outline", "")
            if h2:
                if h2[-1] not in PUNCTUATIONS:
                    h2 = h2 + "."
            h2_map[(region_idx, chunk_idx)] = h2
    return h2_map


def strip_h2_prefix(text: str, h2_prefix: str) -> str:
    """Strip H2 prefix from text"""
    if not h2_prefix:
        return text
    if text.startswith(h2_prefix):
        return text[len(h2_prefix):]
    return text


def parse_chunk_info(chunk_info: str) -> tuple[int, int, int] | None:
    m = CHUNK_INFO_PATTERN.search(chunk_info)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def count_length_units(text: str) -> int:
    """Count English words + Chinese characters"""
    count = 0
    i = 0
    while i < len(text):
        ch = text[i]
        code = ord(ch)
        if (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or
            0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x2A6DF):
            count += 1
            i += 1
        elif ch.isascii() and (ch.isalpha() or ch.isdigit()):
            while i < len(text) and text[i].isascii() and (text[i].isalpha() or text[i].isdigit() or text[i] in "'-"):
                i += 1
            count += 1
        else:
            i += 1
    return count


def generate_variants(split_data: dict, h2_map: dict[tuple[int, int], str], book_stem: str):
    """Generate 3 ablation variants"""
    items = split_data.get("items", [])

    variants = {}

    # wo_h1: remove H1 region summary slices
    wo_h1_items = [item for item in items if not item.get("is_region_summary", False)]
    variants["wo_h1"] = wo_h1_items

    # wo_h2: keep H1 summaries, strip H2 prefix from content slices
    wo_h2_items = []
    for item in items:
        new_item = dict(item)
        if not item.get("is_region_summary", False):
            chunk_info = item.get("chunk_info", "")
            parsed = parse_chunk_info(chunk_info)
            if parsed:
                region_idx, chunk_idx, _ = parsed
                h2_prefix = h2_map.get((region_idx, chunk_idx), "")
                new_text = strip_h2_prefix(item["text"], h2_prefix)
                new_item["text"] = new_text
                new_item["length"] = count_length_units(new_text)
            # Remove original_length if present
            new_item.pop("original_length", None)
        wo_h2_items.append(new_item)
    variants["wo_h2"] = wo_h2_items

    # wo_h1h2: remove both H1 and H2
    wo_h1h2_items = []
    for item in items:
        if item.get("is_region_summary", False):
            continue
        new_item = dict(item)
        chunk_info = item.get("chunk_info", "")
        parsed = parse_chunk_info(chunk_info)
        if parsed:
            region_idx, chunk_idx, _ = parsed
            h2_prefix = h2_map.get((region_idx, chunk_idx), "")
            new_text = strip_h2_prefix(item["text"], h2_prefix)
            new_item["text"] = new_text
            new_item["length"] = count_length_units(new_text)
        new_item.pop("original_length", None)
        wo_h1h2_items.append(new_item)
    variants["wo_h1h2"] = wo_h1h2_items

    return variants


def main():
    print("=" * 60)
    print("Ablation Experiment - Step 0: Generate variant splits & copy dependencies")
    print("=" * 60)

    # 1. Copy queries
    print("\n[1/4] Copying queries...")
    QUERIES_BY_BOOK_DIR.mkdir(parents=True, exist_ok=True)
    query_files = sorted(SOURCE_QUERIES_BY_BOOK_DIR.glob("*.json"))
    for qf in query_files:
        shutil.copy2(qf, QUERIES_BY_BOOK_DIR / qf.name)
    print(f"  Copied {len(query_files)} query files")

    # 2. Copy qa_drafts
    print("\n[2/4] Copying qa_drafts...")
    QA_DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_QA_DRAFTS_ALL.exists():
        shutil.copy2(SOURCE_QA_DRAFTS_ALL, QA_DRAFTS_DIR / "qa_drafts_all.json")
        print(f"  Copied qa_drafts_all.json")

    # 3. Copy query embeddings
    print("\n[3/4] Copying query embeddings...")
    EMBEDDINGS_BY_BOOK_DIR.mkdir(parents=True, exist_ok=True)
    query_emb_files = sorted(SOURCE_EMBEDDINGS_BY_BOOK_DIR.glob("*.queries.npy")) + \
                      sorted(SOURCE_EMBEDDINGS_BY_BOOK_DIR.glob("*.queries.meta.json"))
    for ef in query_emb_files:
        shutil.copy2(ef, EMBEDDINGS_BY_BOOK_DIR / ef.name)
    print(f"  Copied {len(query_emb_files)} query embedding files")

    # 4. Generate variant splits
    print("\n[4/4] Generating ablation variant splits...")
    SOURCE_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS:
        (VARIANT_SPLITS_DIR / v).mkdir(parents=True, exist_ok=True)

    split_files = sorted(OURS_SPLIT_JSON_DIR.glob("*_split.json"))
    print(f"  Found {len(split_files)} _split.json files")

    total_stats = {v: {"books": 0, "items": 0} for v in VARIANTS}

    for fi, split_fp in enumerate(split_files, 1):
        book_stem = split_fp.stem.replace("_split", "")

        # Copy original split to source_splits
        shutil.copy2(split_fp, SOURCE_SPLITS_DIR / split_fp.name)

        # Load split data
        with open(split_fp, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        # Load L1Outline file for h2_outline
        l1_fp = find_l1outline_file(split_fp)
        if l1_fp:
            with open(l1_fp, "r", encoding="utf-8") as f:
                l1_data = json.load(f)
            h2_map = build_h2_map(l1_data)
        else:
            print(f"  [WARNING] L1Outline file not found: {book_stem}")
            h2_map = {}

        # Generate variants
        variants = generate_variants(split_data, h2_map, book_stem)

        for variant_name, variant_items in variants.items():
            lengths = [item.get("length", 0) for item in variant_items]
            out_data = {
                "split_method": "HiSumChunk",
                "model": "Qwen3-4B-Instruct-2507",
                "variant": variant_name,
                "max_length": max(lengths) if lengths else 0,
                "min_length": min(lengths) if lengths else 0,
                "items": variant_items,
            }
            out_path = VARIANT_SPLITS_DIR / variant_name / f"{book_stem}_split.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)
            total_stats[variant_name]["books"] += 1
            total_stats[variant_name]["items"] += len(variant_items)

        print(f"  [{fi}/{len(split_files)}] {split_fp.stem[:50]}...")

    print("\n" + "=" * 60)
    print("Variant generation summary:")
    for v in VARIANTS:
        print(f"  {v}: {total_stats[v]['books']} books, {total_stats[v]['items']} chunks")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
