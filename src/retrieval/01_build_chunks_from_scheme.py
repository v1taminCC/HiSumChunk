#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    BOOK_CHUNK_DIR, BOOKSUM_SPLIT_JSON_DIR, CHUNK_DIR,
    FIXED_SPLIT_JSON_DIR, FIXED_SPLIT_SUFFIX,
    LUMBER_SPLIT_JSON_DIR, LBD_SPLIT_JSON_DIR, SEMANTIC_SPLIT_JSON_DIR,
    RAPTOR_SPLIT_JSON_DIR, RAPTOR_SPLIT_SUFFIX,
    LUMBER_SPLIT_SUFFIX, LBD_SPLIT_SUFFIX, SEMANTIC_SPLIT_SUFFIX,
    SPLIT_GLOB, SUPPORTED_SCHEMES,
)
from utils_book import (
    extract_book_title,
    find_booksum_split_file, find_fixed_split_file,
    find_lumber_split_file, find_lbd_split_file, find_semantic_split_file,
    find_raptor_nodes_file,
    list_book_titles, strip_split_suffix,
)
from utils_io import ensure_dir, read_json, write_json


EXCLUDE_SUFFIXES = {
    ".anchors.json",
    ".checkpoint.json",
}


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    total = max(total, 1)
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix}: [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


def infer_book_title_from_item(item: dict, source_file: Path) -> str:
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
    return strip_split_suffix(source_file.name)


def iter_split_files(scheme: str, book: str | None) -> list[Path]:
    if book:
        return [_find_one(scheme, book)]
    # All books: iterate corresponding directory by scheme
    dir_map = {
        "fixed": FIXED_SPLIT_JSON_DIR,
        "lumberchunker": LUMBER_SPLIT_JSON_DIR,
        "lbdsemantic": LBD_SPLIT_JSON_DIR,
        "semanticembed": SEMANTIC_SPLIT_JSON_DIR,
        "raptor": RAPTOR_SPLIT_JSON_DIR,
    }
    if scheme in dir_map:
        suffix_map = {
            "fixed": FIXED_SPLIT_SUFFIX,
            "lumberchunker": LUMBER_SPLIT_SUFFIX,
            "lbdsemantic": LBD_SPLIT_SUFFIX,
            "semanticembed": SEMANTIC_SPLIT_SUFFIX,
            "raptor": RAPTOR_SPLIT_SUFFIX,
        }
        d = dir_map[scheme]; sfx = suffix_map[scheme]
        return sorted(p for p in d.glob(f"*{sfx}") if p.is_file())
    # ours / booksum_split
    return sorted(p for p in BOOKSUM_SPLIT_JSON_DIR.glob(SPLIT_GLOB) if p.is_file())


def _find_one(scheme: str, book: str) -> Path:
    finders = {
        "fixed": find_fixed_split_file,
        "lumberchunker": find_lumber_split_file,
        "lbdsemantic": find_lbd_split_file,
        "semanticembed": find_semantic_split_file,
        "raptor": find_raptor_nodes_file,
    }
    return finders.get(scheme, find_booksum_split_file)(book)


def load_split_chunks(scheme: str, only_book: str | None = None) -> list[dict]:
    all_rows: list[dict] = []
    split_files = iter_split_files(scheme, only_book)
    total_files = len(split_files)
    chunk_prefix = {
        "fixed": "fixed_chunk",
        "lumberchunker": "lumber_chunk",
        "lbdsemantic": "lbd_chunk",
        "semanticembed": "semantic_chunk",
    }.get(scheme, "split_chunk")

    for file_idx, fp in enumerate(split_files, start=1):
        data = read_json(fp)
        items = data.get("items", []) or []
        by_book_count: dict[str, int] = {}
        for item in items:
            book_title = infer_book_title_from_item(item, fp)
            if only_book and book_title != only_book:
                continue
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            by_book_count[book_title] = by_book_count.get(book_title, 0) + 1
            all_rows.append(
                {
                    "chunk_id": f"{book_title}::{chunk_prefix}::{by_book_count[book_title]}",
                    "book_title": book_title,
                    "text": text,
                    "active_chapter_id": str(item.get("active_chapter_id", "") or "").strip(),
                    "covered_chapter_ids": list(item.get("covered_chapter_ids", []) or []),
                    "crosses_chapter": bool(item.get("crosses_chapter", False)),
                    "is_outline": bool(item.get("is_region_summary", False)),
                    "section_region": str(item.get("region_id", "") or ""),
                    "info": str(item.get("chunk_info", "") or ""),
                    "source_file": fp.name,
                    "source_start": item.get("source_start"),
                    "source_end": item.get("source_end"),
                    "length": item.get("length"),
                    "source_scheme": scheme,
                }
            )
        print_progress(file_idx, total_files, prefix="Reading split")
    return all_rows


def load_raptor_nodes_chunks(only_book: str | None = None) -> list[dict]:
    all_rows: list[dict] = []
    node_files = iter_split_files("raptor", only_book)
    total_files = len(node_files)
    for file_idx, fp in enumerate(node_files, start=1):
        data = read_json(fp)
        nodes = data.get("nodes", []) or []
        book_title = data.get("book_title") or fp.name.replace(RAPTOR_SPLIT_SUFFIX, "")
        for node in nodes:
            raw_text = str(node.get("text", "") or "").strip()
            if not raw_text:
                continue
            node_type = str(node.get("node_type", "leaf") or "leaf")
            level = int(node.get("level", 0) or 0)
            prefix = "[RAPTOR Leaf | Level 0]" if node_type == "leaf" else f"[RAPTOR Summary | Level {level}]"
            all_rows.append(
                {
                    "chunk_id": f"{book_title}::raptor_node::{node.get('node_id', len(all_rows) + 1)}",
                    "book_title": book_title,
                    "text": f"{prefix}\n{raw_text}",
                    "active_chapter_id": str(node.get("active_chapter_id", "") or "").strip(),
                    "covered_chapter_ids": list(node.get("covered_chapter_ids", []) or []),
                    "crosses_chapter": bool(node.get("crosses_chapter", False)),
                    "is_outline": False,
                    "section_region": "",
                    "info": f"RAPTOR L{level} {node_type}",
                    "source_file": fp.name,
                    "source_start": node.get("source_start"),
                    "source_end": node.get("source_end"),
                    "length": node.get("length"),
                    "source_scheme": "raptor",
                    "node_type": node_type,
                    "level": level,
                    "child_node_ids": list(node.get("child_node_ids", []) or node.get("children", []) or []),
                    "source_chunk_ids": list(node.get("source_chunk_ids", []) or []),
                }
            )
        print_progress(file_idx, total_files, prefix="Reading RAPTOR nodes")
    return all_rows


def load_existing_scheme_chunks(scheme: str, only_book: str | None, include_outline: bool) -> list[dict]:
    source_dir = SCHEME_OURS_DIR if scheme == "ours" else SCHEME_FIXED_DIR
    input_dir = source_dir / "data" / "chunks" / "by_book"
    rows: list[dict] = []
    files = sorted(
        p for p in input_dir.glob("*.json")
        if p.is_file() and (include_outline or not p.stem.endswith("_h1_outline"))
    )
    total_files = len(files)
    for file_idx, fp in enumerate(files, start=1):
        book_title = fp.stem.replace("_h1_outline", "")
        if only_book and book_title != only_book:
            print_progress(file_idx, total_files, prefix="Reading chunks")
            continue
        part_rows = read_json(fp)
        if not include_outline:
            part_rows = [row for row in part_rows if not bool(row.get("is_outline", False))]
        rows.extend([{**row, "source_scheme": scheme} for row in part_rows])
        print_progress(file_idx, total_files, prefix="Reading chunks")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunks from a split scheme; process all books if --book not specified")
    parser.add_argument("--scheme", type=str, choices=SUPPORTED_SCHEMES, default="booksum_split")
    parser.add_argument("--book", type=str, default=None, help="Book title; process all books if not specified")
    parser.add_argument("--include_outline", action="store_true")
    args = parser.parse_args()

    ensure_dir(CHUNK_DIR)
    ensure_dir(BOOK_CHUNK_DIR)

    if args.scheme == "raptor":
        rows = load_raptor_nodes_chunks(only_book=args.book)
    elif args.scheme in {"booksum_split", "fixed", "ours", "lumberchunker", "lbdsemantic", "semanticembed"}:
        rows = load_split_chunks(args.scheme, only_book=args.book)
    else:
        rows = load_existing_scheme_chunks(args.scheme, args.book, args.include_outline)

    by_book: dict[str, list[dict]] = {}
    for row in rows:
        by_book.setdefault(row["book_title"], []).append(row)

    all_rows = []
    total_books = len(by_book)
    for book_idx, (book_title, book_rows) in enumerate(sorted(by_book.items()), start=1):
        write_json(BOOK_CHUNK_DIR / f"{book_title}.{args.scheme}.json", book_rows)
        all_rows.extend(book_rows)
        print_progress(book_idx, total_books, prefix="Writing chunks")

    if not args.book:
        write_json(CHUNK_DIR / f"chunks_all.{args.scheme}.json", all_rows)
    print(f"Chunk count: {len(all_rows)}")


if __name__ == "__main__":
    main()
