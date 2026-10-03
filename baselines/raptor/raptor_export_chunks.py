#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
WORK_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))
EXAMPLE_BOOK_DIR = WORK_BOOK_DIR
"""
RAPTOR Chunks Export
====================
Export RAPTOR nodes into the unified chunk format used by the retrieval pipeline.

Input:  example_data/Far From the Madding Crowd/*_RAPTOR_nodes.json
output:  .../data/chunks/by_book/{BookTitle}.raptor.json
       .../data/chunks/chunks_all.raptor.json
"""

# ================================================================================================
# ==================================== Parameters ====================================================
# ================================================================================================

INPUT_DIR: str = str(EXAMPLE_BOOK_DIR)
NODES_SUFFIX: str = "_RAPTOR_nodes.json"
OUTPUT_CHUNKS_DIR: str = str(REPO_ROOT / "src" / "retrieval" / "data" / "chunks" / "by_book")
OUTPUT_ALL_CHUNKS_DIR: str = str(REPO_ROOT / "src" / "retrieval" / "data" / "chunks")
SCHEME: str = "raptor"
LEAF_PREFIX: str = "[RAPTOR Leaf | Level 0]"
SUMMARY_PREFIX_TEMPLATE: str = "[RAPTOR Summary | Level {level}]"
BOOK_FILTER: str = None

# ================================================================================================
# ========================================== Implementation ============================================
# ================================================================================================

import argparse
import json
import sys
from typing import Any, Dict, List


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp_path), str(path))


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def node_to_chunk(node: Dict[str, Any]) -> Dict[str, Any]:
    node_type = node.get("node_type", "leaf")
    level = node.get("level", 0)
    raw_text = str(node.get("text", "") or "").strip()
    prefix = SUMMARY_PREFIX_TEMPLATE.format(level=level) if node_type == "summary" else LEAF_PREFIX
    text = f"{prefix}\n{raw_text}"
    chunk_id = f"{node['book_title']}::raptor_node::{node['node_id']}"
    return {
        "chunk_id": chunk_id,
        "book_title": node.get("book_title", ""),
        "text": text,
        "active_chapter_id": str(node.get("active_chapter_id", "") or "").strip(),
        "covered_chapter_ids": list(node.get("covered_chapter_ids", []) or []),
        "crosses_chapter": bool(node.get("crosses_chapter", False)),
        "is_outline": False,
        "section_region": "",
        "info": f"RAPTOR L{level} {node_type}",
        "source_file": f"{node.get('book_title', '')}_RAPTOR_tree.json",
        "source_start": node.get("source_start"),
        "source_end": node.get("source_end"),
        "length": len(raw_text.split()),
        "source_scheme": SCHEME,
        "node_type": node_type,
        "level": level,
        "child_node_ids": list(node.get("child_node_ids", []) or node.get("children", []) or []),
        "source_chunk_ids": list(node.get("source_chunk_ids", []) or []),
    }


def export_one_book(nodes_path: Path) -> Dict[str, Any]:
    data = read_json(nodes_path)
    book_title = data.get("book_title", nodes_path.name.replace(NODES_SUFFIX, ""))
    nodes = data.get("nodes", [])
    chunks = [node_to_chunk(n) for n in nodes if n.get("text", "").strip()]
    leaf_count = sum(1 for c in chunks if c["node_type"] == "leaf")
    summary_count = sum(1 for c in chunks if c["node_type"] == "summary")

    out_path = Path(OUTPUT_CHUNKS_DIR) / f"{book_title}.{SCHEME}.json"
    atomic_write_json(out_path, chunks)

    return {
        "book_title": book_title, "total_chunks": len(chunks),
        "leaf_chunks": leaf_count, "summary_chunks": summary_count,
        "output": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAPTOR chunk export")
    parser.add_argument("--book", type=str, default=None, help="process only one book; do not write chunks_all by default")
    parser.add_argument("--write-all", action="store_true", help="explicitly write chunks_all.raptor.json")
    args = parser.parse_args()

    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"directory does not exist: {INPUT_DIR}")
        sys.exit(1)

    is_single_book = bool(args.book or BOOK_FILTER)

    if args.book:
        nodes_files = [input_path / f"{args.book}{NODES_SUFFIX}"]
        if not nodes_files[0].exists():
            print(f"file does not exist: {nodes_files[0]}")
            sys.exit(1)
    elif BOOK_FILTER:
        nodes_files = [input_path / f"{BOOK_FILTER}{NODES_SUFFIX}"]
    else:
        nodes_files = sorted(input_path.glob(f"*{NODES_SUFFIX}"))

    if not nodes_files:
        print(f"not found: {INPUT_DIR}/*{NODES_SUFFIX}")
        sys.exit(1)

    print(f"RAPTOR chunk export: {len(nodes_files)} books")

    all_chunks: List[Dict[str, Any]] = []
    total = len(nodes_files)

    for idx, nf in enumerate(nodes_files, start=1):
        result = export_one_book(nf)
        all_chunks.extend(read_json(Path(result["output"])))
        print(f"  [{idx}/{total}] {result['book_title']}: "
              f"{result['total_chunks']} chunks "
              f"(leaf={result['leaf_chunks']}, summary={result['summary_chunks']})")
        print(f"    -> {result['output']}")

    # In --book mode, do not write chunks_all by default to avoid overwriting full outputs
    if is_single_book and not args.write_all:
        print(f"\nsingle-book mode: not writing chunks_all.{SCHEME}.json (use --write-all to force writing)")
    else:
        all_path = Path(OUTPUT_ALL_CHUNKS_DIR) / f"chunks_all.{SCHEME}.json"
        atomic_write_json(all_path, all_chunks)
        print(f"\nSummary: {len(nodes_files)} books, {len(all_chunks)} total chunks")
        print(f"merged file: {all_path}")


if __name__ == "__main__":
    main()
