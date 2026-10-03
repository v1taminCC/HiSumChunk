#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform in-book retrieval for ablation variants, returning top-k.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    DEFAULT_TOP_K,
    EMBEDDINGS_BY_BOOK_DIR,
    QUERIES_BY_BOOK_DIR,
    RUNS_IN_BOOK_DIR,
    VARIANTS,
)
from utils_embedding import cosine_search, load_embeddings, load_meta
from utils_io import ensure_dir, read_json, write_json


def list_book_titles() -> list[str]:
    books = set()
    for fp in QUERIES_BY_BOOK_DIR.glob("*.json"):
        books.add(fp.stem)
    return sorted(books)


def retrieve_in_book(book: str, variant: str, top_k: int) -> list[dict]:
    q_emb_path = EMBEDDINGS_BY_BOOK_DIR / f"{book}.queries.npy"
    q_meta_path = EMBEDDINGS_BY_BOOK_DIR / f"{book}.queries.meta.json"
    d_emb_path = EMBEDDINGS_BY_BOOK_DIR / f"{book}.{variant}.chunks.npy"
    d_meta_path = EMBEDDINGS_BY_BOOK_DIR / f"{book}.{variant}.chunks.meta.json"

    if not all(p.exists() for p in [q_emb_path, q_meta_path, d_emb_path, d_meta_path]):
        return []

    q_emb = load_embeddings(q_emb_path)
    d_emb = load_embeddings(d_emb_path)
    q_meta = load_meta(q_meta_path)
    d_meta = load_meta(d_meta_path)

    idxs, scores = cosine_search(q_emb, d_emb, top_k=top_k)

    results = []
    for q_row_idx, q_info in enumerate(q_meta):
        top_rows = []
        for rank, (doc_idx, score) in enumerate(zip(idxs[q_row_idx], scores[q_row_idx]), start=1):
            top_rows.append({
                "rank": rank,
                "chunk_id": d_meta[int(doc_idx)]["chunk_id"],
                "score": float(score),
            })
        results.append({
            "query_id": q_info["query_id"],
            "book_title": q_info["book_title"],
            "chapter_id": q_info.get("chapter_id", ""),
            "top_k": top_rows,
        })
    return results


def main():
    print("=" * 60)
    print("Ablation Experiment - Step 3: Retrieval")
    print("=" * 60)

    ensure_dir(RUNS_IN_BOOK_DIR)
    books = list_book_titles()
    print(f"Books: {len(books)}")
    print(f"Variants: {VARIANTS}")
    print(f"top_k: {DEFAULT_TOP_K}")

    total_queries = 0

    for variant in VARIANTS:
        print(f"\n--- Variant: {variant} ---")
        variant_queries = 0
        for bi, book in enumerate(books, 1):
            out_path = RUNS_IN_BOOK_DIR / f"{book}.{variant}.top{DEFAULT_TOP_K}.json"
            if out_path.exists():
                existing = read_json(out_path)
                variant_queries += len(existing)
                print(f"  [{bi}/{len(books)}] {book} - skip ({len(existing)} queries)")
                continue
            rows = retrieve_in_book(book, variant, DEFAULT_TOP_K)
            if rows:
                write_json(out_path, rows)
                variant_queries += len(rows)
                print(f"  [{bi}/{len(books)}] {book} - {len(rows)} queries")
            else:
                print(f"  [{bi}/{len(books)}] {book} - skip (missing embedding files)")

        total_queries += variant_queries
        print(f"  Subtotal: {variant_queries} queries")

    print(f"\n{'=' * 60}")
    print(f"Retrieval complete: {total_queries} queries")
    print("=" * 60)


if __name__ == "__main__":
    main()
