#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

from config import BOOK_EMBEDDING_DIR, BOOK_QUERY_DIR, DEFAULT_TOP_K, GLOBAL_RUN_DIR, IN_BOOK_RUN_DIR, SUPPORTED_SCHEMES
from utils_book import list_book_titles
from utils_embedding import cosine_search, load_embeddings, load_meta
from utils_io import ensure_dir, write_json
from utils_query_consistency import SUPPORTED_QUERY_VARIANTS, query_variant_suffix, validate_global_query_embeddings, validate_in_book_query_embeddings


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    total = max(total, 1)
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix}: [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


def retrieve_in_book(book: str, scheme: str, top_k: int, query_variant: str) -> list[dict]:
    suffix = query_variant_suffix(query_variant)
    query_path = BOOK_QUERY_DIR / f"{book}.json"
    meta_path = BOOK_EMBEDDING_DIR / f"{book}.queries{suffix}.meta.json"
    validate_in_book_query_embeddings(book, query_path, meta_path, query_variant=query_variant)
    q_emb = load_embeddings(BOOK_EMBEDDING_DIR / f"{book}.queries{suffix}.npy")
    d_emb = load_embeddings(BOOK_EMBEDDING_DIR / f"{book}.{scheme}.chunks.npy")
    q_meta = load_meta(meta_path)
    d_meta = load_meta(BOOK_EMBEDDING_DIR / f"{book}.{scheme}.chunks.meta.json")
    idxs, scores = cosine_search(q_emb, d_emb, top_k=top_k)
    results = []
    total_queries = len(q_meta)
    for q_row_idx, q_info in enumerate(q_meta):
        top_rows = []
        for rank, (doc_idx, score) in enumerate(zip(idxs[q_row_idx], scores[q_row_idx]), start=1):
            top_rows.append({"rank": rank, "chunk_id": d_meta[int(doc_idx)]["chunk_id"], "score": float(score)})
        results.append(
            {
                "query_id": q_info["query_id"],
                "book_title": q_info["book_title"],
                "chapter_id": q_info.get("chapter_id", ""),
                "query_variant": query_variant,
                "top_k": top_rows,
            }
        )
        print_progress(q_row_idx + 1, total_queries, prefix=f"Retrieving {book}")
    return results


def retrieve_global(scheme: str, top_k: int, query_variant: str) -> list[dict]:
    suffix = query_variant_suffix(query_variant)
    query_path = BOOK_QUERY_DIR.parent / "queries_all.json"
    meta_path = BOOK_EMBEDDING_DIR.parent / f"queries_all{suffix}.meta.json"
    validate_global_query_embeddings(query_path, meta_path, query_variant=query_variant)
    q_emb = load_embeddings(BOOK_EMBEDDING_DIR.parent / f"queries_all{suffix}.npy")
    d_emb = load_embeddings(BOOK_EMBEDDING_DIR.parent / f"chunks_all.{scheme}.npy")
    q_meta = load_meta(meta_path)
    d_meta = load_meta(BOOK_EMBEDDING_DIR.parent / f"chunks_all.{scheme}.meta.json")
    idxs, scores = cosine_search(q_emb, d_emb, top_k=top_k)
    results = []
    total_queries = len(q_meta)
    for q_row_idx, q_info in enumerate(q_meta):
        top_rows = []
        for rank, (doc_idx, score) in enumerate(zip(idxs[q_row_idx], scores[q_row_idx]), start=1):
            top_rows.append({"rank": rank, "chunk_id": d_meta[int(doc_idx)]["chunk_id"], "score": float(score)})
        results.append(
            {
                "query_id": q_info["query_id"],
                "book_title": q_info["book_title"],
                "chapter_id": q_info.get("chapter_id", ""),
                "query_variant": query_variant,
                "top_k": top_rows,
            }
        )
        print_progress(q_row_idx + 1, total_queries, prefix="Global retrieval queries")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="QA retrieval; retrieve all books if --book not specified in in_book mode")
    parser.add_argument("--scheme", type=str, choices=SUPPORTED_SCHEMES, default="booksum_split")
    parser.add_argument("--mode", type=str, choices=["in_book", "global"], default="in_book")
    parser.add_argument("--book", type=str, default=None, help="Book title; process all books if not specified in in_book mode")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--query_variant", type=str, choices=SUPPORTED_QUERY_VARIANTS, default="default", help="default=question field; en=retrieval_question_en; default_en=question + retrieval_question_en")
    args = parser.parse_args()
    suffix = query_variant_suffix(args.query_variant)

    if args.mode == "in_book":
        ensure_dir(IN_BOOK_RUN_DIR)
        books = [args.book] if args.book else list_book_titles()
        total_rows = 0
        for book_idx, book in enumerate(books, start=1):
            print(f"\n[{book_idx}/{len(books)}] In-book retrieval: {book}", flush=True)
            rows = retrieve_in_book(book, args.scheme, top_k=args.top_k, query_variant=args.query_variant)
            output = IN_BOOK_RUN_DIR / f"{book}.{args.scheme}{suffix}.top{args.top_k}.json"
            write_json(output, rows)
            total_rows += len(rows)
        print(f"Query count: {total_rows}")
    else:
        ensure_dir(GLOBAL_RUN_DIR)
        rows = retrieve_global(args.scheme, top_k=args.top_k, query_variant=args.query_variant)
        output = GLOBAL_RUN_DIR / f"all_books.{args.scheme}{suffix}.top{args.top_k}.json"
        write_json(output, rows)
        print(f"Query count: {len(rows)}")


if __name__ == "__main__":
    main()
