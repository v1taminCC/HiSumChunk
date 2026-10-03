#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from config import BOOK_EMBEDDING_DIR, BOOK_QUERY_DIR, DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, MODEL_PATH, NORMALIZE_EMBEDDINGS
from utils_embedding import encode_texts, save_embeddings, save_meta
from utils_io import read_json
from utils_query_consistency import SUPPORTED_QUERY_VARIANTS, iter_query_rows, query_variant_suffix


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode QA queries; encode all books if --book not specified")
    parser.add_argument("--book", type=str, default=None, help="Book title; encode queries_all.json if not specified")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--query_variant", type=str, choices=SUPPORTED_QUERY_VARIANTS, default="default", help="default=question field; en=retrieval_question_en; default_en=question + retrieval_question_en")
    args = parser.parse_args()

    if args.book:
        query_path = BOOK_QUERY_DIR / f"{args.book}.json"
        source_rows = read_json(query_path)
    else:
        query_path = BOOK_QUERY_DIR.parent / "queries_all.json"
        source_rows = read_json(query_path)

    filtered_rows = iter_query_rows(source_rows, default_book=args.book, query_variant=args.query_variant)
    suffix = query_variant_suffix(args.query_variant)

    texts = [row["question"] for row in filtered_rows]
    meta = [
        {
            "row": i,
            "query_id": row["query_id"],
            "book_title": row["book_title"],
            "chapter_id": row["chapter_id"],
            "query_variant": args.query_variant,
        }
        for i, row in enumerate(filtered_rows)
    ]
    vectors = encode_texts(
        texts,
        model_path=Path(MODEL_PATH),
        batch_size=args.batch_size,
        device=args.device,
        normalize=NORMALIZE_EMBEDDINGS,
        show_progress_bar=True,
    )
    if args.book:
        save_embeddings(BOOK_EMBEDDING_DIR / f"{args.book}.queries{suffix}.npy", vectors)
        save_meta(BOOK_EMBEDDING_DIR / f"{args.book}.queries{suffix}.meta.json", meta)
    else:
        save_embeddings(BOOK_EMBEDDING_DIR.parent / f"queries_all{suffix}.npy", vectors)
        save_meta(BOOK_EMBEDDING_DIR.parent / f"queries_all{suffix}.meta.json", meta)
    print(f"Query file: {query_path}")
    print(f"query_variant: {args.query_variant}")
    print(f"Query count: {len(filtered_rows)}")


if __name__ == "__main__":
    main()
