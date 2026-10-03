#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from config import BOOK_CHUNK_DIR, BOOK_EMBEDDING_DIR, CHUNK_DIR, DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, MODEL_PATH, NORMALIZE_EMBEDDINGS, SUPPORTED_SCHEMES
from utils_embedding import encode_texts, save_embeddings, save_meta
from utils_io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode chunks for a scheme; encode all books if --book not specified")
    parser.add_argument("--scheme", type=str, choices=SUPPORTED_SCHEMES, default="booksum_split")
    parser.add_argument("--book", type=str, default=None, help="Book title; encode chunks_all if not specified")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()

    if args.book:
        chunk_path = BOOK_CHUNK_DIR / f"{args.book}.{args.scheme}.json"
    else:
        chunk_path = CHUNK_DIR / f"chunks_all.{args.scheme}.json"
    rows = read_json(chunk_path)

    texts = [str(row.get("text", "") or "") for row in rows]
    meta = [
        {
            "row": i,
            "chunk_id": row["chunk_id"],
            "book_title": row["book_title"],
        }
        for i, row in enumerate(rows)
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
        save_embeddings(BOOK_EMBEDDING_DIR / f"{args.book}.{args.scheme}.chunks.npy", vectors)
        save_meta(BOOK_EMBEDDING_DIR / f"{args.book}.{args.scheme}.chunks.meta.json", meta)
    else:
        save_embeddings(BOOK_EMBEDDING_DIR.parent / f"chunks_all.{args.scheme}.npy", vectors)
        save_meta(BOOK_EMBEDDING_DIR.parent / f"chunks_all.{args.scheme}.meta.json", meta)
    print(f"Chunk file: {chunk_path}")
    print(f"Chunk count: {len(rows)}")


if __name__ == "__main__":
    main()
