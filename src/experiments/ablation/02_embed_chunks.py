#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed ablation variant chunks using local GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    CHUNKS_BY_BOOK_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    EMBEDDINGS_BY_BOOK_DIR,
    MODEL_PATH,
    NORMALIZE_EMBEDDINGS,
    VARIANTS,
)
from utils_embedding import encode_texts, save_embeddings, save_meta
from utils_io import ensure_dir, read_json


def list_book_titles() -> list[str]:
    """Infer all book titles from chunks directory"""
    books = set()
    for fp in CHUNKS_BY_BOOK_DIR.glob("*.json"):
        name = fp.stem
        for v in VARIANTS:
            suffix = f".{v}"
            if name.endswith(suffix):
                books.add(name[: -len(suffix)])
                break
    return sorted(books)


def main():
    print("=" * 60)
    print("Ablation Experiment - Step 2: Embed chunks")
    print("=" * 60)

    ensure_dir(EMBEDDINGS_BY_BOOK_DIR)
    books = list_book_titles()
    print(f"Books: {len(books)}")
    print(f"Variants: {VARIANTS}")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEFAULT_DEVICE}")

    total_embedded = 0

    for variant in VARIANTS:
        print(f"\n--- Variant: {variant} ---")
        for bi, book in enumerate(books, 1):
            chunk_path = CHUNKS_BY_BOOK_DIR / f"{book}.{variant}.json"
            if not chunk_path.exists():
                print(f"  [{bi}/{len(books)}] {book} - skip (no chunk file)")
                continue

            emb_path = EMBEDDINGS_BY_BOOK_DIR / f"{book}.{variant}.chunks.npy"
            meta_path = EMBEDDINGS_BY_BOOK_DIR / f"{book}.{variant}.chunks.meta.json"

            if emb_path.exists() and meta_path.exists():
                print(f"  [{bi}/{len(books)}] {book} - skip (already exists)")
                continue

            rows = read_json(chunk_path)
            texts = [str(row.get("text", "") or "") for row in rows]
            meta = [
                {"row": i, "chunk_id": row["chunk_id"], "book_title": row["book_title"]}
                for i, row in enumerate(rows)
            ]

            vectors = encode_texts(
                texts,
                model_path=Path(MODEL_PATH),
                batch_size=DEFAULT_BATCH_SIZE,
                device=DEFAULT_DEVICE,
                normalize=NORMALIZE_EMBEDDINGS,
                show_progress_bar=False,
            )

            save_embeddings(emb_path, vectors)
            save_meta(meta_path, meta)
            total_embedded += len(rows)
            print(f"  [{bi}/{len(books)}] {book} - {len(rows)} chunks embedded")
    
    print(f"\n{'=' * 60}")
    print(f"Embedding complete: {total_embedded} chunks")
    print("=" * 60)


if __name__ == "__main__":
    main()
