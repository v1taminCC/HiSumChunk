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
RAPTOR Node Embedding
=====================
Generate embeddings for RAPTOR chunks book by book, with validation and resumable processing.

Input:  .../data/chunks/by_book/{Book}.raptor.json
output:  .../embeddings/by_book/{Book}.raptor.chunks.npy
       .../embeddings/by_book/{Book}.raptor.chunks.meta.json
"""

# ================================================================================================
# ==================================== Parameters ====================================================
# ================================================================================================

EMBEDDING_MODEL_PATH: str = os.getenv("EMBEDDING_MODEL_PATH", "<MODEL_DIR>/Qwen3-Embedding-0.6B")
CHUNKS_DIR: str = str(REPO_ROOT / "src" / "retrieval" / "data" / "chunks" / "by_book")
EMBEDDINGS_DIR: str = str(REPO_ROOT / "src" / "retrieval" / "embeddings" / "by_book")
SCHEME: str = "raptor"
EMBED_BATCH_SIZE: int = 32
DEVICE: str = "cuda"
NORMALIZE: bool = True
FORCE_RECOMPUTE: bool = False
BOOK_FILTER: str = None

# ================================================================================================
# ========================================== Implementation ============================================
# ================================================================================================

import argparse
import json
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_model_cache: Dict[Tuple[str, str], SentenceTransformer] = {}


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp_path), str(path))


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + ".tmp.npy")
    np.save(str(tmp_path), array)
    os.replace(str(tmp_path), str(path))


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embedding_model(model_path: str, device: str = "cuda") -> SentenceTransformer:
    key = (model_path, device)
    if key not in _model_cache:
        actual = device
        if device == "cuda" and not torch.cuda.is_available():
            actual = "cpu"
        _model_cache[key] = SentenceTransformer(model_path, device=actual)
    return _model_cache[key]


def encode_texts(texts: Sequence[str], model_path: str, batch_size: int = 32,
                 device: str = "cuda", normalize: bool = True) -> np.ndarray:
    truncated = [t[:4096] for t in texts]
    model = load_embedding_model(model_path, device=device)
    vectors = model.encode(list(truncated), batch_size=batch_size, show_progress_bar=False,
                           normalize_embeddings=normalize, convert_to_numpy=True)
    return np.asarray(vectors, dtype=np.float32)


def validate_existing_embeddings(chunk_path: Path, emb_path: Path, meta_path: Path) -> Tuple[bool, str]:
    """Validate whether existing embeddings are usable"""
    if not emb_path.exists() or not meta_path.exists():
        return False, "file does not exist"

    # Read chunks
    try:
        rows = read_json(chunk_path)
        chunk_ids = [r["chunk_id"] for r in rows]
    except Exception as e:
        return False, f"failed to read chunk file: {e}"

    # Read metadata
    try:
        meta = read_json(meta_path)
        meta_ids = [m["chunk_id"] for m in meta]
    except Exception as e:
        return False, f"failed to read metadata file: {e}"

    # Read npy
    try:
        vectors = np.load(str(emb_path))
    except Exception as e:
        return False, f"failed to read npy file: {e}"

    # Validate
    if vectors.ndim != 2:
        return False, f"invalid npy dimensions: ndim={vectors.ndim}"
    if vectors.shape[1] <= 0:
        return False, f"npy feature dimension is zero"
    if len(chunk_ids) != len(meta_ids):
        return False, f"chunk rows {len(chunk_ids)} != meta rows {len(meta_ids)}"
    if len(chunk_ids) != vectors.shape[0]:
        return False, f"chunk rows {len(chunk_ids)} != npy rows {vectors.shape[0]}"
    if chunk_ids != meta_ids:
        return False, f"chunk_id order mismatch"

    return True, f"OK, rows={vectors.shape[0]}, dim={vectors.shape[1]}"


def embed_one_book(chunk_path: Path, emb_path: Path, meta_path: Path) -> Dict[str, Any]:
    rows = read_json(chunk_path)
    texts = [str(r.get("text", "") or "") for r in rows]
    meta = [{"row": i, "chunk_id": r["chunk_id"], "book_title": r["book_title"]}
            for i, r in enumerate(rows)]

    vectors = encode_texts(texts, EMBEDDING_MODEL_PATH, batch_size=EMBED_BATCH_SIZE,
                           device=DEVICE, normalize=NORMALIZE)

    atomic_save_npy(emb_path, vectors)
    atomic_write_json(meta_path, meta)

    return {"rows": len(rows), "dim": vectors.shape[1]}


def main() -> None:
    parser = argparse.ArgumentParser(description="RAPTOR node embedding")
    parser.add_argument("--book", type=str, default=None)
    args = parser.parse_args()

    chunks_dir = Path(CHUNKS_DIR)
    emb_dir = Path(EMBEDDINGS_DIR)

    if args.book:
        chunk_files = [chunks_dir / f"{args.book}.{SCHEME}.json"]
        if not chunk_files[0].exists():
            print(f"file does not exist: {chunk_files[0]}")
            sys.exit(1)
    elif BOOK_FILTER:
        chunk_files = [chunks_dir / f"{BOOK_FILTER}.{SCHEME}.json"]
    else:
        chunk_files = sorted(chunks_dir.glob(f"*.{SCHEME}.json"))

    if not chunk_files:
        print(f"not found: {chunks_dir}/*.{SCHEME}.json")
        sys.exit(1)

    print(f"RAPTOR embedding: {len(chunk_files)} books, model={EMBEDDING_MODEL_PATH}")

    total_chunks = 0
    computed = 0
    skipped = 0

    for idx, chunk_path in enumerate(chunk_files, start=1):
        book_title = chunk_path.name.replace(f".{SCHEME}.json", "")
        emb_path = emb_dir / f"{book_title}.{SCHEME}.chunks.npy"
        meta_path = emb_dir / f"{book_title}.{SCHEME}.chunks.meta.json"

        if not FORCE_RECOMPUTE:
            valid, reason = validate_existing_embeddings(chunk_path, emb_path, meta_path)
            if valid:
                print(f"  [{idx}/{len(chunk_files)}] [skip] {book_title}: {reason}")
                skipped += 1
                total_chunks += int(reason.split("rows=")[1].split(",")[0]) if "rows=" in reason else 0
                continue
            else:
                print(f"  [{idx}/{len(chunk_files)}] [recompute] {book_title}: {reason}")

        result = embed_one_book(chunk_path, emb_path, meta_path)
        computed += 1
        total_chunks += result["rows"]
        print(f"  [{idx}/{len(chunk_files)}] {book_title}: "
              f"{result['rows']} chunks, dim={result['dim']}")
        print(f"    -> {emb_path.name}")

    print(f"\nSummary: {len(chunk_files)} books, skip={skipped}, recompute={computed}, total chunks={total_chunks}")


if __name__ == "__main__":
    main()
