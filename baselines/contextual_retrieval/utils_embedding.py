from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_model_cache: dict[tuple[str, str], SentenceTransformer] = {}


def load_embedding_model(model_path: Path, device: str = "cuda") -> SentenceTransformer:
    key = (str(model_path), device)
    if key not in _model_cache:
        actual_device = device
        if device == "cuda" and not torch.cuda.is_available():
            actual_device = "cpu"
        _model_cache[key] = SentenceTransformer(str(model_path), device=actual_device)
    return _model_cache[key]


def encode_texts(
    texts: Sequence[str],
    model_path: Path,
    batch_size: int = 32,
    device: str = "cuda",
    normalize: bool = True,
    show_progress_bar: bool = True,
) -> np.ndarray:
    # Truncate overly long text to prevent OOM (4096 chars ~ 1000 tokens)
    MAX_CHARS = 4096
    truncated = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in texts]
    model = load_embedding_model(model_path, device=device)
    vectors = model.encode(
        list(truncated),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def save_embeddings(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def save_meta(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(rows), f, ensure_ascii=False, indent=2)


def load_meta(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_search(query_matrix: np.ndarray, doc_matrix: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.matmul(query_matrix, doc_matrix.T)
    k = min(top_k, scores.shape[1])
    if k <= 0:
        return np.empty((scores.shape[0], 0), dtype=np.int64), np.empty((scores.shape[0], 0), dtype=np.float32)
    top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, top_idx, axis=1)
    order = np.argsort(-top_scores, axis=1)
    sorted_idx = np.take_along_axis(top_idx, order, axis=1)
    sorted_scores = np.take_along_axis(top_scores, order, axis=1)
    return sorted_idx, sorted_scores.astype(np.float32)
