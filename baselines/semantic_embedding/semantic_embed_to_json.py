#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic embedding similarity chunking — paragraph-level (not sentence-level).
Embed each paragraph with local-embedding -> adjacent cosine similarity -> local valley boundary detection -> merge to target word count.

Model: local local-embedding  |  GPU required

Usage:
  python3 semantic_embed_to_json.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ============================================================
# ==== Parameters (edit here) ====
# ============================================================
MODEL_PATH          = os.getenv("EMBEDDING_MODEL_PATH", "<MODEL_DIR>/Qwen3-Embedding-0.6B")
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
WORK_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))
INPUT_DIR           = str(WORK_BOOK_DIR)
OUTPUT_DIR          = str(WORK_BOOK_DIR)

TARGET_CHUNK_WORDS  = 1000       # Target chunk size in words
MIN_CHUNK_WORDS     = 400        # Force merge into adjacent chunk if below this
MAX_CHUNK_WORDS     = 2000       # Allow splitting chunks above this size

BOUNDARY_PCT        = 12         # Diff percentile threshold for paragraph boundaries (lower = fewer splits)
MERGE_SMALL_MIN     = 50         # Auto-merge atomic paragraphs below this word count
EMBED_BATCH_SIZE    = 5          # Embedding encode batch size (reduce for longer paragraphs)

SPLIT_METHOD        = "SemanticEmbedding"
OUTPUT_SUFFIX       = "_SemanticEmbed_split.json"
# ============================================================

ANCHOR_PATTERN = re.compile(r"^<<<CHAPTER_ID:(.*?)>>>\s*$", re.MULTILINE)
WORD_PATTERN   = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def load_embedder(model_path: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("pip install sentence-transformers", file=sys.stderr); raise
    print(f"Loading: {model_path} ...", flush=True)
    return SentenceTransformer(model_path, device="cuda")


def parse_book(raw_text: str) -> list[dict]:
    matches = list(ANCHOR_PATTERN.finditer(raw_text))
    if not matches:
        raise ValueError("No CHAPTER_ID anchors found")
    chapters = []
    for idx, m in enumerate(matches):
        cid   = m.group(1).strip()
        start = m.end()
        end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        text  = raw_text[start:end].strip()
        if text:
            chapters.append({"chapter_id": cid, "text": text, "offset": start})
    return chapters


def split_paragraphs(text: str) -> list[str]:
    raw = re.split(r"\n\s*\n", text)
    result = []
    for p in raw:
        p = p.strip()
        if p and count_words(p) >= 10:
            result.append(p)
        elif p and result:
            result[-1] += " " + p
    # Merge tiny paragraphs
    merged = []
    for p in result:
        if merged and count_words(p) < MERGE_SMALL_MIN:
            merged[-1] += "\n" + p
        else:
            merged.append(p)
    return merged


def embed_paragraphs(paragraphs: list[str], embedder) -> np.ndarray:
    if not paragraphs:
        return np.array([])
    # Truncate long paragraphs to prevent OOM
    truncated = [p[:4096] for p in paragraphs]
    embs = embedder.encode(truncated, batch_size=EMBED_BATCH_SIZE,
                           normalize_embeddings=True, show_progress_bar=False)
    torch.cuda.empty_cache()
    return embs


def find_boundaries(embeddings: np.ndarray) -> list[int]:
    """Return boundary point indices (between paragraph i and i+1)"""
    if len(embeddings) <= 1:
        return []
    diffs = np.array([1.0 - float(np.dot(embeddings[i], embeddings[i + 1]))
                      for i in range(len(embeddings) - 1)])
    threshold = np.percentile(diffs, 100 - BOUNDARY_PCT)

    boundaries = []
    for i in range(len(diffs)):
        is_local_max = True
        if i > 0 and diffs[i - 1] > diffs[i]:
            is_local_max = False
        if i < len(diffs) - 1 and diffs[i + 1] > diffs[i]:
            is_local_max = False
        if diffs[i] >= threshold and is_local_max:
            boundaries.append(i)
    return boundaries


def _split_oversize(paras: list[str], out: list[str]):
    """Recursively bisect oversized paragraph groups"""
    text = "\n".join(paras)
    wc = count_words(text)
    if wc <= MAX_CHUNK_WORDS or len(paras) < 2:
        out.append(text)
    else:
        mid = len(paras) // 2
        _split_oversize(paras[:mid], out)
        _split_oversize(paras[mid:], out)


def build_chunks(paragraphs: list[str], boundaries: list[int]) -> list[str]:
    """Merge paragraphs into chunks by boundary points, then merge/split"""
    if not paragraphs:
        return []
    if not boundaries:
        return ["\n".join(paragraphs)]

    # Split
    groups = []
    prev = 0
    for b in boundaries:
        if b + 1 > prev:
            groups.append(paragraphs[prev:b + 1])
            prev = b + 1
    if prev < len(paragraphs):
        groups.append(paragraphs[prev:])

    # Merge undersized groups
    merged_groups = []
    for g in groups:
        merged_groups.append(g)
    # Multi-round merge until all groups >= MIN_CHUNK_WORDS
    changed = True
    while changed:
        changed = False
        new_groups = []
        i = 0
        while i < len(merged_groups):
            wc = count_words("\n".join(merged_groups[i]))
            if wc < MIN_CHUNK_WORDS and i + 1 < len(merged_groups):
                merged_groups[i + 1] = merged_groups[i] + merged_groups[i + 1]
                changed = True
                i += 1
            elif wc < MIN_CHUNK_WORDS and i > 0 and new_groups:
                new_groups[-1].extend(merged_groups[i])
                changed = True
                i += 1
            else:
                new_groups.append(merged_groups[i])
                i += 1
        merged_groups = new_groups

    # Recursively bisect groups > MAX_CHUNK_WORDS
    final_texts = []
    for g in merged_groups:
        _split_oversize(g, final_texts)

    return final_texts


def build_items(book_txt_path: Path, embedder) -> list[dict]:
    raw = book_txt_path.read_text(encoding="utf-8")
    chapters = parse_book(raw)
    items: list[dict] = []
    idx = 0
    t0 = time.time()
    total_paras = 0
    for ch in chapters:
        paras = split_paragraphs(ch["text"])
        total_paras += len(paras)
        if len(paras) <= 1:
            items.append(_item(ch, ch["text"].strip(), idx := idx + 1))
            continue
        embs = embed_paragraphs(paras, embedder)
        boundaries = find_boundaries(embs)
        chunks = build_chunks(paras, boundaries)
        for c in chunks:
            c = c.strip()
            if c:
                items.append(_item(ch, c, idx := idx + 1))

    print(f"  {idx} chunks from {total_paras} paras, {time.time()-t0:.0f}s", flush=True)
    return items


def _item(chapter: dict, text: str, idx: int) -> dict:
    return {
        "block_id": idx,
        "length": count_words(text),
        "text": text,
        "char_length": len(text),
        "active_chapter_id": chapter["chapter_id"],
        "covered_chapter_ids": [chapter["chapter_id"]],
        "crosses_chapter": False,
        "source_start": chapter["offset"],
        "source_end": chapter["offset"] + len(chapter["text"]),
    }


def main():
    if not Path(INPUT_DIR).exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    embedder = load_embedder(MODEL_PATH)
    txt_files = sorted(p for p in Path(INPUT_DIR).glob("*.txt") if p.is_file())

    total_c = 0
    t0 = time.time()
    for fi, fp in enumerate(txt_files, 1):
        print(f"[{fi}/{len(txt_files)}] {fp.stem}", flush=True)
        items = build_items(fp, embedder)
        data = {
            "split_method": SPLIT_METHOD, "model": MODEL_PATH.split("/")[-1],
            "target_words": TARGET_CHUNK_WORDS, "min_chunk_words": MIN_CHUNK_WORDS,
            "max_chunk_words": MAX_CHUNK_WORDS, "boundary_pct": BOUNDARY_PCT,
            "max_length": max((i["length"] for i in items), default=0),
            "min_length": min((i["length"] for i in items), default=0),
            "items": items,
        }
        out = Path(OUTPUT_DIR) / f"{fp.stem}{OUTPUT_SUFFIX}"
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        total_c += len(items)
        lens = [i["length"] for i in items]
        print(f"  -> {len(items)} chunks, avg={sum(lens)/max(len(lens),1):.0f}w, [{min(lens)}-{max(lens)}]", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}\nDone: {len(txt_files)} books, {total_c} chunks, {elapsed/60:.1f}min\n{'='*60}")


if __name__ == "__main__":
    main()
