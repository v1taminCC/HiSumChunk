# -*- coding: utf-8 -*-
"""Context-retrieval chunk embedding (local Qwen3-Embedding-0.6B; GPU preferred)

Usage:
  python scripts/03_embed_ctx_chunks.py --limit 64        # smoke (writes embeddings_smoke/)
  python scripts/03_embed_ctx_chunks.py                   # full run (per book)
  python scripts/03_embed_ctx_chunks.py --book "Common Sense"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHUNKS_DIR, EMBED_BATCH, EMBED_DEVICE, EMBED_MODEL_PATH, EMB_DIR  # noqa: E402
from utils_embedding import encode_texts, save_embeddings, save_meta  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--book', default=None)
    ap.add_argument('--limit', type=int, default=0, help='smoke: encode only the first N blocks, writing to embeddings_smoke/')
    ap.add_argument('--batch_size', type=int, default=EMBED_BATCH)
    ap.add_argument('--device', default=EMBED_DEVICE)
    args = ap.parse_args()

    files = [CHUNKS_DIR / f'{args.book}.ctx.json'] if args.book else \
        sorted(p for p in CHUNKS_DIR.glob('*.ctx.json') if not p.name.startswith('chunks_all'))
    if not files:
        print('ctx chunks not found; run 02_build_ctx_chunks.py first')
        return 1

    out_dir = (EMB_DIR.parent / 'embeddings_smoke') if args.limit else EMB_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in files:
        book = p.name.replace('.ctx.json', '')
        rows = json.loads(p.read_text(encoding='utf-8'))
        if args.limit:
            rows = rows[:args.limit]
        texts = [r['text'] for r in rows]
        t0 = time.time()
        vectors = encode_texts(texts, model_path=EMBED_MODEL_PATH, batch_size=args.batch_size,
                               device=args.device, normalize=True, show_progress_bar=len(texts) > 200)
        dt = time.time() - t0
        save_embeddings(out_dir / f'{book}.ctx.chunks.npy', vectors)
        save_meta(out_dir / f'{book}.ctx.chunks.meta.json',
                  [{'row': i, 'chunk_id': r['chunk_id'], 'book_title': book}
                   for i, r in enumerate(rows)])
        print(f'[{book}] {len(rows)} blocks | dim {vectors.shape[1]} | {dt:.1f}s | {len(rows)/dt:.1f} blocks/s | '
              f'out={out_dir.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
