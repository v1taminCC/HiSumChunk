# -*- coding: utf-8 -*-
"""In-book top-10 retrieval (reuses main-experiment query embeddings; local cosine)

Usage:
  python scripts/04_retrieve_top10.py [--book "Common Sense"] [--topk 10]
Output:
  data/runs/{book}.ctx.top10.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EMB_DIR, RUNS_DIR, SOURCE, TOPK  # noqa: E402
from utils_embedding import cosine_search, load_embeddings, load_meta  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--book', default=None)
    ap.add_argument('--topk', type=int, default=TOPK)
    args = ap.parse_args()

    q_dir = SOURCE / 'query_embeddings'
    books = [args.book] if args.book else sorted(
        {p.name.replace('.ctx.chunks.npy', '') for p in EMB_DIR.glob('*.ctx.chunks.npy')
         if not p.name.startswith('chunks_all')})
    if not books:
        print('ctx embeddings not found; run 03_embed_ctx_chunks.py first')
        return 1
    if not args.book:  # full run: all 21 books must have embeddings to prevent silent omissions
        q_books = {p.name.replace('.queries.meta.json', '') for p in q_dir.glob('*.queries.meta.json')}
        missing = sorted(q_books - set(books))
        if missing:
            print(f'[FAIL] Books missing ctx embeddings ({len(missing)}): {missing}')
            return 1

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for book in books:
        q_emb = load_embeddings(q_dir / f'{book}.queries.npy')
        q_meta = load_meta(q_dir / f'{book}.queries.meta.json')
        d_emb = load_embeddings(EMB_DIR / f'{book}.ctx.chunks.npy')
        d_meta = load_meta(EMB_DIR / f'{book}.ctx.chunks.meta.json')
        idxs, scores = cosine_search(q_emb, d_emb, top_k=args.topk)
        rows = []
        for i, q in enumerate(q_meta):
            top = [{'rank': r + 1, 'chunk_id': d_meta[int(j)]['chunk_id'], 'score': float(s)}
                   for r, (j, s) in enumerate(zip(idxs[i], scores[i]))]
            rows.append({
                'query_id': q['query_id'], 'book_title': book,
                'chapter_id': q.get('chapter_id', ''), 'query_variant': 'zh',
                'top_k': top,
            })
        (RUNS_DIR / f'{book}.ctx.top10.json').write_text(
            json.dumps(rows, ensure_ascii=False), encoding='utf-8')
        print(f'[{book}] queries {len(rows)} | index {len(d_meta)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
