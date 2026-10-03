# -*- coding: utf-8 -*-
"""Assembles contexts for answer generation (top-5, body blocks)

Usage:
  python scripts/05_build_top5_contexts.py
Output:
  data/contexts/contexts_ctx.json (577 questions)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ANSWER_TOPK, CHUNKS_DIR, CONTEXTS_DIR, RUNS_DIR, SOURCE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--book', default=None)
    args = ap.parse_args()
    qa_rows = json.loads((SOURCE / 'qa_drafts_all.json').read_text(encoding='utf-8'))
    qa_map = {}
    for row in qa_rows:
        book = row.get('book_title', '')
        for qa in row.get('qas', []) or []:
            qid = qa.get('qa_id', '')
            if qid:
                qa_map[qid] = {
                    'book_title': book, 'chapter_id': row.get('chapter_id', ''),
                    'question': qa.get('question', ''), 'ground_truth_answer': qa.get('answer', ''),
                    'question_type': qa.get('question_type', ''), 'difficulty': qa.get('difficulty', ''),
                }

    rows_out = []
    skipped = 0
    for p in sorted(RUNS_DIR.glob('*.ctx.top10.json')):
        book = p.name.split('.ctx.top10')[0]
        if args.book and book != args.book:
            continue
        runs = json.loads(p.read_text(encoding='utf-8'))
        chunks = json.loads((CHUNKS_DIR / f'{book}.ctx.json').read_text(encoding='utf-8'))
        cmap = {c['chunk_id']: c for c in chunks}
        for run in runs:
            qa = qa_map.get(run['query_id'])
            if not qa:
                skipped += 1
                continue
            entries = []
            for item in run['top_k'][:ANSWER_TOPK]:
                c = cmap[item['chunk_id']]
                entries.append({
                    'chunk_id': item['chunk_id'], 'rank': item['rank'], 'score': item['score'],
                    'is_outline': False, 'original_rank': item['rank'], 'text': c['text'],
                })
            rows_out.append({
                'query_id': run['query_id'], 'book_title': book, 'chapter_id': qa['chapter_id'],
                'question': qa['question'], 'ground_truth_answer': qa['ground_truth_answer'],
                'question_type': qa['question_type'], 'difficulty': qa['difficulty'],
                'chunks': entries, 'meta': {'arm': 'ctx', 'scheme': 'ctxretrieval'},
            })

    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
    (CONTEXTS_DIR / 'contexts_ctx.json').write_text(
        json.dumps(rows_out, ensure_ascii=False), encoding='utf-8')
    print(f'contexts_ctx: questions {len(rows_out)} | blocks {sum(len(r["chunks"]) for r in rows_out)} | skipped (no QA) {skipped}')
    if not args.book:
        if skipped:
            print('[FAIL] Some questions have no QA mapping; check the qa_drafts_all snapshot')
            return 1
        if len(rows_out) != 577:
            print(f'[FAIL] Question count {len(rows_out)} != 577 (retrieval or run incomplete)')
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
