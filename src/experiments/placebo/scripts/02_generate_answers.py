# -*- coding: utf-8 -*-
"""Placebo-arm (P) answer generation: API qwen3.6-plus; incremental persistence, checkpoint every 20 items, retry on failure

Usage:
  python scripts/02_generate_answers.py --limit 30 --yes
  python scripts/02_generate_answers.py --yes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    ANSWERS_DIR, CONTEXTS_DIR, GEN_TEMP, MODEL_GEN, PROMPTS_DIR, WORKERS,
)
from utils_simple_api import API_BASE_QWEN, API_KEY_QWEN, chat_batch  # noqa: E402

CHUNK = 20


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def write_json(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='only run the first N questions (0=all)')
    ap.add_argument('--book', default=None)
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--yes', action='store_true')
    args = ap.parse_args()

    contexts = read_json(CONTEXTS_DIR / 'contexts_P.json')
    by_book: dict[str, list[dict]] = defaultdict(list)
    for row in contexts:
        by_book[row['book_title']].append(row)
    if args.limit:  # global limit (across books)
        picked = [(b, r) for b in sorted(by_book) for r in by_book[b]][:args.limit]
        by_book = defaultdict(list)
        for b, r in picked:
            by_book[b].append(r)

    system_prompt = (PROMPTS_DIR / 'answer_generation_system.txt').read_text(encoding='utf-8').strip()
    user_template = (PROMPTS_DIR / 'answer_generation_user.txt').read_text(encoding='utf-8').strip()

    books = [args.book] if args.book else sorted(by_book)
    total_new, total_err = 0, 0
    t0 = time.time()
    for book in books:
        rows = by_book.get(book, [])
        if not rows:
            continue
        order = [r['query_id'] for r in rows]
        out_path = ANSWERS_DIR / 'by_book' / f'{book}.fourP.top5.json'
        existing = read_json(out_path) if out_path.exists() else []
        by_qid = {r['query_id']: r for r in existing}
        pending = [r for r in rows if (r['query_id'] not in by_qid) or by_qid[r['query_id']].get('error')]
        print(f'[{book}] total {len(rows)} | existing {len(existing)} | to run {len(pending)} (incl. failed retries)')
        if not pending or not args.yes:
            continue

        for start in range(0, len(pending), CHUNK):
            batch = pending[start:start + CHUNK]
            calls, metas = [], []
            for row in batch:
                chunks_text = '\n\n'.join(f"[Chunk {c['rank']}] {c['text']}" for c in row['chunks'])
                user_prompt = user_template.replace('{question}', row['question']).replace('{chunks_text}', chunks_text)
                calls.append({'system': system_prompt, 'user': user_prompt})
                metas.append(row)

            results = [None] * len(calls)

            def on_result(idx: int, result: dict, _res=results) -> None:
                _res[idx] = result

            chat_batch(calls, model=MODEL_GEN, temperature=GEN_TEMP, workers=args.workers,
                       api_base=API_BASE_QWEN, api_key=API_KEY_QWEN, on_result=on_result)

            for row, res in zip(metas, results):
                res = res or {'ok': False, 'content': '', 'error': 'no_result'}
                by_qid[row['query_id']] = {
                    'query_id': row['query_id'], 'book_title': book,
                    'question': row['question'], 'ground_truth_answer': row['ground_truth_answer'],
                    'question_type': row['question_type'], 'difficulty': row['difficulty'],
                    'context_chunks': [{'chunk_id': c['chunk_id'], 'text': c['text'],
                                        'rank': c['rank'], 'score': c['score'],
                                        'is_outline': c['is_outline']} for c in row['chunks']],
                    'generated_answer': (res.get('content') or '').strip() if res.get('ok') else '',
                    'generation_model': MODEL_GEN,
                    'generation_model_actual': res.get('model_actual', ''),
                    'gen_usage': res.get('usage', {}),
                    'scheme': 'fourP',
                    'error': '' if res.get('ok') else str(res.get('error', ''))[:300],
                }
            write_json(out_path, [by_qid[q] for q in order if q in by_qid])
            ok_n = sum(1 for r in batch if not by_qid[r['query_id']].get('error'))
            print(f'  Checkpoint: +{len(batch)} (ok {ok_n})')

        total_new += len(pending)
        total_err += sum(1 for r in pending if by_qid[r['query_id']].get('error'))

    all_rows = []
    for p in sorted((ANSWERS_DIR / 'by_book').glob('*.fourP.top5.json')):
        all_rows.extend(read_json(p))
    if all_rows:
        write_json(ANSWERS_DIR / 'answers_all.fourP.json', all_rows)
    print(f'Arm P: new/retried this run {total_new} (failed {total_err}), elapsed {time.time()-t0:.0f}s, mode {"RUN" if args.yes else "DRY-RUN"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
