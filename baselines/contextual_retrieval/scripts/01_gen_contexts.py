# -*- coding: utf-8 -*-
"""Per-block description generation (Contextual Retrieval, local SGLang Qwen3-4B-Instruct-2507)

- Input: data/source/source_chunks.jsonl (plain source chunks)
- Document context: an excerpt of the containing chapter (centred on the block, <= DOC_EXCERPT_CHARS characters)
- Output: data/cr_contexts.jsonl (checkpoint resume)
- Failures (empty/overlong) are retried automatically once

Usage:
  python scripts/01_gen_contexts.py --limit 50 --yes
  python scripts/01_gen_contexts.py --yes
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    CR_WORDS_SOFT_MAX, DOC_EXCERPT_CHARS, PROMPTS_DIR, PROMPT_MAX_CHARS, REPORTS_DIR,
    SGLANG_CONCURRENCY, SOURCE,
)
from utils_sglang import chat_batch  # noqa: E402


def read_jsonl(p: Path):
    rows = []
    if not p.exists():
        return rows
    with open(p, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_context(text: str) -> str:
    t = (text or '').strip()
    if t.startswith('```'):
        t = t.split('\n', 1)[-1]
        if t.endswith('```'):
            t = t[:-3]
    t = t.strip().strip('"').strip("'").strip()
    return t


def build_chapter_index(chunks: list[dict]):
    """Groups by (book, chapter), concatenates the full text in order, and records each block position"""
    groups: dict[tuple[str, str], dict] = {}
    order: list[str] = []
    for row in chunks:
        key = (row['book'], row['chapter_id'])
        if key not in groups:
            groups[key] = {'text': '', 'offsets': {}}
            order.append(key)
        g = groups[key]
        g['offsets'][row['chunk_id']] = len(g['text'])
        g['text'] += row['text'] + '\n'
    return groups, order


def make_excerpt(chapter_text: str, pos: int, chunk_len: int) -> str:
    if len(chapter_text) <= DOC_EXCERPT_CHARS:
        return chapter_text
    budget = DOC_EXCERPT_CHARS
    start = max(0, pos - max(1200, (budget - chunk_len) // 2))
    if start + budget > len(chapter_text):
        start = max(0, len(chapter_text) - budget)
    excerpt = chapter_text[start:start + budget]
    return ('…' if start > 0 else '') + excerpt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--book', default=None)
    ap.add_argument('--workers', type=int, default=SGLANG_CONCURRENCY)
    ap.add_argument('--yes', action='store_true')
    args = ap.parse_args()

    chunks = read_jsonl(SOURCE / 'source_chunks.jsonl')
    if args.book:
        chunks = [c for c in chunks if c['book'] == args.book]
    groups, _ = build_chapter_index(chunks)
    template = (PROMPTS_DIR / 'cr_context_user.txt').read_text(encoding='utf-8')

    out_path = SOURCE.parent / 'cr_contexts.jsonl'
    existing = {r['chunk_id']: r for r in read_jsonl(out_path)}
    pending = [c for c in chunks if not (existing.get(c['chunk_id'], {}).get('ok'))]
    if args.limit:
        pending = pending[:args.limit]
    print(f'Total blocks {len(chunks)} | existing {len(existing)} | to run {len(pending)}')
    if not pending or not args.yes:
        if not args.yes:
            print('(--yes not given; preview only)')
        return 0

    items = []
    for c in pending:
        g = groups[(c['book'], c['chapter_id'])]
        excerpt = make_excerpt(g['text'], g['offsets'][c['chunk_id']], len(c['text']))
        prompt = (template
                  .replace('{book_title}', c['book'])
                  .replace('{chapter_title}', c['chapter_id'] or '(unknown)')
                  .replace('{document_excerpt}', excerpt)
                  .replace('{chunk_text}', c['text']))
        if len(prompt) > PROMPT_MAX_CHARS:
            prompt = prompt[:PROMPT_MAX_CHARS]
        items.append({**c, 'prompt': prompt, 'prompt_chars': len(prompt)})
    items.sort(key=lambda x: x['prompt_chars'])

    t_all = time.time()
    stats = Counter()
    with open(out_path, 'a', encoding='utf-8') as fout:
        for start in range(0, len(items), args.workers):
            batch = items[start:start + args.workers]
            prompts = [x['prompt'] for x in batch]
            t0 = time.time()
            results = chat_batch(prompts, workers=len(batch))
            latency = time.time() - t0

            rows, retry = [], []
            for x, raw in zip(batch, results):
                ctx = clean_context(raw)
                words = len(ctx.split())
                ok = bool(ctx) and words <= CR_WORDS_SOFT_MAX
                row = {'chunk_id': x['chunk_id'], 'book': x['book'], 'chapter_id': x['chapter_id'],
                       'context': ctx, 'chars': len(ctx), 'words': words, 'ok': ok,
                       'prompt_chars': x['prompt_chars']}
                rows.append(row)
                if not ok:
                    retry.append(x)
                stats['done'] += 1

            if retry:
                r_prompts = [x['prompt'] for x in retry]
                r_results = chat_batch(r_prompts, workers=len(retry))
                r_map = {x['chunk_id']: clean_context(r) for x, r in zip(retry, r_results)}
                for row in rows:
                    if row['chunk_id'] in r_map:
                        ctx = r_map[row['chunk_id']]
                        words = len(ctx.split())
                        row.update({'context': ctx, 'chars': len(ctx), 'words': words,
                                    'ok': bool(ctx) and words <= CR_WORDS_SOFT_MAX})

            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            fout.flush()
            n_ok = sum(1 for r in rows if r['ok'])
            print(f'  Batch {start//args.workers + 1}: +{len(rows)} (ok {n_ok}) in {latency:.1f}s', flush=True)

    all_map = {}
    for r in read_jsonl(out_path):        # checkpoint resume may duplicate rows; keep the latest
        all_map[r['chunk_id']] = r
    all_rows = list(all_map.values())
    ok_rows = [r for r in all_rows if r.get('ok')]
    words = [r['words'] for r in ok_rows]
    lines = ['=== CR description generation stats ===',
             f'Cumulative {len(all_rows)} (ok {len(ok_rows)} / target 17,271 at full scale)',
             f'New this run: {stats["done"]}, elapsed {(time.time()-t_all)/60:.1f} min']
    if words:
        lines.append(f'Description words mean/median/max: {statistics.mean(words):.1f}/{statistics.median(words):.0f}/{max(words)}')
    bad = [r for r in all_rows if not r.get('ok')]
    lines.append(f'Empty/overlong: {len(bad)}')
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / 'cr_gen_stats.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
