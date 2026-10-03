# -*- coding: utf-8 -*-
"""Assembles context-retrieval chunks (description + source block) and exports a standard split file

Usage:
  python scripts/02_build_ctx_chunks.py
Output:
  data/chunks/{book}.ctx.json + chunks_all.ctx.json
  split_output/json/{book}_ContextualRetrieval_split.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHUNKS_DIR, REPORTS_DIR, SOURCE, SPLIT_OUT  # noqa: E402


def read_jsonl(p: Path):
    rows = []
    with open(p, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--book', default=None)
    args = ap.parse_args()

    source = read_jsonl(SOURCE / 'source_chunks.jsonl')
    contexts = {r['chunk_id']: r for r in read_jsonl(SOURCE.parent / 'cr_contexts.jsonl')}
    if args.book:
        source = [c for c in source if c['book'] == args.book]

    missing_rows = []
    for c in source:
        ctx = contexts.get(c['chunk_id'], {})
        context = (ctx.get('context') or '').strip() if ctx.get('ok') else ''
        if not context:
            missing_rows.append(c)
    missing_ctx = len(missing_rows)
    missing_books = sorted({c['book'] for c in missing_rows})
    if missing_ctx > 20:   # guard cap: above this the generation is incomplete; abort
        print(f'[FAIL] Too many blocks without descriptions: {missing_ctx} (threshold 20); generation likely incomplete')
        return 1

    by_book: dict[str, list[dict]] = defaultdict(list)
    lens = []
    for c in source:
        ctx = contexts.get(c['chunk_id'], {})
        context = (ctx.get('context') or '').strip() if ctx.get('ok') else ''
        if not context:    # isolated failure: use the minimal locating template as fallback
            context = f'This passage is from "{c["book"]}", {c["chapter_id"] or "unknown chapter"}.'
        text = (context + '\n\n' + c['text'])
        lens.append(len(text))
        by_book[c['book']].append({**c, '_context': context, '_text': text})

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for book, rows in sorted(by_book.items()):
        chunk_rows, items = [], []
        for c in rows:
            chunk_rows.append({
                'chunk_id': c['chunk_id'], 'book_title': book, 'text': c['_text'],
                'active_chapter_id': c['chapter_id'],
                'covered_chapter_ids': c['covered_chapter_ids'],
                'crosses_chapter': c['crosses_chapter'],
                'is_outline': False,
                'section_region': c['region'],
                'info': c['info'],
                'source_file': f'{book}_ContextualRetrieval_split.json',
                'source_start': c['source_start'], 'source_end': c['source_end'],
                'length': len(c['_text']),
            })
            items.append({
                'chapter_region': c['region'], 'is_region_summary': False, 'info': c['info'],
                'length': len(c['_text']), 'text': c['_text'],
                'original_length': len(c['text']), 'word_count': len(c['_text'].split()),
                'active_chapter_id': c['chapter_id'],
                'covered_chapter_ids': c['covered_chapter_ids'],
                'crosses_chapter': c['crosses_chapter'],
                'source_start': c['source_start'], 'source_end': c['source_end'],
            })
        (CHUNKS_DIR / f'{book}.ctx.json').write_text(
            json.dumps(chunk_rows, ensure_ascii=False, indent=1), encoding='utf-8')
        (SPLIT_OUT / f'{book}_ContextualRetrieval_split.json').write_text(
            json.dumps({
                'split_method': 'ContextualRetrieval (ours physical chunks + chapter locating description prepended)',
                'max_length': max(x['length'] for x in items),
                'min_length': min(x['length'] for x in items),
                'items': items,
            }, ensure_ascii=False), encoding='utf-8')
        all_rows.extend(chunk_rows)

    (CHUNKS_DIR / 'chunks_all.ctx.json').write_text(
        json.dumps(all_rows, ensure_ascii=False), encoding='utf-8')

    lines = ['=== CR assembly audit ===',
             f'Blocks: {len(all_rows)} (expect 17271)',
             f'Blocks missing descriptions: {missing_ctx} (<=20 allowed, template fallback used)',
             f'description+block chars mean/median/max: {statistics.mean(lens):.0f}/'
             f'{statistics.median(lens):.0f}/{max(lens)}']
    if missing_ctx:
        lines.append(f'Books with fallback blocks: {missing_books}')
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = '\n'.join(lines)
    (REPORTS_DIR / 'cr_chunks_audit.txt').write_text(report, encoding='utf-8')
    print(report)
    ok = missing_ctx <= 20
    print('G2 result:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
