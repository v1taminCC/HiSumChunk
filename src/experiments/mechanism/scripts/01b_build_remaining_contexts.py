# -*- coding: utf-8 -*-
"""G2b: builds extended four-arm contexts for the remaining questions (pool 559 - sample 200 = 359).

Only new files are added; the 200-question main outputs under 02-contexts/ are never touched.

Usage:
  python scripts/01b_build_remaining_contexts.py                       # full 359
  python scripts/01b_build_remaining_contexts.py --limit 5 --out-dir 99-temp/preview
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import CONTEXTS_EXT_DIR, MIN_BODY_CHUNKS, SAMPLE_DIR, TOPN  # noqa: E402

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def write_json(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def load_base_module():
    """Reuses load_all / build_arms / ensure_prefix / TOPK from 01 to avoid duplicating logic."""
    spec = importlib.util.spec_from_file_location(
        'fourarm_base_01', Path(__file__).parent / '01_build_sample_contexts.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='build only the first N questions (0=all)')
    ap.add_argument('--out-dir', default=None, help='output directory override (for testing; default 02-contexts-remaining)')
    args = ap.parse_args()
    full_run = args.limit == 0
    out_dir = Path(args.out_dir) if args.out_dir else CONTEXTS_EXT_DIR

    base = load_base_module()
    qa_map, runs_by_query, chunk_map, prefix_map, old_ctx = base.load_all()
    sample = read_json(SAMPLE_DIR / 'sample_200.json')
    sample_qids = set(sample['sampled_query_ids'])

    pool, excluded = [], []
    for qid, run in runs_by_query.items():
        body_n = sum(1 for it in run.get('top_k', [])[:TOPN]
                     if not chunk_map.get(it['chunk_id'], {}).get('is_outline'))
        (pool if body_n >= MIN_BODY_CHUNKS else excluded).append(qid)
    if len(pool) != sample.get('pool_size') or len(excluded) != sample.get('excluded_count'):
        print(f"[ERROR] Pool size differs from sample_200.json (pool {len(pool)} vs {sample.get('pool_size')},"
              f"excluded {len(excluded)} vs {sample.get('excluded_count')}); source data may have changed; stopping")
        return 1

    remaining = sorted(set(pool) - sample_qids)
    if args.limit:
        remaining = remaining[:args.limit]
    print(f'Remaining questions: {len(remaining)} (pool {len(pool)} - sample {len(sample_qids)})')

    contexts = {arm: [] for arm in 'ABCD'}
    strip_stats = defaultdict(lambda: {'strip_ok': 0, 'strip_fallback': 0})
    a_text_mismatch = 0
    c_d_outline_left = 0
    chunk_count_bad = 0
    missing_qa = 0

    for qid in remaining:
        if qid not in qa_map:
            missing_qa += 1
            continue
        run = runs_by_query[qid]
        qa = qa_map[qid]
        pmap = prefix_map.get(qa['book_title'], {})
        built = base.build_arms(qid, run, chunk_map, pmap)
        for arm in 'ABCD':
            entries = built['arms'][arm]
            if len(entries) != base.TOPK:
                chunk_count_bad += 1
            if arm in ('C', 'D') and any(e['is_outline'] for e in entries):
                c_d_outline_left += 1
            contexts[arm].append({
                'query_id': qid, 'book_title': qa['book_title'], 'chapter_id': qa['chapter_id'],
                'question': qa['question'], 'ground_truth_answer': qa['ground_truth_answer'],
                'question_type': qa['question_type'], 'difficulty': qa['difficulty'],
                'chunks': entries,
                'meta': {'arm': arm, 'subset': 'remaining', **built['stats'][arm]},
            })
        old = old_ctx.get(qid, [])
        new_a = contexts['A'][-1]['chunks']
        if len(old) != len(new_a) or any(o[0] != n['chunk_id'] or o[1] != n['text'] for o, n in zip(old, new_a)):
            a_text_mismatch += 1

    def wc(text: str) -> int:
        return len(WORD_RE.findall(text))

    per_book = Counter(qa_map[q]['book_title'] for q in remaining if q in qa_map)
    audit = ['=== Remaining-questions four-arm context build audit ===',
             f'remaining={len(remaining)} | pool={len(pool)} | excluded={len(excluded)} | books={len(per_book)}',
             f'Questions missing from qa_map: {missing_qa} (expect 0)',
             f'Questions whose A context differs from old answers: {a_text_mismatch} (expect 0)',
             f'Questions where C/D still contain H1 slices: {c_d_outline_left} (expect 0)',
             f'Occurrences of an arm with block count != 5: {chunk_count_bad} (expect 0)']
    for arm in 'ABCD':
        lens = [sum(wc(c['text']) for c in row['chunks']) for row in contexts[arm]]
        audit.append(f'Arm {arm}: context word count min/median/max = {min(lens)}/{int(statistics.median(lens))}/{max(lens)}')
    for arm in 'BD':
        s = Counter()
        for row in contexts[arm]:
            s['strip_ok'] += row['meta']['strip_ok']
            s['strip_fallback'] += row['meta']['strip_fallback']
        audit.append(f'Arm {arm}: prefix stripped ok {s["strip_ok"]} / fallback {s["strip_fallback"]}')

    for arm in 'ABCD':
        write_json(out_dir / f'contexts_{arm}.json', contexts[arm])
    (out_dir / 'context_audit.txt').write_text('\n'.join(audit), encoding='utf-8')

    if full_run and out_dir == CONTEXTS_EXT_DIR:
        write_json(SAMPLE_DIR / 'remaining.json', {
            'pool_size': len(pool), 'excluded_count': len(excluded), 'sample_size': len(sample_qids),
            'remaining_count': len(remaining), 'remaining_query_ids': remaining,
            'excluded_query_ids': sorted(excluded),
            'per_book_counts': dict(sorted(per_book.items())),
        })

    ok = (missing_qa == 0 and a_text_mismatch == 0 and c_d_outline_left == 0 and chunk_count_bad == 0)
    print('\n'.join(audit))
    print('\nG2b result:', 'PASS' if ok else 'FAIL', '| output:', out_dir)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
