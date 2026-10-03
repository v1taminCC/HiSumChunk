# -*- coding: utf-8 -*-
"""Placebo-arm context construction

Replaces each body block's real H2 prefix in arm A with a cross-book, length-matched, content-irrelevant H2 summary;
H1 slices, block order, and the retrieval set are left untouched.

Usage:
  python scripts/01_build_placebo_contexts.py
Output:
  data/contexts/contexts_P_{main,remaining,fill18}.json
  data/contexts/contexts_P.json          # merged (577 questions)
  data/placebo_mapping.json              # full replacement mapping (auditable)
  reports/placebo_audit.txt
"""
from __future__ import annotations

import bisect
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    CHUNK_PREFIX, CONTEXTS_A, CONTEXTS_DIR, H2_POOL, PLACEBO_CAND_N, PLACEBO_LEN_TOL,
    PUNCT, REPORTS_DIR, SEED,
)


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def read_jsonl(p: Path):
    rows = []
    with open(p, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_prefix(text: str) -> str:
    if not text:
        return ''
    return text if text[-1] in PUNCT else text + '。'


def main() -> int:
    pool = read_jsonl(H2_POOL)
    pool_sorted = sorted(pool, key=lambda x: (x['clen'], x['book'], x['region'], x['block']))
    lens = [x['clen'] for x in pool_sorted]
    cpf = read_json(CHUNK_PREFIX)

    all_rows, mapping, audit_rows = [], [], []
    stats = Counter()
    real_lens, placebo_lens = [], []
    placebo_books = Counter()

    for tag, path in CONTEXTS_A.items():
        src_rows = read_json(path)
        out_rows = []
        for row in src_rows:
            qid, book = row['query_id'], row['book_title']
            new_chunks = []
            for c in row['chunks']:
                if c['is_outline']:
                    new_chunks.append(c)
                    continue
                info = cpf.get(c['chunk_id'])
                prefix = ensure_prefix(info['prefix']) if info and info['prefix'] else ''
                old_text = c['text']
                if not prefix or not old_text.startswith(prefix):
                    stats['fallback_keep'] += 1
                    new_chunks.append(c)
                    continue
                real_len = len(prefix)
                lo, hi = real_len * (1 - PLACEBO_LEN_TOL), real_len * (1 + PLACEBO_LEN_TOL)
                i, j = bisect.bisect_left(lens, lo), bisect.bisect_right(lens, hi)
                cands = [x for x in pool_sorted[i:j] if x['book'] != book]
                if not cands:
                    stats['fallback_global'] += 1
                    cands = sorted(
                        (x for x in pool_sorted if x['book'] != book),
                        key=lambda x: abs(x['clen'] - real_len),
                    )[:200]
                cands.sort(key=lambda x: abs(x['clen'] - real_len))
                rng = random.Random(f'{SEED}|{qid}|{c["chunk_id"]}')
                picked = rng.choice(cands[:PLACEBO_CAND_N])
                placebo = ensure_prefix(picked['text'])
                new_text = placebo + old_text[len(prefix):]
                new_chunks.append({**c, 'text': new_text})
                stats['replaced'] += 1
                real_lens.append(real_len)
                placebo_lens.append(len(placebo))
                placebo_books[picked['book']] += 1
                mapping.append({
                    'query_id': qid, 'book_title': book, 'chunk_id': c['chunk_id'],
                    'region': info['region'], 'block': info['block'],
                    'real_prefix': prefix, 'real_len': real_len,
                    'placebo_book': picked['book'], 'placebo_prefix': placebo,
                    'placebo_len': len(placebo),
                })
            out_rows.append({**row, 'chunks': new_chunks,
                             'meta': {**row.get('meta', {}), 'arm': 'P', 'placebo': True}})
        out_path = CONTEXTS_DIR / f'contexts_P_{tag}.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_rows, ensure_ascii=False), encoding='utf-8')
        all_rows.extend(out_rows)
        audit_rows.append(f'[{tag}] questions {len(out_rows)} | body blocks {sum(1 for r in out_rows for c in r["chunks"] if not c["is_outline"])}')

    combined = CONTEXTS_DIR / 'contexts_P.json'
    combined.write_text(json.dumps(all_rows, ensure_ascii=False), encoding='utf-8')
    (CONTEXTS_DIR.parent / 'placebo_mapping.json').write_text(
        json.dumps(mapping, ensure_ascii=False), encoding='utf-8')

    lines = ['=== Placebo-arm construction audit ===']
    lines += audit_rows
    lines.append(f'Total questions: {len(all_rows)}')
    lines.append(f'Replaced: {stats["replaced"]} | no candidate in length band, global fallback: {stats["fallback_global"]} | prefix mismatch, kept original: {stats["fallback_keep"]}')
    bad_src = sum(1 for m in mapping if m['placebo_book'] == m['book_title'])
    if real_lens:
        lines.append(f'Real prefix char length mean/median: {statistics.mean(real_lens):.1f}/{statistics.median(real_lens):.0f}')
        lines.append(f'Placebo prefix char length mean/median: {statistics.mean(placebo_lens):.1f}/{statistics.median(placebo_lens):.0f}')
        lines.append(f'Placebo source books: {len(placebo_books)} | records sourced from the same book: {bad_src} (expect 0)')
    lines.append('Example mapping:')
    for m in mapping[:5]:
        lines.append(f'  [{m["book_title"]}] {m["chunk_id"]}')
        lines.append(f'    real: {m["real_prefix"][:70]}')
        lines.append(f'    placebo: {m["placebo_prefix"][:70]}  (from {m["placebo_book"]})')
    report = '\n'.join(lines)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / 'placebo_audit.txt').write_text(report, encoding='utf-8')
    print(report)

    ok = (stats['fallback_keep'] == 0 and len(all_rows) == 577 and bad_src == 0)
    print('\nG1 result:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
