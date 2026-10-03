# -*- coding: utf-8 -*-
"""Comparison report: B/C/D arms across 577 questions

- Comparisons: B-C (remove H2 prefix while H1 present), D-C (remove H2 prefix with no H1), B-D (remove H1 slice with no prefix);
- Note: A exists only for the 200 sample; A-D / A-B listed separately (n=200).
Output: 05-reports/within_batch_577_report.txt
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DIMS, JUDG_EXT_DIR, JUDG_FILL18_DIR, JUDG_OUT_DIR, REPORTS_DIR, SAMPLE_DIR, SEED  # noqa: E402

ARMS = ['A', 'B', 'C', 'D']


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def load_all() -> dict[str, dict[str, dict]]:
    data = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for root in (JUDG_OUT_DIR, JUDG_EXT_DIR, JUDG_FILL18_DIR):
            for p in sorted((root / 'by_book').glob(f'*.four{arm}.3dim.json')):
                for j in read_json(p):
                    if not j.get('error'):
                        data[arm][j['query_id']] = j
    return data


def comp100(j: dict):
    vals = [j.get(d) for d in DIMS]
    if None in vals or any(v == 0 for v in vals):
        return None
    return sum(v - 1 for v in vals) / 12 * 100


def boot(deltas_by_book: dict[str, list[float]], B: int = 10000, seed: int = SEED):
    rng = random.Random(seed)
    books = list(deltas_by_book)
    n = sum(len(v) for v in deltas_by_book.values())
    obs = sum(sum(v) for v in deltas_by_book.values()) / n
    dist = []
    for _ in range(B):
        s = nn = 0
        for _b in range(len(books)):
            b = rng.choice(books)
            s += sum(deltas_by_book[b])
            nn += len(deltas_by_book[b])
        dist.append(s / nn)
    dist.sort()
    lo, hi = dist[int(B * .025)], dist[int(B * .975)]
    p = (min(sum(1 for x in dist if x >= 0), sum(1 for x in dist if x <= 0)) + 1) / (B + 1)
    return obs, lo, hi, p


def paired(four, x_arm, y_arm, dim, subset=None):
    db = defaultdict(list)
    for q, j in four[x_arm].items():
        if subset is not None and q not in subset:
            continue
        ref = four[y_arm].get(q)
        if not ref:
            continue
        x = comp100(j) if dim == 'comp100' else j.get(dim)
        y = comp100(ref) if dim == 'comp100' else ref.get(dim)
        if x is None or y is None:
            continue
        db[j['book_title']].append(float(x) - float(y))
    return boot(db)


four = load_all()
sample_json = read_json(SAMPLE_DIR / 'sample_200.json')
sample_qids = set(sample_json['sampled_query_ids'])
ext_qids = set(read_json(SAMPLE_DIR / 'remaining.json')['remaining_query_ids'])
fill_qids = set(sample_json['excluded_query_ids'])

lines = ['===== Comparison report (B/C/D; full 577) =====',
         'Direction: positive = the former is better. B-C=remove H2 prefix (H1 present); D-C=remove H2 prefix (no H1); B-D=remove H1 slice (no prefix).']
for name, subset in [('all 577', None), ('sample 200', sample_qids), ('extended 359', ext_qids), ('fill 18', fill_qids)]:
    lines.append(f'\n--- {name} ---')
    for x, y, label in [('B', 'C', 'B-C (prefix effect, H1 present)'),
                        ('D', 'C', 'D-C (prefix effect, no H1)'),
                        ('B', 'D', 'B-D (H1 effect, no prefix)')]:
        for dim in ['grounding', 'comp100']:
            obs, lo, hi, p = paired(four, x, y, dim, subset)
            flag = '***' if p < 0.01 else ('*' if p < 0.05 else '')
            lines.append(f'  {label} [{dim:>10}] Δ={obs:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.4f} {flag}')

lines.append('\n--- A-specific (sample 200 only) ---')
for x, y, label in [('A', 'B', 'A-B (H2 prefix effect, H1 present)'),
                    ('A', 'C', 'A-C (H1 effect, prefix present)'),
                    ('A', 'D', 'A-D (overall effect of both summary levels)')]:
    for dim in ['grounding', 'comp100']:
        obs, lo, hi, p = paired(four, x, y, dim, sample_qids)
        flag = '***' if p < 0.01 else ('*' if p < 0.05 else '')
        lines.append(f'  {label} [{dim:>10}] Δ={obs:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.4f} {flag}')

lines.append('\nNote: arms B/C/D cover all 577 questions; arm A covers only the 200 sample.'
             'For 11 of the 18 fill questions, C/D use top-20 for deeper retrieval; for 7 Rime questions, C/D use 4 blocks (the book has only 4 body blocks; the single protocol exception).')
report = '\n'.join(lines) + '\n'
out = REPORTS_DIR / 'within_batch_577_report.txt'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(report, encoding='utf-8')
print(report)
print('Wrote:', out)
