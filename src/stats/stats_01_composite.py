# -*- coding: utf-8 -*-
"""Percent composite score (formula A: (x-1)/4*100 per dimension, equal-weight average) + book-clustered paired bootstrap + Holm

Usage:
  python scripts/stats_01_composite.py [--input ../data/scores_main.csv] [--out ./out] [--boot 20000]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

DIMS = ['grounding', 'correctness', 'hallucination_absence']
SEED = 2026
BASELINE = 'ours'   # scheme name of the proposed method; pass --baseline fourA for four-arm CSVs


def comp100(row: dict) -> float | None:
    try:
        vals = [int(row[d]) for d in DIMS]
    except (KeyError, ValueError):
        return None
    if any(v == 0 for v in vals):
        return None
    return sum(v - 1 for v in vals) / 12 * 100


def load(path: Path) -> dict[str, dict[str, dict]]:
    """scheme -> query_id -> row"""
    data: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('scheme') == 'booksum_split':
                continue
            data[row['scheme']][row['query_id']] = row
    return data


def boot(deltas_by_book: dict[str, list[float]], B: int, rng: random.Random):
    books = list(deltas_by_book)
    n = sum(len(v) for v in deltas_by_book.values())
    obs = sum(sum(v) for v in deltas_by_book.values()) / n
    dist = []
    for _ in range(B):
        s, nn = 0.0, 0
        for _b in range(len(books)):
            b = rng.choice(books)
            s += sum(deltas_by_book[b])
            nn += len(deltas_by_book[b])
        dist.append(s / nn)
    dist.sort()
    lo, hi = dist[int(B * .025)], dist[int(B * .975)]
    p = (min(sum(1 for x in dist if x >= 0), sum(1 for x in dist if x <= 0)) + 1) / (B + 1)
    return obs, lo, hi, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(Path(__file__).parent.parent / 'data' / 'scores_main.csv'))
    ap.add_argument('--out', default=str(Path(__file__).parent / 'out'))
    ap.add_argument('--boot', type=int, default=20000)
    ap.add_argument('--baseline', default=BASELINE)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load(Path(args.input))
    scheme_order = ['ours', 'fixed', 'lumberchunker', 'lbdsemantic', 'semanticembed', 'raptor']
    schemes = [s for s in scheme_order if s in data] or sorted(data)

    lines = ['# Percent composite score (formula A: (x-1)/4*100 per dimension, equal-weight average)', '']
    lines.append(f'Input: {args.input} | bootstrap B={args.boot} | baseline={args.baseline}')
    lines.append('')
    lines.append('| Scheme | Composite (0-100) | n |')
    lines.append('|---|---:|---:|')
    means = {}
    for s in schemes:
        vals = [comp100(r) for r in data[s].values()]
        vals = [v for v in vals if v is not None]
        means[s] = sum(vals) / len(vals)
        lines.append(f'| {s} | {means[s]:.2f} | {len(vals)} |')

    base = args.baseline
    lines.append('')
    lines.append(f'## Paired comparisons (vs {base}; book-clustered; Holm-corrected)')
    lines.append('')
    lines.append('| Comparison | Δ(pts) | 95% CI | p | Holm p | Significant |')
    lines.append('|---|---:|---|---:|---:|---|')
    rng = random.Random(SEED)
    results = []
    base_rows = data[base]
    for s in schemes:
        if s == base:
            continue
        db: dict[str, list[float]] = defaultdict(list)
        for qid, r1 in base_rows.items():
            r2 = data[s].get(qid)
            if not r2:
                continue
            a, b = comp100(r1), comp100(r2)
            if a is None or b is None:
                continue
            db[r1['book']].append(a - b)
        if not db:
            continue
        obs, lo, hi, p = boot(db, args.boot, rng)
        results.append([s, obs, lo, hi, p])
    results.sort(key=lambda r: r[4])
    m = len(results)
    output = []
    for k, (s, obs, lo, hi, p) in enumerate(results):
        padj = min(1.0, p * (m - k))
        flag = '***' if padj < 0.01 else ('*' if padj < 0.05 else '')
        lines.append(f'| {base} vs {s} | {obs:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {p:.4f} | {padj:.4f} | {flag} |')
        output.append({'other': s, 'delta': round(obs, 3), 'ci_lo': round(lo, 3),
                       'ci_hi': round(hi, 3), 'p': round(p, 4), 'holm': round(padj, 4)})

    report = '\n'.join(lines) + '\n'
    (out_dir / 'composite_report.md').write_text(report, encoding='utf-8')
    (out_dir / 'composite_results.json').write_text(
        json.dumps({'means': {k: round(v, 3) for k, v in means.items()}, 'pairwise': output},
                   ensure_ascii=False, indent=2), encoding='utf-8')
    print(report)
    print('Output ->', out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
