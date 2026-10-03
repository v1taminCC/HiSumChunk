# -*- coding: utf-8 -*-
"""Weight sensitivity + ordinal-mapping robustness (pure stdlib; outputs markdown)

Usage:
  python scripts/stats_02_sensitivity.py [--input ../data/scores_main.csv] [--out ./out]
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

DIMS = ['grounding', 'correctness', 'hallucination_absence']
WEIGHTS = {
    'equal 1/3,1/3,1/3': (1 / 3, 1 / 3, 1 / 3),
    'grounding-heavy 0.5,0.25,0.25': (0.5, 0.25, 0.25),
    'correctness-heavy 0.25,0.5,0.25': (0.25, 0.5, 0.25),
    'hallucination-heavy 0.25,0.25,0.5': (0.25, 0.25, 0.5),
    'w/o hallucination 0.5,0.5,0': (0.5, 0.5, 0.0),
}


def load(path: Path):
    data: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('scheme') == 'booksum_split':
                continue
            data[row['scheme']].append(row)
    return data


def valid(row: dict) -> bool:
    try:
        vals = [int(row[d]) for d in DIMS]
    except (KeyError, ValueError):
        return False
    return all(v > 0 for v in vals)


def weighted(row: dict, w) -> float:
    return sum(wi * (int(row[d]) - 1) / 4 * 100 for wi, d in zip(w, DIMS))


# Ordinal -> normal quantile mapping (5 equiprobability quantiles)
def _ppf(p: float) -> float:
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= 1 - pl:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


Q = [_ppf((i + 0.5) / 5) for i in range(5)]
ZMIN, ZMAX = Q[0], Q[-1]


def ordinal_score(row: dict) -> float:
    z = sum(Q[int(row[d]) - 1] for d in DIMS) / 3
    return (z - ZMIN) / (ZMAX - ZMIN) * 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(Path(__file__).parent.parent / 'data' / 'scores_main.csv'))
    ap.add_argument('--out', default=str(Path(__file__).parent / 'out'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(Path(args.input))
    order = ['ours', 'fixed', 'lumberchunker', 'lbdsemantic', 'semanticembed', 'raptor']
    schemes = [s for s in order if s in data] or sorted(data)

    lines = ['# Weight sensitivity and scale robustness', '']
    lines.append('## 1. Weight sensitivity (total score per scheme)')
    lines.append('')
    lines.append('| Weight set | ' + ' | '.join(schemes) + ' | Top-3 |')
    lines.append('|---|' + '---:|' * len(schemes) + '---|')
    for name, w in WEIGHTS.items():
        vals = {}
        for s in schemes:
            rs = [weighted(r, w) for r in data[s] if valid(r)]
            vals[s] = sum(rs) / len(rs)
        top3 = '>'.join(sorted(schemes, key=lambda s: -vals[s])[:3])
        lines.append('| ' + name + ' | ' + ' | '.join(f'{vals[s]:.2f}' for s in schemes) + f' | {top3} |')

    lines.append('')
    lines.append('## 2. Ordinal -> normal quantile mapping (vs. formula A)')
    lines.append('')
    lines.append('| Scheme | Linear A | Ordinal mapping |')
    lines.append('|---|---:|---:|')
    for s in schemes:
        la = [sum(int(r[d]) - 1 for d in DIMS) / 12 * 100 for r in data[s] if valid(r)]
        od = [ordinal_score(r) for r in data[s] if valid(r)]
        lines.append(f'| {s} | {sum(la)/len(la):.2f} | {sum(od)/len(od):.2f} |')

    report = '\n'.join(lines) + '\n'
    (out / 'sensitivity_report.md').write_text(report, encoding='utf-8')
    print(report)
    print('Output ->', out / 'sensitivity_report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
