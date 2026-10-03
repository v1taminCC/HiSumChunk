# -*- coding: utf-8 -*-
"""ctxretrieval aggregate report: same-protocol comparison across schemes

Usage:
  python scripts/08_report.py [--boot 10000]
Output:
  reports/cr_report.txt / cr_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHUNKS_DIR, DIMS, JUDG_DIR, REPORTS_DIR, SEED, SOURCE  # noqa: E402

SCHEMES = ['ours', 'ctxretrieval', 'fixed', 'lumberchunker', 'lbdsemantic', 'semanticembed', 'raptor']


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def load_ctx() -> dict[str, dict]:
    rows = {}
    for p in sorted((JUDG_DIR / 'by_book').glob('*.ctx.3dim.json')):
        for j in read_json(p):
            rows[j['query_id']] = j
    return rows


def load_main() -> dict[str, dict[str, dict]]:
    data: dict[str, dict[str, dict]] = defaultdict(dict)
    root = SOURCE / 'main_judgments' / 'by_book_3dim'
    for p in sorted(root.glob('*.3dim.json')):
        for j in read_json(p):
            scheme = j.get('scheme', '')
            if scheme in SCHEMES and scheme != 'ctxretrieval':
                data[scheme][j['query_id']] = j
    data['ctxretrieval'] = load_ctx()
    return data


def comp100(j: dict) -> float | None:
    vals = [j.get(d) for d in DIMS]
    if None in vals or any(v == 0 for v in vals):
        return None
    return sum(v - 1 for v in vals) / 12 * 100


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float('nan')


def boot_pair(deltas_by_book: dict[str, list[float]], B: int, seed: int):
    rng = random.Random(seed)
    books = list(deltas_by_book)
    n = sum(len(v) for v in deltas_by_book.values())
    obs = sum(sum(v) for v in deltas_by_book.values()) / n if n else float('nan')
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
    ap.add_argument('--boot', type=int, default=10000)
    args = ap.parse_args()

    data = load_main()
    qids = set(data['ctxretrieval'])

    lines = ['=== Contextual Retrieval baseline report ===', f'ctx judged {len(qids)} questions']
    lines.append('\n== Scheme means (same protocol) ==')
    lines.append(f"{'scheme':>14} | " + ' | '.join(f'{d[:12]:>12}' for d in DIMS) + f" | {'comp100':>8} | n")
    csv_rows = []
    for scheme in SCHEMES:
        recs = [j for qid, j in data[scheme].items() if qid in qids and not j.get('error')]
        m = {d: mean([j.get(d) for j in recs]) for d in DIMS}
        cp = mean([comp100(j) for j in recs])
        lines.append(f'{scheme:>14} | ' + ' | '.join(f'{m[d]:12.3f}' for d in DIMS) + f' | {cp:8.2f} | {len(recs)}')
        csv_rows.append({'scheme': scheme, **{d: round(m[d], 4) for d in DIMS},
                         'comp100': round(cp, 3), 'n': len(recs)})

    lines.append(f'\n== ours vs ctxretrieval (book-clustered bootstrap B={args.boot}) ==')
    lines.append(f"{'dim':>24} | {'Δ(ours-ctx)':>12} | {'95%CI':>20} | {'p':>6} | n")
    for dim in DIMS + ['comp100']:
        db = defaultdict(list)
        for qid in qids:
            a, b = data['ours'].get(qid), data['ctxretrieval'].get(qid)
            if not a or not b or a.get('error') or b.get('error'):
                continue
            va = comp100(a) if dim == 'comp100' else a.get(dim)
            vb = comp100(b) if dim == 'comp100' else b.get(dim)
            if va is None or vb is None:
                continue
            db[a.get('book_title', '')].append(va - vb)
        if db:
            obs, lo, hi, p = boot_pair(db, args.boot, SEED)
            lines.append(f'{dim:>24} | {obs:+12.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} | {sum(len(v) for v in db.values())}')

    wins = ties = losses = 0
    for qid in qids:
        a, b = data['ours'].get(qid), data['ctxretrieval'].get(qid)
        if not a or not b:
            continue
        va, vb = comp100(a), comp100(b)
        if va is None or vb is None:
            continue
        if abs(va - vb) < 1e-9:
            ties += 1
        elif va > vb:
            wins += 1
        else:
            losses += 1
    lines.append(f'\ncomp100 wins/ties/losses (ours vs ctx): {wins}/{ties}/{losses}')

    chunks = []
    for p in sorted(CHUNKS_DIR.glob('*.ctx.json')):
        if p.name.startswith('chunks_all'):
            continue
        chunks.extend(read_json(p))
    chars = [c['length'] for c in chunks]
    trunc = sum(1 for x in chars if x > 4096)
    lines.append('\n== Index size (ctx) ==')
    lines.append(f'units {len(chunks)} | chars mean/median/max: {statistics.mean(chars):.0f}/'
                 f'{statistics.median(chars):.0f}/{max(chars)} | >4096 chars (embedding truncation): {trunc} '
                 f'({trunc/len(chars)*100:.1f}%)')
    lines.append('Note: the ours index has 19,885 units (17,271 body + 2,614 H1); ctx is single-layer with 17,271.')

    report = '\n'.join(lines)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / 'cr_report.txt').write_text(report, encoding='utf-8')
    with open(REPORTS_DIR / 'cr_table.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scheme', *DIMS, 'comp100', 'n'])
        w.writeheader()
        w.writerows(csv_rows)
    print(report)
    print('\nWrote:', REPORTS_DIR / 'cr_report.txt')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
