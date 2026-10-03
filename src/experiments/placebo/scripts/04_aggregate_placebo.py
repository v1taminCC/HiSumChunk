# -*- coding: utf-8 -*-
"""Placebo-arm aggregation: P vs A/B/C/D (+ old ours baseline), book-clustered bootstrap

Usage:
  python scripts/04_aggregate_placebo.py [--boot 10000]
Output:
  reports/placebo_report.txt / placebo_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    DIMS, JUDG_DIR, JUDG_DIRS, OLD_OURS_DIR, REPORTS_DIR, SAMPLE_200, REMAINING, SEED,
)

ARMS = ['P', 'A', 'B', 'C', 'D']


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def load_p() -> dict[str, dict]:
    rows = {}
    for p in sorted((JUDG_DIR / 'by_book').glob('*.fourP.3dim.json')):
        for j in read_json(p):
            rows[j['query_id']] = j
    return rows


def load_fourarm() -> dict[str, dict[str, dict]]:
    data: dict[str, dict[str, dict]] = {arm: {} for arm in 'ABCD'}
    for _tag, root in JUDG_DIRS.items():
        for arm in 'ABCD':
            for p in sorted(root.glob(f'*.four{arm}.3dim.json')):
                for j in read_json(p):
                    data[arm][j['query_id']] = j
    return data


def load_old_ours() -> dict[str, dict]:
    rows = {}
    for p in sorted(OLD_OURS_DIR.glob('*.3dim.json')):
        for j in read_json(p):
            rows[j['query_id']] = j
    return rows


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


def compare(name: str, data_a: dict, data_b: dict, qids: set, boot: int, lines: list):
    lines.append(f'\n== {name} (book-clustered bootstrap B={boot}) ==')
    lines.append(f"{'dim':>24} | {'Δ':>7} | {'95%CI':>18} | {'p':>6} | n")
    for dim in DIMS + ['comp100']:
        db = defaultdict(list)
        for qid in qids:
            ja, jb = data_a.get(qid), data_b.get(qid)
            if not ja or not jb or ja.get('error') or jb.get('error'):
                continue
            va = comp100(ja) if dim == 'comp100' else ja.get(dim)
            vb = comp100(jb) if dim == 'comp100' else jb.get(dim)
            if va is None or vb is None:
                continue
            db[ja.get('book_title', '')].append(va - vb)
        if not db:
            continue
        obs, lo, hi, p = boot_pair(db, boot, SEED)
        lines.append(f'{dim:>24} | {obs:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} | {sum(len(v) for v in db.values())}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=10000)
    args = ap.parse_args()

    p_rows = load_p()
    four = load_fourarm()
    old = load_old_ours()

    sample = read_json(SAMPLE_200)
    remaining = read_json(REMAINING)
    qids_all = set(sample['sampled_query_ids']) | set(remaining['remaining_query_ids']) | set(sample['excluded_query_ids'])
    qids_200 = set(sample['sampled_query_ids'])

    lines = ['=== Placebo-arm aggregate report ===', f'P judged {len(p_rows)} | target {len(qids_all)}']
    lines.append('\n== Arm means (577 protocol) ==')
    lines.append(f"{'arm':>4} | " + ' | '.join(f'{d[:12]:>12}' for d in DIMS) + f" | {'comp100':>8} | n")
    csv_rows = []
    for arm in ARMS:
        src = p_rows if arm == 'P' else four[arm]
        recs = [j for qid, j in src.items() if qid in qids_all and not j.get('error')]
        m = {d: mean([j.get(d) for j in recs]) for d in DIMS}
        cp = mean([comp100(j) for j in recs])
        lines.append(f'{arm:>4} | ' + ' | '.join(f'{m[d]:12.3f}' for d in DIMS) + f' | {cp:8.2f} | {len(recs)}')
        csv_rows.append({'arm': arm, **{d: round(m[d], 4) for d in DIMS}, 'comp100': round(cp, 3), 'n': len(recs)})
    recs = [j for qid, j in old.items() if qid in qids_all and not j.get('error')]
    m = {d: mean([j.get(d) for j in recs]) for d in DIMS}
    cp = mean([comp100(j) for j in recs])
    lines.append(f'{"A_old":>4} | ' + ' | '.join(f'{m[d]:12.3f}' for d in DIMS) + f' | {cp:8.2f} | {len(recs)}')
    csv_rows.append({'arm': 'A_old', **{d: round(m[d], 4) for d in DIMS}, 'comp100': round(cp, 3), 'n': len(recs)})

    compare('P vs A_new (200 subset)', p_rows, four['A'], qids_200, args.boot, lines)
    compare('P vs A_old（577）', p_rows, old, qids_all, args.boot, lines)
    compare('P vs B (577 available)', p_rows, four['B'], qids_all, args.boot, lines)
    compare('P vs C (577 available)', p_rows, four['C'], qids_all, args.boot, lines)
    compare('P vs D (577 available)', p_rows, four['D'], qids_all, args.boot, lines)

    report = '\n'.join(lines)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / 'placebo_report.txt').write_text(report, encoding='utf-8')
    with open(REPORTS_DIR / 'placebo_table.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['arm', *DIMS, 'comp100', 'n'])
        w.writeheader()
        w.writerows(csv_rows)
    print(report)
    print('\nWrote:', REPORTS_DIR / 'placebo_report.txt')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
