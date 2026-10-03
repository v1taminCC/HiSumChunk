# -*- coding: utf-8 -*-
"""Four-arm aggregation: three-dimension means + paired/book-clustered bootstrap + pre-registered criteria

Usage:
  python scripts/04_aggregate.py [--baseline auto] [--boot 10000]
  python scripts/04_aggregate.py --include-remaining            # merge remaining questions (559)
Output: 05-reports/{fourarm|fourarm559}_report.txt / _table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DIMS, JUDG_EXT_DIR, JUDG_FILL18_DIR, JUDG_OUT_DIR, OLD_JUDG_DIR, REPORTS_DIR, SAMPLE_DIR, SEED  # noqa: E402

ARMS = ['A', 'B', 'C', 'D']


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def load_fourarm(include_remaining: bool = False, include_fill18: bool = False) -> dict[str, dict[str, dict]]:
    data = {arm: {} for arm in ARMS}
    roots = [JUDG_OUT_DIR]
    if include_remaining:
        roots.append(JUDG_EXT_DIR)
    if include_fill18:
        roots.append(JUDG_FILL18_DIR)
    for arm in ARMS:
        for root in roots:
            for p in sorted((root / 'by_book').glob(f'*.four{arm}.3dim.json')):
                for j in read_json(p):
                    data[arm][j['query_id']] = j
    return data


def load_old_ours() -> dict[str, dict]:
    rows = {}
    for p in sorted(OLD_JUDG_DIR.glob('*.3dim.json')):
        for j in read_json(p):
            if j.get('scheme') == 'ours':
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default='auto', choices=['auto', 'A_new', 'A_old'])
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--include-remaining', action='store_true',
                    help='merge 04-judgments-remaining/ (run 01b and 02/03 --ext first)')
    ap.add_argument('--include-fill18', action='store_true',
                    help='merge 04-judgments-fill18/ and remaining (full 577)')
    ap.add_argument('--tag', default=None, help='report filename prefix (default fourarm / fourarm559 / fourarm577)')
    args = ap.parse_args()
    include_ext = args.include_remaining or args.include_fill18
    include_fill = args.include_fill18
    tag = args.tag or ('fourarm577' if include_fill else ('fourarm559' if include_ext else 'fourarm'))

    sample = read_json(SAMPLE_DIR / 'sample_200.json')
    qids = set(sample['sampled_query_ids'])
    if include_ext:
        remaining_path = SAMPLE_DIR / 'remaining.json'
        if not remaining_path.exists():
            print('[ERROR] 01-sample/remaining.json not found; run 01b_build_remaining_contexts.py first')
            return 1
        qids |= set(read_json(remaining_path)['remaining_query_ids'])
    if include_fill:
        qids |= set(sample['excluded_query_ids'])
    four = load_fourarm(include_remaining=include_ext, include_fill18=include_fill)
    old = {q: j for q, j in load_old_ours().items() if q in qids}

    lines, csv_rows = [], []
    n_arms = {arm: len(four[arm]) for arm in ARMS}
    lines.append(f'Four-arm judgment counts: ' + ', '.join(f'{a}={n_arms[a]}' for a in ARMS))

    # Baseline selection
    baseline_arm = None
    if args.baseline == 'A_new' or (args.baseline == 'auto' and n_arms['A'] >= len(qids)):
        baseline_arm = 'A'
        baseline = four['A']
        baseline_name = 'A'
    else:
        baseline = old
        baseline_name = 'A'
    lines.append(f'Baseline: {baseline_name}')

    # Arm means (per dimension + composite)
    lines.append('\n== Arm means ==')
    hdr = f"{'arm':>5} | " + ' | '.join(f'{d[:10]:>10}' for d in DIMS) + f" | {'comp100':>8}"
    lines.append(hdr)
    for arm in ARMS:
        recs = [j for j in four[arm].values() if not j.get('error')]
        if arm == 'A' and baseline_arm is None:
            recs = [j for j in old.values() if not j.get('error')]
        m = {d: mean([j.get(d) for j in recs]) for d in DIMS}
        cp = mean([comp100(j) for j in recs])
        lines.append(f'{arm:>5} | ' + ' | '.join(f'{m[d]:10.2f}' for d in DIMS) + f' | {cp:8.2f}')
        csv_rows.append({'arm': arm, **{d: round(m[d], 4) for d in DIMS}, 'comp100': round(cp, 3), 'n': len(recs)})

    # Paired comparisons: B/C/D vs. baseline (three dimensions + composite)
    lines.append(f'\n== Paired comparisons (vs {baseline_name}; book-clustered bootstrap B={args.boot}) ==')
    for arm in ['B', 'C', 'D']:
        for dim in DIMS + ['comp100']:
            db = defaultdict(list)
            for qid, j in four[arm].items():
                bj = baseline.get(qid)
                if not bj or j.get('error') or bj.get('error'):
                    continue
                a = comp100(j) if dim == 'comp100' else j.get(dim)
                b = comp100(bj) if dim == 'comp100' else bj.get(dim)
                if a is None or b is None:
                    continue
                db[j['book_title']].append(float(a) - float(b))
            if not db:
                continue
            obs, lo, hi, p = boot_pair(db, args.boot, SEED)
            flag = '***' if p < 0.01 else ('*' if p < 0.05 else '')
            lines.append(f'{arm} vs {baseline_name} [{dim:>20}] Δ={obs:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.4f} {flag}')

    # Pre-registered criteria
    lines.append('\n== Pre-registered criteria ==')
    try:
        results = {}
        for arm in ['B', 'C', 'D']:
            key = f'{arm}_vs_{baseline_name}_grounding'
            db = defaultdict(list)
            for qid, j in four[arm].items():
                bj = baseline.get(qid)
                if bj and not j.get('error') and not bj.get('error'):
                    db[j['book_title']].append(float(j.get('grounding', 0)) - float(bj.get('grounding', 0)))
            obs, lo, hi, p = boot_pair(db, args.boot, SEED)
            results[arm] = (obs, p)
        d_drop = -results['D'][0]
        b_drop = -results['B'][0]
        c_drop = -results['C'][0]
        lines.append(f'D drop={d_drop:.4f} (p={results["D"][1]:.4f}) | B drop={b_drop:.4f} | C drop={c_drop:.4f}')
        if d_drop >= 0.05 and b_drop < c_drop:
            lines.append('-> Verdict: mechanism supported (D>=0.05 and B>C)')
        elif d_drop >= 0.05:
            lines.append('-> Verdict: D drop meets threshold but B<=C; mechanism direction unsupported/ambiguous')
        elif d_drop >= 0.02:
            lines.append('-> Verdict: weakly supported (0.02<=D<0.05); must be written together with the retrieval-side story')
        else:
            lines.append('-> Verdict: mechanism rejected/unsupported (no drop or drop<0.02); switch to budget-efficiency/evaluation-analysis positioning')
    except Exception as exc:  # noqa: BLE001
        lines.append(f'Criterion computation failed (data may be incomplete): {exc}')

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = '\n'.join(lines)
    (REPORTS_DIR / f'{tag}_report.txt').write_text(report, encoding='utf-8')
    if csv_rows:
        with open(REPORTS_DIR / f'{tag}_table.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
