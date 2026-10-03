# -*- coding: utf-8 -*-
"""Exports slim tables for the statistical pipeline (for external server use; no full text, scores only)

Usage:
  python scripts/10_export_slim_stats.py                # main experiment, 6 schemes
  python scripts/10_export_slim_stats.py --fourarm      # append the four arms (run after they finish)
Output: stats_package/../data/scores_main.csv (and scores_fourarm.csv)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DIMS, JUDG_OUT_DIR, OLD_JUDG_DIR, STATS_PKG  # noqa: E402

HEADER = ['scheme', 'query_id', 'book', 'question_type', 'difficulty', *DIMS, 'any_zero']


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def row_of(j: dict) -> dict:
    dims = {d: j.get(d) for d in DIMS}
    return {
        'scheme': j.get('scheme', ''), 'query_id': j.get('query_id', ''),
        'book': j.get('book_title', ''), 'question_type': j.get('question_type', ''),
        'difficulty': j.get('difficulty', ''),
        **{d: dims[d] for d in DIMS},
        'any_zero': int(any(v == 0 for v in dims.values() if v is not None)),
    }


def export_main() -> Path:
    rows = []
    for p in sorted(OLD_JUDG_DIR.glob('*.3dim.json')):
        for j in read_json(p):
            rows.append(row_of(j))
    out = STATS_PKG / 'data' / 'scores_main.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f'scores_main.csv: {len(rows)} rows -> {out}')
    return out


def export_fourarm() -> Path:
    rows = []
    for arm in 'ABCD':
        for p in sorted((JUDG_OUT_DIR / 'by_book').glob(f'*.four{arm}.3dim.json')):
            for j in read_json(p):
                rows.append(row_of(j))
    out = STATS_PKG / 'data' / 'scores_fourarm.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f'scores_fourarm.csv: {len(rows)} rows -> {out}')
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', action='store_true', default=True)
    ap.add_argument('--fourarm', action='store_true')
    args = ap.parse_args()
    export_main()
    if args.fourarm:
        export_fourarm()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
