#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summary faithfulness audit - step 3: human verification export/import.

Export:
  - Positives (census): units the LLM labelled unsupported/contradicted (or with entity missing/mismatch);
  - Negatives (sampling): 15% random sample of units the LLM labelled clean (seed=2026);
  - Produces a CSV for reviewers to fill in item by item, plus JSON context (summary/source/LLM details).
  - Produces a separate R2 independent re-judgement sheet for positives (without the LLM verdict column).

Import:
  - Reads the filled CSV -> data/human/human_verdicts.json;
  - Computes R1/R2 positive agreement (Cohen's kappa, 3 classes).

Usage:
  python scripts/04_human_verify.py export --annot results/annotations/full_v3.jsonl
  python scripts/04_human_verify.py import
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FULL_RUN_ANN, HUMAN_DIR, INVENTORY_DIR, PROMPT_VERSION, SEED  # noqa: E402

BAD_CLAIM = {'unsupported', 'contradicted'}
BAD_ENTITY = {'missing', 'mismatch'}
NEG_RATIO = 0.15

CSV_COLUMNS = ['unit_id', 'group', 'llm_verdict', 'human_verdict', 'human_main_issue',
               'human_note', 'reviewer', 'date']
CSV_COLUMNS_R2 = ['unit_id', 'group', 'human_verdict', 'human_main_issue', 'human_note', 'reviewer', 'date']


def load_annotations(paths: list[Path]) -> dict[str, dict]:
    """Takes the last ok record per unit with a matching prompt_version."""
    best: dict[str, dict] = {}
    for p in paths:
        if not p.exists():
            continue
        with open(p, encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if rec.get('ok') and rec.get('prompt_version') == PROMPT_VERSION:
                    cur = best.get(rec['unit_id'])
                    if cur is None or (cur.get('calibration') and not rec.get('calibration')):
                        best[rec['unit_id']] = rec
    return best


def unit_flags(parsed: dict) -> tuple[bool, bool]:
    """Returns (primary: fact-level bad; strict: including interpretations/entities)."""
    fact_bad = any((c.get('label') in BAD_CLAIM and (c.get('claim_type') or 'fact') == 'fact')
                   for c in parsed.get('claims', []))
    any_bad = any((c.get('label') in BAD_CLAIM) for c in parsed.get('claims', []))
    bad_entity = any((e.get('status') in BAD_ENTITY) for e in parsed.get('entities', []))
    return fact_bad, (any_bad or bad_entity)


def load_units() -> dict[str, dict]:
    units = {}
    with open(INVENTORY_DIR / 'units_all.jsonl', encoding='utf-8') as f:
        for line in f:
            u = json.loads(line)
            units[u['unit_id']] = u
    return units


def export(annot_path: Path) -> int:
    ann = load_annotations([annot_path])
    units = load_units()
    positives, clean = [], []
    for uid, rec in ann.items():
        _fact_bad, strict_bad = unit_flags(rec.get('parsed', {}))
        (positives if strict_bad else clean).append(uid)
    positives.sort()
    rng = random.Random(SEED)
    n_neg = math.ceil(len(clean) * NEG_RATIO)
    negatives = sorted(rng.sample(clean, n_neg)) if clean else []

    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    context = []
    for group, ids in (('pos', positives), ('neg', negatives)):
        for uid in ids:
            u = units[uid]
            rec = ann[uid]
            context.append({
                'unit_id': uid, 'group': group, 'level': u['level'], 'book': u['book'], 'path': u['path'],
                'summary_wp': u['summary_wp'], 'source_wp': u['source_wp'],
                'summary': u['summary'], 'source': u['source'],
                'llm': rec.get('parsed', {}),
            })
    (HUMAN_DIR / 'review_context.json').write_text(
        json.dumps(context, ensure_ascii=False, indent=1), encoding='utf-8')

    def write_sheet(path: Path, rows: list[dict], cols: list[str]) -> None:
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    pos_rows, neg_rows, pos_r2 = [], [], []
    for uid in positives:
        rec = ann[uid]
        v = rec.get('parsed', {}).get('verdict', '')
        pos_rows.append({'unit_id': uid, 'group': 'pos', 'llm_verdict': v,
                         'human_verdict': '', 'human_main_issue': '', 'human_note': '', 'reviewer': '', 'date': ''})
        pos_r2.append({'unit_id': uid, 'group': 'pos', 'human_verdict': '', 'human_main_issue': '',
                       'human_note': '', 'reviewer': '', 'date': ''})
    for uid in negatives:
        rec = ann[uid]
        v = rec.get('parsed', {}).get('verdict', '')
        neg_rows.append({'unit_id': uid, 'group': 'neg', 'llm_verdict': v,
                         'human_verdict': '', 'human_main_issue': '', 'human_note': '', 'reviewer': '', 'date': ''})

    write_sheet(HUMAN_DIR / 'review_positives_r1.csv', pos_rows, CSV_COLUMNS)
    write_sheet(HUMAN_DIR / 'review_positives_r2.csv', pos_r2, CSV_COLUMNS_R2)
    write_sheet(HUMAN_DIR / 'review_negatives.csv', neg_rows, CSV_COLUMNS)

    print(f'[export] annotated units {len(ann)} | positives {len(positives)} | negative pool {len(clean)} | negative sample {len(negatives)}')
    print(f'[export] context: {HUMAN_DIR/"review_context.json"}')
    print(f'[export] R1 sheet: {HUMAN_DIR/"review_positives_r1.csv"}')
    print(f'[export] R2 sheet: {HUMAN_DIR/"review_positives_r2.csv"}')
    print(f'[export] negative sheet: {HUMAN_DIR/"review_negatives.csv"}')
    return 0


def _read_sheet(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            out[row['unit_id']] = row
    return out


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    pairs = [(a, b) for a, b in pairs if a and b]
    if not pairs:
        return float('nan')
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    idx = {l: i for i, l in enumerate(labels)}
    n = len(pairs)
    m = [[0] * len(labels) for _ in labels]
    for a, b in pairs:
        m[idx[a]][idx[b]] += 1
    po = sum(m[i][i] for i in range(len(labels))) / n
    pe = sum(sum(m[i]) * sum(m[r][i] for r in range(len(labels))) for i in range(len(labels))) / (n * n)
    return (po - pe) / (1 - pe) if pe != 1 else float('nan')


def import_sheets() -> int:
    r1 = _read_sheet(HUMAN_DIR / 'review_positives_r1.csv')
    r2 = _read_sheet(HUMAN_DIR / 'review_positives_r2.csv')
    neg = _read_sheet(HUMAN_DIR / 'review_negatives.csv')
    verdicts = {}
    for uid, row in {**neg, **r1}.items():
        verdicts[uid] = {
            'group': row.get('group', ''),
            'verdict_r1': row.get('human_verdict', '').strip().lower(),
            'main_issue': row.get('human_main_issue', '').strip(),
            'note': row.get('human_note', '').strip(),
            'reviewer': row.get('reviewer', '').strip(),
            'date': row.get('date', '').strip(),
        }
    for uid, row in r2.items():
        verdicts.setdefault(uid, {'group': 'pos'})
        verdicts[uid]['verdict_r2'] = row.get('human_verdict', '').strip().lower()
        verdicts[uid]['note_r2'] = row.get('human_note', '').strip()
    (HUMAN_DIR / 'human_verdicts.json').write_text(
        json.dumps(verdicts, ensure_ascii=False, indent=1), encoding='utf-8')
    pairs = [(v.get('verdict_r1', ''), v.get('verdict_r2', '')) for v in verdicts.values()
             if v.get('verdict_r1') and v.get('verdict_r2')]
    k = cohen_kappa(pairs)
    filled = sum(1 for v in verdicts.values() if v.get('verdict_r1'))
    print(f'[import] filled {filled} | R1/R2 double-annotated {len(pairs)} | Cohen kappa = {k:.3f}' if pairs else
          f'[import] filled {filled} | no double-annotated pairs')
    print(f'[import] Output: {HUMAN_DIR/"human_verdicts.json"}')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('export')
    p1.add_argument('--annot', default=str(FULL_RUN_ANN))
    sub.add_parser('import')
    args = ap.parse_args()
    if args.cmd == 'export':
        return export(Path(args.annot))
    return import_sheets()


if __name__ == '__main__':
    raise SystemExit(main())
