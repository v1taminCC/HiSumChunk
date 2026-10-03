# -*- coding: utf-8 -*-
"""Merges four-arm results into the ablation-style directory convention (by book x by arm, two mirrored trees):

  results/judgments/by_book_3dim/{book}.four{A}.3dim.json    # judgments: 21 books x 4 arms = 84
  data/answers/by_book/{book}.four{A}.top5.json              # answers: 21 books x 4 arms = 84

Protocol:
- A = main-experiment data (answers + judgments joined on query_id; scheme=='ours' only), 577 questions;
- B/C/D = four-arm data (sample 200 + remaining 359 + fill 18 = 577 questions);
- Each record gains arm / subset(sample200|remaining359|fill18) / data_source / variant(fourA..fourD);
- scheme normalized to fourA..fourD (A's original scheme 'ours' rewritten to 'fourA');
- Read-only sources; writes only the two directory trees above; re-runnable (overwrites only its own outputs).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    ANSWERS_EXT_DIR, ANSWERS_FILL18_DIR, ANSWERS_OUT_DIR,
    JUDG_EXT_DIR, JUDG_FILL18_DIR, JUDG_OUT_DIR,
    OLD_ANSWERS_DIR, OLD_JUDG_DIR, RESULTS_ANSWERS_DIR, RESULTS_JUDG_DIR, SAMPLE_DIR,
)

ARMS = 'ABCD'
JUDG_ROOTS = {'sample200': JUDG_OUT_DIR, 'remaining359': JUDG_EXT_DIR, 'fill18': JUDG_FILL18_DIR}
ANS_ROOTS = {'sample200': ANSWERS_OUT_DIR, 'remaining359': ANSWERS_EXT_DIR, 'fill18': ANSWERS_FILL18_DIR}


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def write_json_atomic(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def main() -> int:
    t0 = time.time()
    sample = read_json(SAMPLE_DIR / 'sample_200.json')
    sample_qids = set(sample['sampled_query_ids'])
    ext_qids = set(read_json(SAMPLE_DIR / 'remaining.json')['remaining_query_ids'])
    fill_qids = set(sample['excluded_query_ids'])
    all_qids = sample_qids | ext_qids | fill_qids

    def subset_of(qid: str) -> str:
        if qid in sample_qids:
            return 'sample200'
        if qid in ext_qids:
            return 'remaining359'
        return 'fill18'

    def decorate(rec: dict, arm: str, subset: str, data_source: str) -> dict:
        rec['arm'] = arm
        rec['subset'] = subset
        rec['data_source'] = data_source
        rec['variant'] = f'four{arm}'
        rec['scheme'] = f'four{arm}'
        return rec

    # ---- Collect: judg[arm][book] / ans[arm][book] ----
    judg = {a: defaultdict(list) for a in ARMS}
    ans = {a: defaultdict(list) for a in ARMS}

    # A: old main experiment (answers + judgments joined)
    old_ans, old_judg = {}, {}
    for p in sorted(OLD_ANSWERS_DIR.glob('*in_book.ours.top5.json')):
        for r in read_json(p):
            if r['query_id'] in all_qids:
                old_ans[r['query_id']] = r
    for p in sorted(OLD_JUDG_DIR.glob('*.3dim.json')):
        for r in read_json(p):
            if r.get('scheme') == 'ours' and r['query_id'] in all_qids:
                old_judg[r['query_id']] = r
    missing = [q for q in all_qids if q not in old_ans or q not in old_judg]
    if missing:
        print(f'[ERROR] Baseline data missing for {len(missing)} questions: {missing[:5]}')
        return 1
    for qid in all_qids:
        sub = subset_of(qid)
        a_rec = decorate(dict(old_ans[qid]), 'A', sub, 'main')
        j_rec = dict(old_ans[qid]); j_rec.update(old_judg[qid])
        j_rec['error'] = old_judg[qid].get('error') or old_ans[qid].get('error', '')
        j_rec = decorate(j_rec, 'A', sub, 'main')
        ans['A'][a_rec['book_title']].append(a_rec)
        judg['A'][j_rec['book_title']].append(j_rec)

    # B/C/D: new four-arm (three sources)
    for sub, root in JUDG_ROOTS.items():
        for arm in 'BCD':
            for p in sorted((root / 'by_book').glob(f'*.four{arm}.3dim.json')):
                for r in read_json(p):
                    rec = decorate(dict(r), arm, sub, 'fourarm')
                    judg[arm][rec['book_title']].append(rec)
    for sub, root in ANS_ROOTS.items():
        for arm in 'BCD':
            for p in sorted((root / 'by_book').glob(f'*.four{arm}.top5.json')):
                for r in read_json(p):
                    rec = decorate(dict(r), arm, sub, 'fourarm')
                    ans[arm][rec['book_title']].append(rec)

    # ---- Write both trees ----
    j_files = a_files = 0
    j_total = a_total = 0
    for arm in ARMS:
        for book in sorted(judg[arm]):
            rows = sorted(judg[arm][book], key=lambda r: r['query_id'])
            out = RESULTS_JUDG_DIR / f'{book}.four{arm}.3dim.json'
            write_json_atomic(out, rows)
            j_files += 1; j_total += len(rows)
        for book in sorted(ans[arm]):
            rows = sorted(ans[arm][book], key=lambda r: r['query_id'])
            out = RESULTS_ANSWERS_DIR / f'{book}.four{arm}.top5.json'
            write_json_atomic(out, rows)
            a_files += 1; a_total += len(rows)

    print(f'Judgment tree: {j_files} files / {j_total} records -> {RESULTS_JUDG_DIR}')
    print(f'Answer tree: {a_files} files / {a_total} records -> {RESULTS_ANSWERS_DIR}')
    print(f'Elapsed {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
