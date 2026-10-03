# -*- coding: utf-8 -*-
"""Source snapshot integrity check (reads only data/source in this directory)"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    CHUNK_PREFIX, CONTEXTS_A, H2_POOL, JUDG_DIRS, OLD_OURS_DIR, QA_DRAFTS, REMAINING,
    SAMPLE_200,
)


def read_jsonl(p: Path):
    rows = []
    with open(p, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ok = True
    problems = []

    # 1) contexts_A three roots
    for tag, p in CONTEXTS_A.items():
        if not p.exists():
            problems.append(f'Missing {tag}: {p}')
            ok = False
            continue
        rows = json.loads(p.read_text(encoding='utf-8'))
        n_body = sum(1 for r in rows for c in r['chunks'] if not c['is_outline'])
        n_h1 = sum(1 for r in rows for c in r['chunks'] if c['is_outline'])
        print(f'contexts_A[{tag}]: questions {len(rows)} | body blocks {n_body} | H1 slices {n_h1}')

    # 2) A-D judgment three roots
    for tag, d in JUDG_DIRS.items():
        files = sorted(d.glob('*.four[ABCD].3dim.json'))
        arms = {}
        for p in files:
            m = re.search(r'\.four([ABCD])\.3dim\.json$', p.name)
            arms[m.group(1)] = arms.get(m.group(1), 0) + 1
        print(f'judgments[{tag}]: files {len(files)} | arms {arms}')
        if len(files) == 0:
            problems.append(f'Judgments missing: {tag}')
            ok = False

    # 3) Old ours
    old_files = sorted(OLD_OURS_DIR.glob('*.3dim.json'))
    n_old = sum(len(json.loads(p.read_text(encoding='utf-8'))) for p in old_files)
    print(f'old_ours: files {len(old_files)} | rows {n_old} (expect 577)')
    if n_old != 577:
        problems.append(f'old_ours rows {n_old} != 577')
        ok = False

    # 4) H2 pool
    pool = read_jsonl(H2_POOL)
    print(f'H2 pool: {len(pool)} records (expect 16261), {len({r["book"] for r in pool})} books')
    if len(pool) != 16261:
        problems.append(f'H2 pool {len(pool)} != 16261')
        ok = False

    # 5) Sampling / QA
    s = json.loads(SAMPLE_200.read_text(encoding='utf-8'))
    r = json.loads(REMAINING.read_text(encoding='utf-8'))
    qa = json.loads(QA_DRAFTS.read_text(encoding='utf-8'))
    n_qa = sum(len(row.get('qas', []) or []) for row in qa)
    print(f'sample200={s["sample_size"]} | remaining={len(r["remaining_query_ids"])} | '
          f'excluded(fill18)={len(s["excluded_query_ids"])} | QA drafts {n_qa}')
    if s['sample_size'] != 200 or len(r['remaining_query_ids']) != 359 or len(s['excluded_query_ids']) != 18:
        problems.append('Abnormal sample sizes (expect 200/359/18)')
        ok = False

    # 6) Prefix-strip spot check (chunk_prefix snapshot; checks body blocks of the first 20 A-contexts)
    cpf = json.loads(CHUNK_PREFIX.read_text(encoding='utf-8'))
    print(f'chunk_prefix snapshot: {len(cpf)} blocks | no prefix {sum(1 for v in cpf.values() if not v["ok"])}')
    rows = json.loads(CONTEXTS_A['main'].read_text(encoding='utf-8'))
    checked = hit = 0
    for row in rows[:20]:
        for c in row['chunks']:
            if c['is_outline']:
                continue
            info = cpf.get(c['chunk_id'])
            if not info or not info['prefix']:
                continue
            checked += 1
            if c['text'].startswith(info['prefix']):
                hit += 1
    print(f'Prefix-strip spot check: {hit}/{checked}')
    if checked == 0 or hit != checked:
        problems.append(f'Prefix-strip spot check abnormal {hit}/{checked}')
        ok = False

    print()
    if ok:
        print('G0 check: PASS')
        return 0
    print('G0 check: FAIL')
    for x in problems:
        print(' -', x)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
