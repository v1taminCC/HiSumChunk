# -*- coding: utf-8 -*-
"""Exports top-20 using the original 05_retrieve.py logic (writes only to the four-arm runs_top20/; original project read-only).

- Processes only the 4 books needing deeper retrieval (covering 11 excluded questions): Lord Jim, Sense and Sensibility,
  Tess of the D'Urbervilles、The Prince；
- Calls the original retrieve_in_book(book, 'ours', 20, 'zh'): same function, same validation, same normalization;
- Output schema identical to the original *.top10.json, written to runs_top20/{book}.ours.top20.json;
- md5 snapshots before and after, proving zero changes to the original project;
- Validation: top20[:10] of every question matches the original top10 item by item.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

FOUR = Path(__file__).resolve().parent.parent
ORIG = Path(__file__).resolve().parent.parent.parent.parent.parent / 'src' / 'retrieval'
OUT_DIR = FOUR / 'runs_top20'
BOOKS = ['Lord Jim', 'Sense and Sensibility', "Tess of the D'Urbervilles", 'The Prince']
TOP_K = 20
QUERY_VARIANT = 'zh'


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def snapshot() -> dict:
    files = []
    for book in BOOKS:
        for rel in (f'embeddings/by_book/{book}.queries.npy',
                    f'embeddings/by_book/{book}.queries.meta.json',
                    f'embeddings/by_book/{book}.ours.chunks.npy',
                    f'embeddings/by_book/{book}.ours.chunks.meta.json',
                    f'runs/in_book/{book}.ours.top10.json'):
            p = ORIG / rel
            files.append((rel, md5_file(p) if p.exists() else ''))
    return dict(files)


def load_originals():
    """Loads original-project modules in dependency order (file path -> original sys.modules name); returns (retrieve_in_book, saved)."""
    names = ['config', 'utils_io', 'utils_embedding', 'utils_book', 'utils_query_consistency']
    saved = {n: sys.modules.get(n) for n in names}
    for name in names:
        spec = importlib.util.spec_from_file_location(name, ORIG / f'{name}.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location('orig_05_retrieve', ORIG / '05_retrieve.py')
    mod05 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod05)
    return mod05.retrieve_in_book, saved


def restore(saved: dict) -> None:
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def main() -> int:
    before = snapshot()
    retrieve_in_book, saved = load_originals()
    try:
        results = {}
        for book in BOOKS:
            rows = retrieve_in_book(book, 'ours', TOP_K, QUERY_VARIANT)
            results[book] = rows
            write_json_atomic(OUT_DIR / f'{book}.ours.top20.json', rows)
            print(f'[Export] {book}: {len(rows)} questions -> {OUT_DIR.name}/{book}.ours.top20.json')
    finally:
        restore(saved)

    # Validation: top20[:10] matches the original top10 item by item
    bad = 0
    checked = 0
    max_score_diff = 0.0
    for book in BOOKS:
        old = {r['query_id']: r for r in json.load(open(ORIG / 'runs/in_book' / f'{book}.ours.top10.json', encoding='utf-8'))}
        new = {r['query_id']: r for r in results[book]}
        if set(old) != set(new):
            print(f'[ERROR] {book}: question sets differ old={len(old)} new={len(new)}')
            bad += 1
            continue
        for qid, orow in old.items():
            nrow = new[qid]
            oids = [t['chunk_id'] for t in orow['top_k']]
            nids = [t['chunk_id'] for t in nrow['top_k'][:len(oids)]]
            checked += 1
            if oids != nids:
                bad += 1
                print(f'[MISMATCH] {qid}')
                continue
            for t_old, t_new in zip(orow['top_k'], nrow['top_k'][:len(oids)]):
                max_score_diff = max(max_score_diff, abs(t_old['score'] - t_new['score']))
    print(f'Validation: {checked} questions, chunk-sequence mismatches {bad} (expect 0), max score diff {max_score_diff:.2e}')

    # Original-project md5 re-check
    after = snapshot()
    changed = [k for k in before if before[k] != after[k]]
    print(f'Original-project snapshot re-check: {len(before)} files, changes {len(changed)} (expect 0) {changed if changed else ""}')

    ok = bad == 0 and not changed
    print('Step1 result:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
