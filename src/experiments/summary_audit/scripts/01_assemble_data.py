#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summary faithfulness audit - step 1: data assembly (zero API cost).

Builds:
  1) Full summary-unit inventory units_all.jsonl (H1 2,614 + H2 16,263 = 18,877)
     - H1: region h1_outline, source = concatenated member-block text
     - H2: block h2_outline, source = block text
  2) chunk -> unit mapping (19,885 chunks)
  3) question -> unit mapping (577 questions x top-5 = 2,885 slots)
  4) Component B (all 1,078 retrieved units)
  5) Component A (stratified sample of 400 non-retrieved units, seed=2026)
  6) Pilot 20 (including longest/shortest source extremes)

Read-only sources; outputs are written only to the summary_audit_experiment/ directory.
Usage: python scripts/01_assemble_data.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    ANSWERS_FILE, CHUNKS_FILE, COMPONENT_A_MIN_PER_STRATUM, COMPONENT_A_SIZE,
    FRAMES_DIR, INVENTORY_DIR, REPORTS_DIR, SEED, SPLIT_DIR,
)

WORD = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
PUNCT = re.compile(r"[^\w\s]")
INFO_RE = re.compile(r'Region\s*(\d+)---Block\s*(\d+)---Subchunk\s*(\d+)')


def wp(text: str) -> int:
    """Word + punctuation count (matches the project convention)."""
    return len(WORD.findall(text)) + len(PUNCT.findall(text))


def book_of(fname: str) -> str:
    return re.sub(r'_MergeRule.*$', '', fname)


def path_of(fname: str) -> str:
    return 'title' if '_Title_' in fname else 'notitle'


def main() -> int:
    for d in (INVENTORY_DIR, FRAMES_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Full unit inventory ----------
    units: dict[str, dict] = {}
    books: dict[str, dict] = {}
    for f in sorted(SPLIT_DIR.glob('*_L1Outline.json')):
        if f.name.endswith('_split.json'):
            continue
        book = book_of(f.name)
        path = path_of(f.name)
        d = json.loads(f.read_text(encoding='utf-8'))
        books[book] = {'path': path, 'file': f.name, 'n_regions': len(d.get('h2_regions', []))}
        for ri, region in enumerate(d.get('h2_regions', []), 1):
            blocks = region.get('chunks', [])
            h1_id = f'H1::{book}::R{ri}'
            starts = [b.get('source_start') for b in blocks if b.get('source_start') is not None]
            ends = [b.get('source_end') for b in blocks if b.get('source_end') is not None]
            chapters: list[str] = []
            for b in blocks:
                for cid in (b.get('covered_chapter_ids') or []):
                    if cid not in chapters:
                        chapters.append(cid)
            units[h1_id] = {
                'unit_id': h1_id,
                'level': 'H1',
                'book': book,
                'path': path,
                'region': ri,
                'block': None,
                'parent_h1': None,
                'summary': region.get('h1_outline', ''),
                'source': '\n\n'.join(b.get('text', '') for b in blocks),
                'summary_wp': wp(region.get('h1_outline', '')),
                'source_wp': sum(wp(b.get('text', '')) for b in blocks),
                'source_start': min(starts) if starts else None,
                'source_end': max(ends) if ends else None,
                'chapter_ids': chapters,
                'chunk_ids': [],
                'retrieved': False,
                'n_slots': 0,
                'n_questions': 0,
            }
            for bi, b in enumerate(blocks, 1):
                h2_id = f'H2::{book}::R{ri}::B{bi}'
                units[h2_id] = {
                    'unit_id': h2_id,
                    'level': 'H2',
                    'book': book,
                    'path': path,
                    'region': ri,
                    'block': bi,
                    'parent_h1': h1_id,
                    'summary': b.get('h2_outline', ''),
                    'source': b.get('text', ''),
                    'summary_wp': wp(b.get('h2_outline', '')),
                    'source_wp': wp(b.get('text', '')),
                    'source_start': b.get('source_start'),
                    'source_end': b.get('source_end'),
                    'chapter_ids': list(b.get('covered_chapter_ids') or []),
                    'chunk_ids': [],
                    'retrieved': False,
                    'n_slots': 0,
                    'n_questions': 0,
                }

    # ---------- 2) chunk -> unit mapping ----------
    chunks = json.loads(CHUNKS_FILE.read_text(encoding='utf-8'))
    chunks_by_book: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        chunks_by_book[c['book_title']].append(c)

    chunk_to_unit: dict[str, str] = {}
    check_ok = check_bad = 0
    for f in sorted(SPLIT_DIR.glob('*_L1Outline_split.json')):
        book = book_of(f.name)
        items = json.loads(f.read_text(encoding='utf-8'))['items']
        cs = chunks_by_book[book]
        assert len(items) == len(cs), f'{book}: items {len(items)} != chunks {len(cs)}'
        for it, ch in zip(items, cs):
            if it.get('is_region_summary') is True:
                ri = int(re.search(r'\d+', str(it.get('region_id'))).group())
                uid = f'H1::{book}::R{ri}'
            else:
                m = INFO_RE.match(it.get('chunk_info', ''))
                if not m:
                    check_bad += 1
                    continue
                ri, bi = int(m.group(1)), int(m.group(2))
                uid = f'H2::{book}::R{ri}::B{bi}'
            chunk_to_unit[ch['chunk_id']] = uid
            units[uid]['chunk_ids'].append(ch['chunk_id'])
            check_ok += 1

    # ---------- 3) question -> unit mapping ----------
    answers = json.loads(ANSWERS_FILE.read_text(encoding='utf-8'))
    question_map: dict[str, dict] = {}
    for rec in answers:
        qid = rec['query_id']
        slots = []
        for rank, cc in enumerate(rec['context_chunks'], 1):
            uid = chunk_to_unit.get(cc['chunk_id'])
            if uid is None:
                continue
            u = units[uid]
            u['retrieved'] = True
            u['n_slots'] += 1
            slots.append({
                'rank': rank,
                'chunk_id': cc['chunk_id'],
                'unit_id': uid,
                'level': u['level'],
            })
        seen = {s['unit_id'] for s in slots}
        for uid in seen:
            units[uid]['n_questions'] += 1
        question_map[qid] = {
            'query_id': qid,
            'book': rec['book_title'],
            'n_slots': len(slots),
            'slots': slots,
        }

    # ---------- 4) Component B (all retrieved) ----------
    retrieved_units = [uid for uid, u in units.items() if u['retrieved']]
    retrieved_units.sort(key=lambda x: (units[x]['book'], units[x]['level'], units[x]['region'],
                                        -1 if units[x]['block'] is None else units[x]['block']))

    # ---------- 5) Component A (stratified sample of non-retrieved) ----------
    non_retrieved = [uid for uid, u in units.items() if not u['retrieved']]
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for uid in non_retrieved:
        u = units[uid]
        strata[(u['level'], u['path'])].append(uid)
    total_nr = len(non_retrieved)
    alloc: dict[tuple[str, str], int] = {}
    for k, ids in strata.items():
        alloc[k] = max(COMPONENT_A_MIN_PER_STRATUM, int(round(COMPONENT_A_SIZE * len(ids) / total_nr)))
    # Adjust to the target total
    diff = COMPONENT_A_SIZE - sum(alloc.values())
    order = sorted(strata, key=lambda k: len(strata[k]), reverse=True)
    i = 0
    while diff != 0 and order:
        k = order[i % len(order)]
        if diff > 0:
            alloc[k] += 1
            diff -= 1
        elif alloc[k] > COMPONENT_A_MIN_PER_STRATUM:
            alloc[k] -= 1
            diff += 1
        i += 1

    rng = random.Random(SEED)
    component_a: list[dict] = []
    for k in sorted(strata):
        ids = sorted(strata[k])
        n = min(alloc[k], len(ids))
        picked = rng.sample(ids, n)
        for uid in picked:
            u = units[uid]
            component_a.append({
                'unit_id': uid,
                'level': u['level'],
                'book': u['book'],
                'path': u['path'],
                'summary_wp': u['summary_wp'],
                'source_wp': u['source_wp'],
                'stratum': f'{u["level"]}|{u["path"]}',
            })
    component_a.sort(key=lambda x: x['unit_id'])

    # ---------- 6) Pilot 20 ----------
    def pick_from(pool: list[str], level: str, path: str | None, n: int, exclude: set[str]) -> list[str]:
        cand = [u for u in pool
                if units[u]['level'] == level and (path is None or units[u]['path'] == path)
                and u not in exclude]
        cand.sort(key=lambda u: (units[u]['source_wp'], u))
        return rng.sample(cand, min(n, len(cand))) if cand else []

    pilot: list[dict] = []
    used: set[str] = set()

    def add(uid: str, reason: str) -> None:
        if uid in used:
            return
        used.add(uid)
        u = units[uid]
        pilot.append({
            'unit_id': uid, 'level': u['level'], 'book': u['book'], 'path': u['path'],
            'summary_wp': u['summary_wp'], 'source_wp': u['source_wp'], 'reason': reason,
        })

    h1_all = [u for u in retrieved_units if units[u]['level'] == 'H1']
    h2_all = [u for u in retrieved_units if units[u]['level'] == 'H2']
    h1_long = max(h1_all, key=lambda u: units[u]['source_wp'])
    h2_long = max(h2_all, key=lambda u: units[u]['source_wp'])
    h2_short = min(h2_all, key=lambda u: units[u]['source_wp'])
    add(h1_long, 'extreme: longest source')
    add(h2_long, 'extreme: longest source')
    add(h2_short, 'extreme: shortest source')
    for uid in pick_from(h1_all, 'H1', 'title', 2, used):
        add(uid, 'stratified random: H1|title')
    for uid in pick_from(h1_all, 'H1', 'notitle', 2, used):
        add(uid, 'stratified random: H1|notitle')
    for uid in pick_from(h2_all, 'H2', 'title', 5, used):
        add(uid, 'stratified random: H2|title')
    for uid in pick_from(h2_all, 'H2', 'notitle', 8, used):
        add(uid, 'stratified random: H2|notitle')

    # ---------- 7) Persist ----------
    with open(INVENTORY_DIR / 'units_all.jsonl', 'w', encoding='utf-8') as f:
        for uid in sorted(units):
            f.write(json.dumps(units[uid], ensure_ascii=False) + '\n')

    (INVENTORY_DIR / 'chunk_to_unit.json').write_text(
        json.dumps(chunk_to_unit, ensure_ascii=False, indent=1), encoding='utf-8')
    (INVENTORY_DIR / 'question_map.json').write_text(
        json.dumps(question_map, ensure_ascii=False, indent=1), encoding='utf-8')

    def frame_record(uid: str) -> dict:
        u = units[uid]
        return {
            'unit_id': uid, 'level': u['level'], 'book': u['book'], 'path': u['path'],
            'region': u['region'], 'block': u['block'],
            'summary_wp': u['summary_wp'], 'source_wp': u['source_wp'],
        }

    (FRAMES_DIR / 'b_retrieved_units.json').write_text(
        json.dumps([frame_record(u) for u in retrieved_units], ensure_ascii=False, indent=1), encoding='utf-8')
    (FRAMES_DIR / 'component_a_sample.json').write_text(
        json.dumps(component_a, ensure_ascii=False, indent=1), encoding='utf-8')
    (FRAMES_DIR / 'pilot_20.json').write_text(
        json.dumps(pilot, ensure_ascii=False, indent=1), encoding='utf-8')

    # ---------- 8) Report ----------
    lines = []
    lines.append('=== Summary faithfulness audit - data assembly report ===')
    lines.append(f'Books: {len(books)} | H1 units: {sum(1 for u in units.values() if u["level"]=="H1")}'
                 f' | H2 units: {sum(1 for u in units.values() if u["level"]=="H2")} | total: {len(units)}')
    lines.append(f'chunk->unit mapping: ok={check_ok} bad={check_bad} | total chunks={len(chunks)}')
    lines.append(f'question->unit mapping: questions={len(question_map)} | slots={sum(q["n_slots"] for q in question_map.values())}')
    lines.append(f'Component B (all retrieved): {len(retrieved_units)} records'
                 f' | H1={sum(1 for u in retrieved_units if units[u]["level"]=="H1")}'
                 f' | H2={sum(1 for u in retrieved_units if units[u]["level"]=="H2")}')
    lines.append(f'Component A (non-retrieved sample): {len(component_a)} records'
                 f' | strata={dict(sorted(Counter(x["stratum"] for x in component_a).items()))}')
    pilot_strata = Counter(f'{x["level"]}|{x["path"]}' for x in pilot)
    lines.append(f'Pilot 20: {len(pilot)} records | strata={dict(sorted(pilot_strata.items()))}')
    lines.append('')
    lines.append('Retrieved H1/H2 source length (words + punctuation):')
    for lvl in ('H1', 'H2'):
        xs = sorted(units[u]['source_wp'] for u in retrieved_units if units[u]['level'] == lvl)
        if xs:
            lines.append(f'  {lvl}: n={len(xs)} min={xs[0]} p50={xs[len(xs)//2]} max={xs[-1]}'
                         f' | >5k={sum(1 for x in xs if x > 5000)} >10k={sum(1 for x in xs if x > 10000)}')
    lines.append('')
    lines.append('Pilot 20 detail:')
    for x in sorted(pilot, key=lambda r: (r['level'], r['path'], r['source_wp'])):
        lines.append(f'  {x["unit_id"]:<42} src={x["source_wp"]:>6} sum={x["summary_wp"]:>4} {x["reason"]}')
    report = '\n'.join(lines)
    (REPORTS_DIR / 'inventory_report.txt').write_text(report, encoding='utf-8')
    print(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
