#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summary faithfulness audit - step 5: natural judge-behaviour case list (zero API).

For units audited as unfaithful and actually retrieved into top-5:
  - List the questions that retrieved it (rank), the generated answer, and the judge's three scores;
  - Attach the problematic claims and entities identified by the LLM;
  - Provides lexical hints on whether the answer may repeat the error (entity/keyword overlap; for manual triage only).
Outputs: data/exposure/case_candidates.json + reports/judge_cases_draft.md

Usage: python scripts/05_judge_cases.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    ANSWERS_FILE, EXPOSURE_DIR, FULL_RUN_ANN, INVENTORY_DIR, MAIN_JUDG_DIR, PROMPT_VERSION, REPORTS_DIR,
)

DIMS = ['grounding', 'correctness', 'hallucination_absence']
BAD_CLAIM = {'unsupported', 'contradicted'}
BAD_ENTITY = {'missing', 'mismatch'}
STOP = set('''a an the of and or but in on at to for with from by as is are was were be been being
this that these those it its he she they his her their him them we our you your not no do does did
have has had will would can could may might must shall should there here when where which who whom
what why how than then so such into over under again further once during before after above below
up down out off only own same too very s t just now'''.split())


def load_units() -> dict[str, dict]:
    units = {}
    with open(INVENTORY_DIR / 'units_all.jsonl', encoding='utf-8') as f:
        for line in f:
            u = json.loads(line)
            units[u['unit_id']] = u
    return units


def load_annotations() -> dict[str, dict]:
    best: dict[str, dict] = {}
    paths = [FULL_RUN_ANN] if FULL_RUN_ANN.exists() else sorted(glob.glob(str(INVENTORY_DIR.parent / 'annotations' / '*.jsonl')))
    for p in paths:
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


def main() -> int:
    units = load_units()
    ann = load_annotations()
    question_map = json.loads((INVENTORY_DIR / 'question_map.json').read_text(encoding='utf-8'))
    answers = {r['query_id']: r for r in json.loads(ANSWERS_FILE.read_text(encoding='utf-8'))}
    judge = {}
    for f in glob.glob(str(MAIN_JUDG_DIR / '*.3dim.json')):
        for r in json.loads(Path(f).read_text(encoding='utf-8')):
            if r.get('scheme') == 'ours':
                judge[r['query_id']] = r

    # Unfaithful units -> retrieved questions
    hits: dict[str, list[str]] = {}
    for qid, q in question_map.items():
        for s in q['slots']:
            uid = s['unit_id']
            rec = ann.get(uid)
            if not rec:
                continue
            parsed = rec.get('parsed', {})
            has_bad = any(c.get('label') in BAD_CLAIM for c in parsed.get('claims', [])) or \
                      any(e.get('status') in BAD_ENTITY for e in parsed.get('entities', []))
            if has_bad:
                hits.setdefault(uid, []).append(qid)

    cases = []
    for uid, qids in sorted(hits.items()):
        u = units[uid]
        parsed = ann[uid].get('parsed', {})
        bad_claims = [c for c in parsed.get('claims', []) if c.get('label') in BAD_CLAIM]
        bad_entities = [e for e in parsed.get('entities', []) if e.get('status') in BAD_ENTITY]
        for qid in qids:
            slot = next(s for s in question_map[qid]['slots'] if s['unit_id'] == uid)
            ans = answers.get(qid, {})
            j = judge.get(qid, {})
            ans_text = ans.get('generated_answer', '') or ''
            # Lexical hint: how many content words of the problematic claims appear in the answer
            claim_words = set()
            for c in bad_claims:
                for w in re.findall(r"[A-Za-z][A-Za-z'’\-]+", c.get('claim', '')):
                    if w.lower() not in STOP and len(w) > 2:
                        claim_words.add(w.lower())
            ans_words = set(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'’\-]+", ans_text))
            overlap = sorted(claim_words & ans_words)
            cases.append({
                'unit_id': uid, 'level': u['level'], 'book': u['book'],
                'query_id': qid, 'rank': slot['rank'],
                'question': ans.get('question', ''),
                'generated_answer': ans_text,
                'judge': {d: j.get(d) for d in DIMS},
                'bad_claims': bad_claims,
                'bad_entities': bad_entities,
                'claim_word_overlap': overlap[:20],
                'summary': u['summary'],
            })

    EXPOSURE_DIR.mkdir(parents=True, exist_ok=True)
    (EXPOSURE_DIR / 'case_candidates.json').write_text(
        json.dumps(cases, ensure_ascii=False, indent=1), encoding='utf-8')

    lines = ['# Judge behaviour - natural case candidates (to be verified manually item by item)', '',
             f'Unfaithful units (retrieved into top-5): {len(hits)} | candidate question-unit pairs: {len(cases)}', '']
    for i, c in enumerate(cases, 1):
        lines.append(f'## Case {i}: {c["unit_id"]} @ question {c["query_id"]} (rank {c["rank"]})')
        lines.append(f'- Question: {c["question"]}')
        lines.append(f'- Judge scores: ' + ', '.join(f'{d}={c["judge"][d]}' for d in DIMS))
        lines.append(f'- Problematic claims:')
        for bc in c['bad_claims']:
            lines.append(f'    - [{bc.get("label")}] {bc.get("claim")} | note: {bc.get("note","")}')
        for be in c['bad_entities']:
            lines.append(f'    - [entity {be.get("status")}] {be.get("entity")}')
        lines.append(f'- Answer: {c["generated_answer"][:600]}')
        lines.append(f'- Lexical overlap hints: {", ".join(c["claim_word_overlap"]) or "(none)"}')
        lines.append('')
    (REPORTS_DIR / 'judge_cases_draft.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'[done] candidate units {len(hits)} | candidate pairs {len(cases)}')
    print(f'[out] {EXPOSURE_DIR/"case_candidates.json"}')
    print(f'[out] {REPORTS_DIR/"judge_cases_draft.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
