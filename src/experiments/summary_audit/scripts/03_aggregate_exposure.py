#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summary faithfulness audit - step 4: aggregation and exposure (dual-criterion).

Input: data/annotations/*.jsonl (takes only the last ok record with prompt_version == config.PROMPT_VERSION)
      + optional data/human/human_verdicts.json
Output:
  - reports/audit_summary.txt / .json
  - data/exposure/question_flags.json (per-question exposure flags for all 577 questions, dual-criterion)

Protocol:
  - Primary criterion: any claim_type=='fact' with label in {unsupported, contradicted}
  - Strict criterion: any claim (including interpretation) bad, or any entity missing/mismatch
Usage: python scripts/03_aggregate_exposure.py
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    ANNOT_DIR, EXPOSURE_DIR, FULL_RUN_ANN, HUMAN_DIR, INVENTORY_DIR, MAIN_JUDG_DIR,
    PROMPT_VERSION, REPORTS_DIR, SEED,
)

DIMS = ['grounding', 'correctness', 'hallucination_absence']
BAD = {'unsupported', 'contradicted'}
BAD_ENTITY = {'missing', 'mismatch'}
BOOT = 20000


def load_annotations(paths: list[Path]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p, encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if rec.get('ok') and rec.get('prompt_version') == PROMPT_VERSION:
                    cur = best.get(rec['unit_id'])
                    # If a unit has calibration duplicate annotations, the primary annotation wins
                    if cur is None or (cur.get('calibration') and not rec.get('calibration')):
                        best[rec['unit_id']] = rec
    return best


def load_units() -> dict[str, dict]:
    units = {}
    with open(INVENTORY_DIR / 'units_all.jsonl', encoding='utf-8') as f:
        for line in f:
            u = json.loads(line)
            units[u['unit_id']] = u
    return units


def classify(parsed: dict) -> dict:
    claims = parsed.get('claims', []) or []
    entities = parsed.get('entities', []) or []
    n_claims = len(claims)
    n_fact_bad = sum(1 for c in claims
                     if c.get('label') in BAD and (c.get('claim_type') or 'fact') == 'fact')
    n_fact_contr = sum(1 for c in claims
                       if c.get('label') == 'contradicted' and (c.get('claim_type') or 'fact') == 'fact')
    n_any_bad = sum(1 for c in claims if c.get('label') in BAD)
    n_bad_ent = sum(1 for e in entities if e.get('status') in BAD_ENTITY)
    primary = n_fact_bad > 0
    strict = (n_any_bad > 0) or (n_bad_ent > 0)
    return {
        'n_claims': n_claims,
        'n_fact_bad': n_fact_bad,
        'n_fact_contradicted': n_fact_contr,
        'n_any_bad': n_any_bad,
        'n_bad_entity': n_bad_ent,
        'primary_unfaithful': primary,
        'strict_unfaithful': strict,
    }


def apply_human(llm: dict, human: dict | None) -> tuple[bool, bool, str]:
    """Returns (primary, strict, source); human adjudication takes precedence and applies to both criteria."""
    if not human:
        return llm['primary_unfaithful'], llm['strict_unfaithful'], 'llm'
    v = (human.get('verdict_r1') or '').strip().lower()
    if not v:
        return llm['primary_unfaithful'], llm['strict_unfaithful'], 'llm'
    if v.startswith('clean'):
        return False, False, 'human'
    if v.startswith('contrad') or v.startswith('unsup'):
        return True, True, 'human'
    return llm['primary_unfaithful'], llm['strict_unfaithful'], 'llm'


def boot_rate(by_book: dict[str, list[bool]], B: int = BOOT, seed: int = SEED) -> tuple[float, float, float]:
    books = list(by_book)
    n = sum(len(v) for v in by_book.values())
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    obs = sum(sum(v) for v in by_book.values()) / n
    rng = random.Random(seed)
    dist = []
    for _ in range(B):
        s = nn = 0
        for _b in range(len(books)):
            b = rng.choice(books)
            s += sum(by_book[b])
            nn += len(by_book[b])
        dist.append(s / nn if nn else 0.0)
    dist.sort()
    return obs, dist[int(B * .025)], dist[int(B * .975)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--annot', default=str(FULL_RUN_ANN),
                    help='annotation files (comma-separated; defaults to the full formal run)')
    args = ap.parse_args()
    paths = [Path(p) for p in args.annot.split(',') if p]
    ann = load_annotations(paths)
    print(f'[input] {", ".join(str(p) for p in paths)} | valid units {len(ann)}')
    units = load_units()
    question_map = json.loads((INVENTORY_DIR / 'question_map.json').read_text(encoding='utf-8'))
    human = {}
    hp = HUMAN_DIR / 'human_verdicts.json'
    if hp.exists():
        human = json.loads(hp.read_text(encoding='utf-8'))

    judge_scores = {}
    for f in glob.glob(str(MAIN_JUDG_DIR / '*.3dim.json')):
        for r in json.loads(Path(f).read_text(encoding='utf-8')):
            if r.get('scheme') == 'ours':
                judge_scores[r['query_id']] = {d: r.get(d) for d in DIMS}

    unit_result: dict[str, dict] = {}
    for uid, rec in ann.items():
        cls = classify(rec.get('parsed', {}))
        primary, strict, src = apply_human(cls, human.get(uid))
        unit_result[uid] = {**cls, 'primary': primary, 'strict': strict, 'verdict_source': src,
                            'level': rec.get('level') or units[uid]['level'],
                            'book': units[uid]['book'], 'path': units[uid]['path'],
                            'model': rec.get('model', '')}

    def rate_block(uids: list[str], key: str = 'primary') -> dict:
        kept = [u for u in uids if u in unit_result]
        vals = [unit_result[u][key] for u in kept]
        by_book: dict[str, list[bool]] = defaultdict(list)
        for u in kept:
            by_book[unit_result[u]['book']].append(unit_result[u][key])
        obs, lo, hi = boot_rate(by_book)
        return {'n': len(vals), 'n_unfaithful': sum(vals), 'rate': obs, 'ci_lo': lo, 'ci_hi': hi}

    lines = ['=== Summary faithfulness audit - aggregate report ===',
             f'Protocol version: {PROMPT_VERSION} | annotated units: {len(ann)}'
             f' | human verification coverage: {sum(1 for v in unit_result.values() if v["verdict_source"]=="human")}'
             f' | model distribution: {dict(Counter(v["model"] for v in unit_result.values()))}', '']

    b_ids = sorted([uid for uid in ann if units[uid]['retrieved']])
    a_ids = sorted([uid for uid in ann if not units[uid]['retrieved']])
    b_h1 = [u for u in b_ids if unit_result[u]['level'] == 'H1']
    b_h2 = [u for u in b_ids if unit_result[u]['level'] == 'H2']

    lines.append('--- Component B: retrieved summaries (exposure source) ---')
    for key, label in (('primary', 'primary criterion (fact-level)'), ('strict', 'strict criterion (all claims)')):
        lines.append(f'[{label}]')
        r = rate_block(b_ids, key)
        lines.append(f'  Overall: n={r["n"]} unfaithful={r["n_unfaithful"]} rate={r["rate"]:.4f} '
                     f'95%CI[{r["ci_lo"]:.4f},{r["ci_hi"]:.4f}]')
        for name, ids in (('H1', b_h1), ('H2', b_h2)):
            rr = rate_block(ids, key)
            lines.append(f'  {name}: n={rr["n"]} unfaithful={rr["n_unfaithful"]} rate={rr["rate"]:.4f} '
                         f'95%CI[{rr["ci_lo"]:.4f},{rr["ci_hi"]:.4f}]')
        for pth in ('title', 'notitle'):
            ids = [u for u in b_ids if unit_result[u]['path'] == pth]
            rr = rate_block(ids, key)
            lines.append(f'  {pth}: n={rr["n"]} unfaithful={rr["n_unfaithful"]} rate={rr["rate"]:.4f} '
                         f'95%CI[{rr["ci_lo"]:.4f},{rr["ci_hi"]:.4f}]')
    tot_claims = sum(unit_result[u]['n_claims'] for u in b_ids)
    fact_bad = sum(unit_result[u]['n_fact_bad'] for u in b_ids)
    any_bad = sum(unit_result[u]['n_any_bad'] for u in b_ids)
    ent_bad = sum(unit_result[u]['n_bad_entity'] for u in b_ids)
    lines.append(f'  claim level: fact-bad {fact_bad} / any-bad {any_bad} / total claims {tot_claims}'
                 f' (claim-level fact bad rate {fact_bad/max(tot_claims,1):.4f}）')
    lines.append(f'  entity-level missing/mismatch: {ent_bad}')
    # ---- Claim-count monitoring (guards against laziness / the 40-claim cap diluting results) ----
    import statistics as _st
    for name, ids in (('H1', b_h1), ('H2', b_h2)):
        cs = [unit_result[u]['n_claims'] for u in ids]
        if not cs:
            continue
        cap = sum(1 for x in cs if x >= 40)
        flag = [unit_result[u]['n_claims'] for u in ids if unit_result[u]['primary']]
        noflag = [unit_result[u]['n_claims'] for u in ids if not unit_result[u]['primary']]
        lines.append(f'  [{name}] claims/unit: mean {_st.mean(cs):.1f} median {_st.median(cs):.0f} '
                     f'| capped (>=40) {cap} units | positive-unit mean {_st.mean(flag):.1f} vs negative {_st.mean(noflag):.1f}'
                     if flag and noflag else
                     f'  [{name}] claims/unit: mean {_st.mean(cs):.1f} median {_st.median(cs):.0f} | capped (>=40) {cap} units')

    if a_ids:
        lines.append('')
        lines.append('--- Component A: non-retrieved-layer sample (overall rate) ---')
        for key, label in (('primary', 'primary criterion'), ('strict', 'strict criterion')):
            r = rate_block(a_ids, key)
            lines.append(f'[{label}] n={r["n"]} unfaithful={r["n_unfaithful"]} rate={r["rate"]:.4f} '
                         f'95%CI[{r["ci_lo"]:.4f},{r["ci_hi"]:.4f}]')
            for name in ('H1', 'H2'):
                ids = [u for u in a_ids if unit_result[u]['level'] == name]
                if ids:
                    rr = rate_block(ids, key)
                    lines.append(f'  {name}: n={rr["n"]} unfaithful={rr["n_unfaithful"]} rate={rr["rate"]:.4f} '
                                 f'95%CI[{rr["ci_lo"]:.4f},{rr["ci_hi"]:.4f}]')

    # ---------- Exposure (dual-criterion) ----------
    flags = {}
    for qid, q in question_map.items():
        bad_primary, bad_strict = [], []
        for s in q['slots']:
            u = s['unit_id']
            if u in unit_result:
                if unit_result[u]['primary']:
                    bad_primary.append({'rank': s['rank'], 'unit_id': u, 'level': unit_result[u]['level']})
                if unit_result[u]['strict']:
                    bad_strict.append({'rank': s['rank'], 'unit_id': u, 'level': unit_result[u]['level']})
        flags[qid] = {'book': q['book'], 'primary_contaminated': bool(bad_primary),
                      'strict_contaminated': bool(bad_strict),
                      'bad_units_primary': bad_primary, 'bad_units_strict': bad_strict}

    n_total = len(question_map)
    summary_exp = {}
    for key, ckey in (('primary', 'primary_contaminated'), ('strict', 'strict_contaminated')):
        cont = [q for q, f in flags.items() if f[ckey]]
        n_h1 = sum(1 for q in cont if any(b['level'] == 'H1' for b in flags[q]['bad_units_' + key]))
        n_h2 = sum(1 for q in cont if any(b['level'] == 'H2' for b in flags[q]['bad_units_' + key]))
        n_top1 = sum(1 for q in cont if any(b['rank'] == 1 for b in flags[q]['bad_units_' + key]))
        summary_exp[key] = {'n_total': n_total, 'n_contaminated': len(cont),
                            'n_h1': n_h1, 'n_h2': n_h2, 'n_top1': n_top1,
                            'clean': n_total - len(cont)}
        lines.append('')
        lines.append(f'--- Bounded exposure ({key} criterion) ---')
        lines.append(f'Questions containing >=1 unfaithful summary: {len(cont)}/{n_total}'
                     f' (H1 causes {n_h1}, H2 causes {n_h2}, top-1 hit {n_top1}); fully clean {n_total-len(cont)} questions')

    # ---------- Score comparison and conservative bounds (primary criterion) ----------
    score_lines, bound_lines = [], []
    for d in DIMS:
        cont_scores = [judge_scores[q][d] for q, f in flags.items() if f['primary_contaminated']
                       and q in judge_scores and isinstance(judge_scores[q][d], (int, float)) and judge_scores[q][d] > 0]
        clean_scores = [judge_scores[q][d] for q, f in flags.items() if not f['primary_contaminated']
                        and q in judge_scores and isinstance(judge_scores[q][d], (int, float)) and judge_scores[q][d] > 0]
        mc = sum(cont_scores) / len(cont_scores) if cont_scores else float('nan')
        mcl = sum(clean_scores) / len(clean_scores) if clean_scores else float('nan')
        score_lines.append(f'  {d}: polluted n={len(cont_scores)} mean={mc:.3f} | clean n={len(clean_scores)} mean={mcl:.3f} | Δ={mc-mcl:+.3f}')
        drop = sum(v - 1 for v in cont_scores) / n_total if cont_scores else 0.0
        bound_lines.append(f'  {d}: if all polluted items were truly scored 1, the mean would drop by at most {drop:.4f}')
    lines.append('')
    lines.append('Score comparison (primary criterion, descriptive):')
    lines.extend(score_lines)
    lines.append('Conservative bounds (primary criterion):')
    lines.extend(bound_lines)

    EXPOSURE_DIR.mkdir(parents=True, exist_ok=True)
    (EXPOSURE_DIR / 'question_flags.json').write_text(
        json.dumps(flags, ensure_ascii=False, indent=1), encoding='utf-8')
    summary = {
        'prompt_version': PROMPT_VERSION,
        'n_annotated': len(ann),
        'n_human': sum(1 for v in unit_result.values() if v['verdict_source'] == 'human'),
        'component_b_primary': {'overall': rate_block(b_ids, 'primary'),
                                'h1': rate_block(b_h1, 'primary'), 'h2': rate_block(b_h2, 'primary')},
        'component_b_strict': {'overall': rate_block(b_ids, 'strict'),
                               'h1': rate_block(b_h1, 'strict'), 'h2': rate_block(b_h2, 'strict')},
        'claims': {'total': tot_claims, 'fact_bad': fact_bad, 'any_bad': any_bad, 'entity_bad': ent_bad},
        'component_a_primary': rate_block(a_ids, 'primary') if a_ids else None,
        'component_a_strict': rate_block(a_ids, 'strict') if a_ids else None,
        'exposure': summary_exp,
        'score_comparison_primary': score_lines,
        'conservative_bound_primary': bound_lines,
    }
    (REPORTS_DIR / 'audit_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    report = '\n'.join(lines)
    (REPORTS_DIR / 'audit_summary.txt').write_text(report, encoding='utf-8')
    print(report)
    if len(b_ids) != 1078:
        print(f'[warn] Component B annotations incomplete: {len(b_ids)}/1078')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
