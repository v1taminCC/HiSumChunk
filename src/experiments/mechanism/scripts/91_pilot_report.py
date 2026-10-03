# -*- coding: utf-8 -*-
"""Generates the pilot report: 05-reports/pilot_report.txt (data quality, four-arm comparison, cost)"""
import json
import glob
import os
import random
import statistics as st
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIMS = ['grounding', 'correctness', 'hallucination_absence']
RATE_GEN = (2.0, 12.0)
RATE_JD = (4.5, 13.5)


def load(pat):
    rows = []
    for p in glob.glob(os.path.join(BASE, pat)):
        rows += json.load(open(p, encoding='utf-8'))
    return rows


def comp100(j):
    vals = [j.get(d) for d in DIMS]
    if None in vals or any(v == 0 for v in vals):
        return None
    return sum(v - 1 for v in vals) / 12 * 100


def boot(deltas_by_book, B=10000, seed=2026):
    rng = random.Random(seed)
    books = list(deltas_by_book)
    n = sum(len(v) for v in deltas_by_book.values())
    if n == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    obs = sum(sum(v) for v in deltas_by_book.values()) / n
    dist = []
    for _ in range(B):
        s = nn = 0
        for _b in range(len(books)):
            b = rng.choice(books)
            s += sum(deltas_by_book[b])
            nn += len(deltas_by_book[b])
        dist.append(s / nn)
    dist.sort()
    lo, hi = dist[int(B * .025)], dist[int(B * .975)]
    p = (min(sum(1 for x in dist if x >= 0), sum(1 for x in dist if x <= 0)) + 1) / (B + 1)
    return obs, lo, hi, p


four = {a: [r for r in load(f'04-judgments/by_book/*.four{a}.3dim.json') if not r.get('error')] for a in 'ABCD'}
four_all = {a: load(f'04-judgments/by_book/*.four{a}.3dim.json') for a in 'ABCD'}
newA = {j['query_id']: j for j in four['A']}

lines = []
lines.append('===== Four-arm pilot report =====')
lines.append('Scope: first 3 sampled questions per book (20 books; 56 questions in practice) x 4 arms = 224 generations + 224 judgments')
lines.append('Criteria (pre-registered, for the full 200): D drop>=0.05 and B>C -> supported; 0.02-0.05 weak; <0.02 rejected')
lines.append('')

# 1) Data quality
lines.append('## 1. Data quality and exclusions')
for a in 'ABCD':
    errs = [r for r in four_all[a] if r.get('error')]
    kinds = {}
    for r in errs:
        kinds[r['error'][:20]] = kinds.get(r['error'][:20], 0) + 1
    lines.append(f'  Arm {a}: valid {len(four[a])}/{len(four_all[a])} | excluded: {kinds if kinds else "none"}')
lines.append('  Note: 2 B and 2 D items were false positives of DashScope content moderation (classical drama text; generation failed -> empty_answer; permanently excluded)')
lines.append('  Fix log: an overly low max_tokens on the judging side truncated JSON in reasoning mode (parse_failed with empty error) -> raised to 16384 and error flags fixed; all re-judged successfully')
lines.append('')

# 2) Four-arm means
lines.append('## 2. Four-arm means (valid samples)')
lines.append(f"{'arm':>3} | {'grounding':>9} | {'correctness':>11} | {'halluc':>7} | {'comp100':>7} | n")
for a in 'ABCD':
    recs = four[a]
    m = {d: st.mean([r[d] for r in recs]) for d in DIMS}
    cp = [comp100(r) for r in recs if comp100(r) is not None]
    lines.append(f"{a:>3} | {m['grounding']:>9.2f} | {m['correctness']:>11.2f} | {m['hallucination_absence']:>7.2f} | {st.mean(cp):>7.2f} | {len(recs)}")
lines.append('')

# 3) Paired comparisons (vs A)
lines.append('## 3. Paired comparisons (vs A, book-clustered bootstrap B=10000; pilot n~56, underpowered)')
for a in 'BCD':
    for d in DIMS + ['comp100']:
        db = defaultdict(list)
        for j in four[a]:
            qid = j['query_id']
            bj = newA.get(qid)
            if not bj:
                continue
            x = comp100(j) if d == 'comp100' else j.get(d)
            y = comp100(bj) if d == 'comp100' else bj.get(d)
            if x is None or y is None:
                continue
            db[j['book_title']].append(float(x) - float(y))
        obs, lo, hi, p = boot(db)
        flag = '***' if p < 0.01 else ('*' if p < 0.05 else '')
        lines.append(f'  {a} vs A [{d:>20}] Δ={obs:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} {flag}')
lines.append('')

# 4) Cost
gin = gout = jin = jout = 0.0
for a in 'ABCD':
    for p in glob.glob(os.path.join(BASE, f'03-answers/by_book/*.four{a}.top5.json')):
        for r in json.load(open(p, encoding='utf-8')):
            u = r.get('gen_usage') or {}
            gin += u.get('prompt_tokens', 0); gout += u.get('completion_tokens', 0)
    for p in glob.glob(os.path.join(BASE, f'04-judgments/by_book/*.four{a}.3dim.json')):
        for r in json.load(open(p, encoding='utf-8')):
            u = r.get('judge_usage') or {}
            jin += u.get('prompt_tokens', 0); jout += u.get('completion_tokens', 0)
cost = gin / 1e6 * RATE_GEN[0] + gout / 1e6 * RATE_GEN[1] + jin / 1e6 * RATE_JD[0] + jout / 1e6 * RATE_JD[1]
lines.append('## 4. Pilot cost (including all re-judgments)')
lines.append(f'  Generation: input {gin:,.0f} / output {gout:,.0f} tokens')
lines.append(f'  Judging: input {jin:,.0f} / output {jout:,.0f} tokens (incl. re-judgments after the truncation bug)')
lines.append(f'  Estimated cost ~ CNY {cost:.2f} (generation 2/12 per M; judging at pro off-peak 4.5/13.5)')
lines.append('')

# 5) Preliminary reading
lines.append('## 5. Preliminary reading (for decisions; not a final conclusion)')
lines.append('  1) Pilot n=56 (54-56 valid per arm), far below the full 200; underpowered for differences below +/-0.1;')
lines.append('  2) Point-estimate direction: B~+0.06, D~+0.11, C~-0.07 (grounding; none significant) — "removing summaries hurts" was not observed;')
lines.append('  3) Recommendation: run the full 200 (800+800, about CNY 76 at off-peak prices) and decide by the pre-registered criteria;')
lines.append('     If the full run still shows a zero/positive effect, pivot the mechanism story to budget-efficiency + evaluation-analysis positioning.')
lines.append('')

report = '\n'.join(lines) + '\n'
out = os.path.join(BASE, '05-reports', 'pilot_report.txt')
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, 'w', encoding='utf-8').write(report)
print(report)
print('Wrote:', out)
