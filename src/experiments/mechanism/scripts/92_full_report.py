# -*- coding: utf-8 -*-
"""Generates the full report: 05-reports/full_report.txt (200 questions x 4 arms: means, paired tests, 2x2 factor decomposition, criteria, cost)"""
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


def paired(db, x, y, B=10000):
    """db[book] = [(x, y), ...] -> (obs, lo, hi, p) on x-y"""
    by_book = defaultdict(list)
    for b, pairs in db.items():
        for xv, yv in pairs:
            by_book[b].append(xv - yv)
    return boot(by_book, B)


four = {a: [r for r in load(f'04-judgments/by_book/*.four{a}.3dim.json') if not r.get('error')] for a in 'ABCD'}
four_all = {a: load(f'04-judgments/by_book/*.four{a}.3dim.json') for a in 'ABCD'}
newA = {j['query_id']: j for j in four['A']}
by_qid = {a: {j['query_id']: j for j in four[a]} for a in 'ABCD'}

lines = []
lines.append('===== Four-arm full report (200 questions x 4 arms) =====')
lines.append('Design: retrieval set fixed, only the context fed to the generator varies. A=H1 slice+H2 prefix; B=H1 slice only; C=H2 prefix only (H1 slice backfilled by body blocks); D=both removed')
lines.append('Pre-registered criteria (grounding): D drops vs. A by >=0.05 and B>C -> supported; 0.02-0.05 weakly supported; <0.02 rejected')
lines.append('')

# 1) Data quality
lines.append('## 1. Data quality and exclusions')
for a in 'ABCD':
    errs = [r for r in four_all[a] if r.get('error')]
    kinds = {}
    for r in errs:
        kinds[r['error'][:20]] = kinds.get(r['error'][:20], 0) + 1
    lines.append(f'  Arm {a}: coverage {len(four_all[a])}/200 | valid {len(four[a])} | excluded: {kinds if kinds else "none"}')
lines.append('  Note: a few B/D items were DashScope content-moderation false positives (blank return -> empty_answer; permanently excluded); the judging side validates empty reasons and out-of-range scores')
lines.append('')

# 2) Four-arm means
lines.append('## 2. Four-arm means (valid samples, full)')
lines.append(f"{'arm':>3} | {'grounding':>9} | {'correctness':>11} | {'halluc':>7} | {'comp100':>7} | n")
for a in 'ABCD':
    recs = four[a]
    m = {d: st.mean([r[d] for r in recs]) for d in DIMS}
    cp = [comp100(r) for r in recs if comp100(r) is not None]
    lines.append(f"{a:>3} | {m['grounding']:>9.2f} | {m['correctness']:>11.2f} | {m['hallucination_absence']:>7.2f} | {st.mean(cp):>7.2f} | {len(recs)}")
wm = {d: sum(j[d] for j in four['A'] if j.get(d)) / len([1 for j in four['A'] if j.get(d)]) for d in DIMS}
lines.append('')

# 3) Paired comparisons (vs A, full)
lines.append('## 3. Paired comparisons (vs A, full; book-clustered bootstrap B=10000)')
for a in 'BCD':
    for d in DIMS + ['comp100']:
        db = defaultdict(list)
        for j in four[a]:
            bj = newA.get(j['query_id'])
            if not bj:
                continue
            x = comp100(j) if d == 'comp100' else j.get(d)
            y = comp100(bj) if d == 'comp100' else bj.get(d)
            if x is None or y is None:
                continue
            db[j['book_title']].append((float(x), float(y)))
        obs, lo, hi, p = paired(db, None, None)
        flag = '***' if p < 0.01 else ('*' if p < 0.05 else '')
        lines.append(f'  {a} - A [{d:>20}] Δ={obs:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.4f} {flag}')
lines.append('')

# 4) 2x2 factor decomposition (items with all four arms)
lines.append('## 4. Factor decomposition (items with all four arms; H1=(A+B)/2-(C+D)/2; H2 prefix=(A+C)/2-(B+D)/2; interaction=A-B-C+D)')


def effect(d, kind):
    db = defaultdict(list)
    for q in by_qid['A']:
        if not all(q in by_qid[a] for a in 'ABCD'):
            continue
        v = {}
        ok = True
        for a in 'ABCD':
            x = comp100(by_qid[a][q]) if d == 'comp100' else by_qid[a][q].get(d)
            if x is None:
                ok = False
                break
            v[a] = float(x)
        if not ok:
            continue
        if kind == 'H1':
            e = (v['A'] + v['B']) / 2 - (v['C'] + v['D']) / 2
        elif kind == 'H2':
            e = (v['A'] + v['C']) / 2 - (v['B'] + v['D']) / 2
        else:
            e = v['A'] - v['B'] - v['C'] + v['D']
        db[by_qid['A'][q]['book_title']].append((e, 0.0))
    obs, lo, hi, p = paired(db, None, None)
    return obs, lo, hi, p


for d in ['grounding', 'comp100']:
    for kind in ['H1', 'H2', 'interaction']:
        obs, lo, hi, p = effect(d, kind)
        flag = '***' if p < 0.01 else ('*' if p < 0.05 else '')
        lines.append(f'  {kind} [{d:>10}] Δ={obs:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.4f} {flag}')
lines.append('')

# 5) Pre-registered criteria
lines.append('## 5. Pre-registered criteria (grounding)')
d_a = None
b_c = None
for a, tag in [('D', 'd_a'), ('B', 'b_c')]:
    db = defaultdict(list)
    for j in four[a]:
        ref = newA.get(j['query_id']) if a == 'D' else by_qid['C'].get(j['query_id'])
        if not ref:
            continue
        x, y = j.get('grounding'), ref.get('grounding')
        if x is None or y is None:
            continue
        db[j['book_title']].append((float(x), float(y)))
    obs, lo, hi, p = paired(db, None, None)
    if tag == 'd_a':
        d_a = (obs, lo, hi, p)
    else:
        b_c = (obs, lo, hi, p)
lines.append(f'  D - A: Δ={d_a[0]:+.3f} CI[{d_a[1]:+.3f},{d_a[2]:+.3f}] p={d_a[3]:.4f}')
lines.append(f'  B - C: Δ={b_c[0]:+.3f} CI[{b_c[1]:+.3f},{b_c[2]:+.3f}] p={b_c[3]:.4f}')
if d_a[0] <= -0.05 and b_c[0] > 0:
    lines.append('  Verdict: mechanism supported (D drops significantly/substantially and B>C)')
elif d_a[0] <= -0.02 and b_c[0] > 0:
    lines.append('  Verdict: weakly supported (D drops 0.02-0.05)')
else:
    lines.append('  Verdict: rejected/unsupported ("removing both summary levels hurts generation quality" is not supported by the data)')
lines.append('')

# 6) Cost
gin = gout = jin = jout = 0.0
for a in 'ABCD':
    for p in glob.glob(os.path.join(BASE, f'03-answers/by_book/*.four{a}.top5.json')):
        for r in json.load(open(p, encoding='utf-8')):
            u = r.get('gen_usage') or {}
            gin += u.get('prompt_tokens', 0)
            gout += u.get('completion_tokens', 0)
    for p in glob.glob(os.path.join(BASE, f'04-judgments/by_book/*.four{a}.3dim.json')):
        for r in json.load(open(p, encoding='utf-8')):
            u = r.get('judge_usage') or {}
            jin += u.get('prompt_tokens', 0)
            jout += u.get('completion_tokens', 0)
cost = gin / 1e6 * RATE_GEN[0] + gout / 1e6 * RATE_GEN[1] + jin / 1e6 * RATE_JD[0] + jout / 1e6 * RATE_JD[1]
lines.append('## 6. Cost (all calls, including historical retries)')
lines.append(f'  Generation: input {gin:,.0f} / output {gout:,.0f} tokens')
lines.append(f'  Judging: input {jin:,.0f} / output {jout:,.0f} tokens')
lines.append(f'  Estimated cost ~ CNY {cost:.2f} (generation 2/12 per M; judging at pro off-peak 4.5/13.5)')
lines.append('')

# 7) Reading
lines.append('## 7. Conclusions and recommendations')
lines.append('  1) Section 5 pre-registered criteria govern; supported or not, the mechanism experiment only serves as evidence for the generation-side role of summaries and does not change the main-results framework;')
lines.append('  2) If rejected: position the paper as route A (context-preserving + significant gains over four baselines + smallest budget) and report faithfully that generation-side gains are limited;')
lines.append('  3) If supported: add the generation-side benefit of summaries to the discussion, still stated separately from retrieval-side benefits.')

report = '\n'.join(lines) + '\n'
out = os.path.join(BASE, '05-reports', 'full_report.txt')
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, 'w', encoding='utf-8').write(report)
print(report)
print('Wrote:', out)
