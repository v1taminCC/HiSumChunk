# -*- coding: utf-8 -*-
"""Progress and cost monitor: counts four-arm answers/judgments, errors, token usage, and estimated cost

Usage:
  python scripts/90_status.py
"""
import json
import glob
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_RATE = (2.0, 12.0)      # qwen CNY/M: input/output
JD_RATE = (4.5, 13.5)       # deepseek pro off-peak: input/output


def scan(ans_dir: str, jd_dir: str):
    rows = []
    for arm in 'ABCD':
        ans_files = sorted(glob.glob(os.path.join(BASE, ans_dir, 'by_book', f'*.four{arm}.top5.json')))
        jd_files = sorted(glob.glob(os.path.join(BASE, jd_dir, 'by_book', f'*.four{arm}.3dim.json')))
        n_ans = err_ans = n_jd = err_jd = 0
        gin = gout = jin = jout = 0.0
        for p in ans_files:
            for r in json.load(open(p, encoding='utf-8')):
                n_ans += 1
                if r.get('error'):
                    err_ans += 1
                u = r.get('gen_usage') or {}
                gin += u.get('prompt_tokens', 0)
                gout += u.get('completion_tokens', 0)
        for p in jd_files:
            for r in json.load(open(p, encoding='utf-8')):
                n_jd += 1
                if r.get('error'):
                    err_jd += 1
                u = r.get('judge_usage') or {}
                jin += u.get('prompt_tokens', 0)
                jout += u.get('completion_tokens', 0)
        cost = gin / 1e6 * GEN_RATE[0] + gout / 1e6 * GEN_RATE[1] + jin / 1e6 * JD_RATE[0] + jout / 1e6 * JD_RATE[1]
        rows.append((arm, n_ans, err_ans, n_jd, err_jd, gin, gout, jin, jout, cost))
    return rows


def show(title: str, rows) -> float:
    print(f'== {title} ==')
    print(f"{'arm':>3} | {'ans':>4} {'err':>3} | {'judge':>4} {'err':>3} | {'gen_in':>9} {'gen_out':>8} | {'jd_in':>9} {'jd_out':>7} | {'cost￥':>7}")
    for r in rows:
        print(f'{r[0]:>3} | {r[1]:>4} {r[2]:>3} | {r[3]:>4} {r[4]:>3} | {r[5]:>9,.0f} {r[6]:>8,.0f} | {r[7]:>9,.0f} {r[8]:>7,.0f} | {r[9]:>7.2f}')
    tot = sum(r[9] for r in rows)
    print(f'  Subtotal ~ CNY {tot:.2f}')
    return tot


t_main = show('Main outputs (sample 200)', scan('03-answers', '04-judgments'))
t_ext = show('Extended outputs (remaining 359, if started)', scan('03-answers-remaining', '04-judgments-remaining'))
t_fill = show('18 fill questions (fill18, if started)', scan('03-answers-fill18', '04-judgments-fill18'))
print(f'-- Total estimated cost ~ CNY {t_main + t_ext + t_fill:.2f} (generation at qwen 2/12; judging at pro off-peak 4.5/13.5)')
