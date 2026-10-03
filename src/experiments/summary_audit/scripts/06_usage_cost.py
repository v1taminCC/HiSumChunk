#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summary faithfulness audit - usage and cost report.

Supports two record kinds:
  - platform=deepseek (formal run; official DeepSeek pricing, USD, Beijing peak/off-peak)
  - platform=tencent/bailian (historical experiments; Tencent Guangzhou pricing, Bailian free quota)
Usage: python scripts/06_usage_cost.py [--annot results/annotations/full_v3.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FULL_RUN_ANN, REPORTS_DIR  # noqa: E402

USD_CNY = 7.1

# DeepSeek official (USD / 1M tokens): {off:(miss,hit,out), peak:(miss,hit,out)}
DS_PRICES = {
    'deepseek-v4-pro':     {'off': (0.66, 0.022, 1.98), 'peak': (1.32, 0.044, 3.96)},
}
# Tencent TokenHub Guangzhou (CNY / million tokens): {off:(in,out), peak:(in,out)}
TENCENT_PRICES = {
    'deepseek-v4-pro':     {'off': (4.5, 13.5), 'peak': (9.0, 27.0)},
}


def is_peak(ts: str) -> bool:
    try:
        t = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    except Exception:  # noqa: BLE001
        return False
    return (9 <= t.hour < 12) or (14 <= t.hour < 18)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--annot', default=str(FULL_RUN_ANN))
    args = ap.parse_args()
    path = Path(args.annot)
    if not path.exists():
        print(f'[skip] file not found: {path}')
        return 0

    agg = defaultdict(lambda: {'in': 0, 'out': 0, 'hit': 0, 'n_ok': 0, 'n_bad': 0, 'n_peak': 0,
                               'usd': 0.0, 'usd_peak': 0.0, 'usd_off': 0.0,
                               'cny': 0.0, 'cny_peak': 0.0, 'cny_off': 0.0, 'quota': False})
    for line in open(path, encoding='utf-8'):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        plat = r.get('platform', '')
        model = r.get('model', '')
        key = f'{plat}/{model}' if plat else model
        a = agg[key]
        if not r.get('ok'):
            a['n_bad'] += 1
            continue
        a['n_ok'] += 1
        u = r.get('usage') or {}
        tin = int(u.get('prompt_tokens', 0) or 0)
        tout = int(u.get('completion_tokens', 0) or 0)
        hit = int(u.get('prompt_cache_hit_tokens', 0) or 0) or int(
            (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0)
        miss = max(tin - hit, 0)
        a['in'] += tin
        a['out'] += tout
        a['hit'] += hit
        peak = is_peak(r.get('ts', ''))
        a['n_peak'] += int(peak)

        if plat == 'deepseek' and model in DS_PRICES:
            (mo, ho, oo), (mp, hp, op) = DS_PRICES[model]['off'], DS_PRICES[model]['peak']
            def usd(miss_p, hit_p, out_p):
                return miss / 1e6 * miss_p + hit / 1e6 * hit_p + tout / 1e6 * out_p
            a['usd'] += usd(mp if peak else mo, hp if peak else ho, op if peak else oo) * USD_CNY
            a['usd_peak'] += usd(mp, hp, op) * USD_CNY
            a['usd_off'] += usd(mo, ho, oo) * USD_CNY
        elif plat == 'tencent' and model in TENCENT_PRICES:
            (mo, oo), (mp, op) = TENCENT_PRICES[model]['off'], TENCENT_PRICES[model]['peak']
            def cny(miss_p, out_p):
                return miss / 1e6 * miss_p + tout / 1e6 * out_p
            a['cny'] += cny(mp if peak else mo, op if peak else oo)
            a['cny_peak'] += cny(mp, op)
            a['cny_off'] += cny(mo, oo)
        else:
            a['quota'] = True

    lines = ['=== Summary faithfulness audit - usage and cost ===', f'file: {path}', '']
    totals = {'cny': 0.0, 'cny_peak': 0.0, 'cny_off': 0.0}
    for k, a in sorted(agg.items()):
        lines.append(f'[{k}] ok={a["n_ok"]} bad={a["n_bad"]} | in={a["in"]:,} out={a["out"]:,} '
                     f'cache={a["hit"]:,} | peak requests {a["n_peak"]}/{a["n_ok"]}')
        if a['n_ok'] and (a['usd'] or a['cny']):
            if a['usd']:
                lines.append(f'  Cost (CNY, FX {USD_CNY}): actual window ~ CNY {a["usd"]:.2f} (all-peak CNY {a["usd_peak"]:.2f} / all-off-peak CNY {a["usd_off"]:.2f}）')
                for kk in totals:
                    totals[kk] += {'cny': a['usd'], 'cny_peak': a['usd_peak'], 'cny_off': a['usd_off']}[kk]
            else:
                lines.append(f'  Cost (CNY): actual window ~ CNY {a["cny"]:.2f} (all-peak CNY {a["cny_peak"]:.2f} / all-off-peak CNY {a["cny_off"]:.2f}）')
                for kk in totals:
                    totals[kk] += a[kk]
        else:
            lines.append('  Cost: free quota / not billed')
    lines.append('')
    lines.append(f'Billed total: actual window ~ CNY {totals["cny"]:.2f} | all-peak CNY {totals["cny_peak"]:.2f} | all-off-peak CNY {totals["cny_off"]:.2f}')
    report = '\n'.join(lines)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / 'usage_report.txt'
    out.write_text(report, encoding='utf-8')
    print(report)
    print(f'[out] {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
