#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summary faithfulness audit - full formal run: v3 protocol with the DeepSeek judge model.

- Default frame: data/frames/full_all.json (B 1,078 + A 400 = 1,478 records)
- Output: data/annotations/{tag}.jsonl (default full_v3; checkpoint resume)
- thinking explicitly disabled; on moderation/JSON truncation, one fallback call
Usage:
  python scripts/02_annotate.py            # dry-run
  python scripts/02_annotate.py --yes      # formal run
  python scripts/02_annotate.py --frame results/frames/trial50_ds.json --tag trial50_ds --yes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    ANNOT_DIR, API_BASE, API_KEY, AUDIT_MAX_TOKENS, DEFAULT_WORKERS, FRAMES_DIR,
    INVENTORY_DIR, LOGS_DIR, MODEL_AUDIT, MODEL_FALLBACK, PROMPTS_DIR, PROMPT_VERSION,
)
from utils_api import chat, chat_batch  # noqa: E402

LEVEL_DESC = {
    'H1': 'H1 region-level summary (a generated summary of the concatenated text blocks given below)',
    'H2': 'H2 local summary (a generated summary of the single text block given below)',
}


def load_units() -> dict[str, dict]:
    units = {}
    with open(INVENTORY_DIR / 'units_all.jsonl', encoding='utf-8') as f:
        for line in f:
            u = json.loads(line)
            units[u['unit_id']] = u
    return units


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', default=str(FRAMES_DIR / 'full_all.json'))
    ap.add_argument('--tag', default='full_v3')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    ap.add_argument('--yes', action='store_true')
    args = ap.parse_args()

    frame = json.loads(Path(args.frame).read_text(encoding='utf-8'))
    units = load_units()
    out_path = ANNOT_DIR / f'{args.tag}.jsonl'
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for line in open(out_path, encoding='utf-8'):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get('ok') and r.get('prompt_version') == PROMPT_VERSION:
                done.add(r['unit_id'])
    todo = [x['unit_id'] for x in frame if x['unit_id'] not in done]
    if args.limit:
        todo = todo[:args.limit]
    est = sum(int(1.201 * (units[u]['summary_wp'] + units[u]['source_wp']) + 823 +
                  0.083 * (units[u]['summary_wp'] + units[u]['source_wp']) + 809) for u in todo)
    print(f'[plan] protocol {PROMPT_VERSION} | frame {Path(args.frame).name} | to run {len(todo)}/{len(frame)} | '
          f'model {MODEL_AUDIT} | estimated {est:,} tokens')
    if not args.yes:
        print('[dry-run] --yes not given; API not called.')
        return 0
    if not todo:
        print('[done] nothing to run.')
        return 0

    pre = chat('You are a ping service.', 'Reply with exactly: OK', model=MODEL_AUDIT,
               base_url=API_BASE, api_key=API_KEY, max_tokens=8, temperature=0, timeout=60, retry=1)
    if not pre.get('ok'):
        print('[preflight] API unavailable; aborting.', pre.get('error', '')[:500])
        return 1
    print('[preflight] API OK')

    system_prompt = (PROMPTS_DIR / f'audit_system.{PROMPT_VERSION}.txt').read_text(encoding='utf-8').strip()
    user_template = (PROMPTS_DIR / f'audit_user.{PROMPT_VERSION}.txt').read_text(encoding='utf-8')

    fout = open(out_path, 'a', encoding='utf-8')
    stats = {'ok': 0, 'bad': 0, 'in': 0, 'out': 0, 'cache': 0}
    status: dict[str, dict] = {}

    def build_calls(ids: list[str], model: str) -> list[dict]:
        calls = []
        for uid in ids:
            u = units[uid]
            calls.append({
                'system': system_prompt,
                'user': user_template.format(level_desc=LEVEL_DESC[u['level']],
                                             summary=u['summary'], source=u['source']),
                'model': model, 'base_url': API_BASE, 'api_key': API_KEY,
                'meta': {'unit_id': uid, 'level': u['level'], 'book': u['book'], 'path': u['path'],
                         'summary_wp': u['summary_wp'], 'source_wp': u['source_wp']},
            })
        return calls

    def make_on_result(call_list, model_name):
        def on_result(idx, result):
            meta = call_list[idx]['meta']
            rec = {**meta, 'prompt_version': PROMPT_VERSION, 'platform': 'deepseek',
                   'model_key': model_name, 'model': model_name,
                   'ok': bool(result.get('ok')), 'error': result.get('error', ''),
                   'fatal': bool(result.get('fatal')), 'raw': (result.get('content') or '')[:2000],
                   'usage': result.get('usage', {}), 'parsed': result.get('parsed', {}),
                   'ts': time.strftime('%Y-%m-%d %H:%M:%S')}
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fout.flush()
            status[meta['unit_id']] = rec
            stats['ok' if rec['ok'] else 'bad'] += 1
            u = rec.get('usage') or {}
            stats['in'] += int(u.get('prompt_tokens', 0) or 0)
            stats['out'] += int(u.get('completion_tokens', 0) or 0)
            stats['cache'] += int(u.get('prompt_cache_hit_tokens', 0) or 0)
            n = stats['ok'] + stats['bad']
            if n % 50 == 0:
                print(f"  [{n}/{len(todo)}] ok={stats['ok']} bad={stats['bad']} in={stats['in']:,} out={stats['out']:,}")
        return on_result

    t0 = time.time()
    calls = build_calls(todo, MODEL_AUDIT)
    chat_batch(calls, model=MODEL_AUDIT, temperature=0.1, max_tokens=AUDIT_MAX_TOKENS,
               workers=args.workers, on_result=make_on_result(calls, MODEL_AUDIT))

    # Fallback: moderation/JSON truncation -> deepseek-v4-pro
    failed = []
    for uid in todo:
        r = status.get(uid)
        if not r or r['ok']:
            continue
        err = (r.get('error') or '').lower()
        if ('inappropriate content' in err) or ('json parse failed' in err):
            failed.append(uid)
    if failed:
        print(f'[fallback] {len(failed)} records switched to {MODEL_FALLBACK}')
        fb_calls = build_calls(failed, MODEL_FALLBACK)
        chat_batch(fb_calls, model=MODEL_FALLBACK, temperature=0.1, max_tokens=AUDIT_MAX_TOKENS,
                   workers=args.workers, on_result=make_on_result(fb_calls, MODEL_FALLBACK))
    fout.close()
    dur = time.time() - t0
    print(f"[done] ok={stats['ok']} bad={stats['bad']} | in={stats['in']:,} out={stats['out']:,} "
          f"cache_hit={stats['cache']:,} | {dur/60:.1f} min")
    print(f'[out] {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
