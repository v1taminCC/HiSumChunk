# -*- coding: utf-8 -*-
"""Four-arm answer judging (3 dimensions; DeepSeek official; incremental persistence, checkpoint every 20 items, retry on failure, 400-truncation retry)

Usage:
  python scripts/03_judge_answers.py --arm A --limit 20 --yes
  python scripts/03_judge_answers.py --arm B --ext --yes      # remaining questions (separate directory)
  python scripts/03_judge_answers.py --arm B --fill18 --yes   # 18 fill questions (separate directory)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    ANSWERS_EXT_DIR, ANSWERS_FILL18_DIR, ANSWERS_OUT_DIR, DIMS, JUDGE_TEMP, JUDG_EXT_DIR,
    JUDG_FILL18_DIR, JUDG_OUT_DIR, MODEL_JUDGE, PROMPTS_DIR, QA_DRAFTS, WORKERS,
)
from utils_simple_api import API_BASE_DEEPSEEK, API_KEY_DEEPSEEK, chat_batch  # noqa: E402

CHUNK = 20
TRUNCATE_CHARS = 1500
JUDGE_MAX_TOKENS = 16384  # headroom for reasoning mode so JSON is never truncated


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def write_json(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def load_evidence() -> dict[str, str]:
    evidence_map: dict[str, str] = {}
    for row in read_json(QA_DRAFTS):
        for qa in row.get('qas', []) or []:
            qid = qa.get('qa_id', '')
            texts = [f"• {u.get('evidence_text', '').strip()}"
                     for u in (qa.get('evidence_units') or []) if u.get('evidence_text')]
            if qid and texts:
                evidence_map[qid] = '\n'.join(texts)
    return evidence_map


def format_chunks(chunks: list[dict], max_chars: int | None = None) -> str:
    parts = []
    for c in chunks:
        text = c['text']
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + f"... (truncated, original {len(c['text'])} chars)"
        parts.append(f"--- Chunk (rank={c['rank']}, score={float(c.get('score', 0.0)):.4f}) ---\n{text}")
    return '\n\n'.join(parts)


def build_call(row: dict, system_prompt: str, user_template: str, evidence_map: dict, truncate: bool) -> dict:
    chunks_text = format_chunks(row.get('context_chunks', []), max_chars=TRUNCATE_CHARS if truncate else None)
    user_prompt = (user_template
                   .replace('{question}', row.get('question', ''))
                   .replace('{ground_truth}', row.get('ground_truth_answer', ''))
                   .replace('{evidence}', evidence_map.get(row['query_id'], '(no reference evidence available)'))
                   .replace('{chunks}', chunks_text)
                   .replace('{answer}', row.get('generated_answer', '')))
    return {'system': system_prompt, 'user': user_prompt}


def parse_judge(content: str) -> dict | None:
    try:
        cleaned = content.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.split('\n', 1)[-1]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
        parsed = json.loads(cleaned)
        scores = {}
        for d in DIMS:
            scores[d] = int(parsed.get(d, 0))
            scores[f'{d}_reason'] = str(parsed.get(f'{d}_reason', ''))
        if any(not (1 <= scores[d] <= 5) for d in DIMS):
            return None
        if any(not scores[f'{d}_reason'].strip() for d in DIMS):
            return None
        return scores
    except Exception:  # noqa: BLE001
        return None


def judge_batch(rows: list[dict], system_prompt: str, user_template: str, evidence_map: dict,
                workers: int, truncate: bool) -> tuple[list[dict], list[dict]]:
    """Returns (score_dicts, results); failed rows carry an error flag in scores"""
    calls, metas = [], []
    for row in rows:
        if not row.get('generated_answer'):
            metas.append({'row': row, 'skip': True})
            calls.append({'system': system_prompt, 'user': ''})
            continue
        calls.append(build_call(row, system_prompt, user_template, evidence_map, truncate))
        metas.append({'row': row, 'skip': False})

    results = [None] * len(calls)

    def on_result(idx: int, result: dict, _res=results) -> None:
        _res[idx] = result

    chat_batch(calls, model=MODEL_JUDGE, temperature=JUDGE_TEMP, workers=workers,
               max_tokens=JUDGE_MAX_TOKENS,
               api_base=API_BASE_DEEPSEEK, api_key=API_KEY_DEEPSEEK, on_result=on_result)

    score_rows = []
    for meta, res in zip(metas, results):
        row = meta['row']
        base = {
            'query_id': row['query_id'], 'book_title': row.get('book_title', ''),
            'scheme': row.get('scheme', ''), 'question': row.get('question', ''),
            'ground_truth_answer': row.get('ground_truth_answer', ''),
            'question_type': row.get('question_type', ''), 'difficulty': row.get('difficulty', ''),
            'context_chunks': row.get('context_chunks', []),
            'generated_answer': row.get('generated_answer', ''),
            'judge_model': MODEL_JUDGE,
            'judge_model_actual': (res or {}).get('model_actual', ''),
            'judge_usage': (res or {}).get('usage', {}),
        }
        if meta['skip']:
            for d in DIMS:
                base[d] = 0
                base[f'{d}_reason'] = 'empty_answer'
            base['error'] = 'empty_answer'
        else:
            scores = parse_judge((res or {}).get('content', '')) if (res or {}).get('ok') else None
            if scores is None:
                for d in DIMS:
                    base[d] = 0
                    base[f'{d}_reason'] = 'parse_failed'
                base['error'] = str((res or {}).get('error') or 'parse_failed')[:300]
            else:
                base.update(scores)
                base['error'] = ''
        score_rows.append(base)
    return score_rows, results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['A', 'B', 'C', 'D'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--book', default=None)
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--ext', action='store_true', help='judge remaining questions (read/write separate directory; off by default)')
    ap.add_argument('--fill18', action='store_true', help='judge the 18 fill questions (read/write separate directory; off by default)')
    ap.add_argument('--yes', action='store_true')
    args = ap.parse_args()
    if args.ext and args.fill18:
        print('--ext and --fill18 cannot be used together')
        return 1

    if args.fill18:
        ans_root, judg_root = ANSWERS_FILL18_DIR, JUDG_FILL18_DIR
    elif args.ext:
        ans_root, judg_root = ANSWERS_EXT_DIR, JUDG_EXT_DIR
    else:
        ans_root, judg_root = ANSWERS_OUT_DIR, JUDG_OUT_DIR

    system_prompt = (PROMPTS_DIR / 'judge_system_3dim.txt').read_text(encoding='utf-8').strip()
    user_template = (PROMPTS_DIR / 'judge_user_3dim.txt').read_text(encoding='utf-8').strip()
    evidence_map = load_evidence()

    ans_files = [p for p in sorted((ans_root / 'by_book').glob(f'*.four{args.arm}.top5.json'))
                 if (args.book is None or p.name.startswith(args.book + '.'))]
    if not ans_files:
        flag = '（--fill18）' if args.fill18 else ('（--ext）' if args.ext else '')
        print('answer files not found; run 02_generate_answers.py first' + flag)
        return 1

    done_new, t0 = 0, time.time()
    for p in ans_files:
        book = p.name.split('.four')[0]
        rows = read_json(p)
        if args.limit:
            rows = rows[:args.limit]
        order = [r['query_id'] for r in rows]
        out_path = judg_root / 'by_book' / f'{book}.four{args.arm}.3dim.json'
        existing = read_json(out_path) if out_path.exists() else []
        by_qid = {r['query_id']: r for r in existing}
        # To judge = no record, or failed (empty_answer is skipped permanently), or an old truncated bad row (parse_failed with empty error)
        def _needs_retry(r: dict) -> bool:
            old = by_qid.get(r['query_id'])
            if old is None:
                return True
            err = old.get('error', '')
            if err == 'empty_answer':
                return False
            if err:
                return True
            if old.get('grounding') == 0 and old.get('grounding_reason') == 'parse_failed':
                return True
            return any(not str(old.get(f'{d}_reason') or '').strip() for d in DIMS)

        pending = [r for r in rows if _needs_retry(r)]
        print(f'[{book}] total {len(rows)} | existing {len(existing)} | to judge {len(pending)}')
        if not pending or not args.yes:
            continue

        for start in range(0, len(pending), CHUNK):
            batch = pending[start:start + CHUNK]
            score_rows, results = judge_batch(batch, system_prompt, user_template, evidence_map,
                                              args.workers, truncate=False)
            # 400 truncation retries
            retry_rows = [sr for sr in score_rows if '400' in sr.get('error', '')]
            if retry_rows:
                print(f'  400 truncation retries: {len(retry_rows)}')
                ok_scores = []
                for _ in range(1):
                    sr2, _ = judge_batch(retry_rows, system_prompt, user_template, evidence_map,
                                         args.workers, truncate=True)
                    ok_scores = sr2
                for sr in ok_scores:
                    for i, old in enumerate(score_rows):
                        if old['query_id'] == sr['query_id']:
                            score_rows[i] = sr
                            break
            for sr in score_rows:
                by_qid[sr['query_id']] = sr
            write_json(out_path, [by_qid[q] for q in order if q in by_qid])
            ok_n = sum(1 for sr in score_rows if not sr.get('error'))
            print(f'  Checkpoint: +{len(batch)} (ok {ok_n})')
        done_new += len(pending)

    print(f'Arm {args.arm}: new/retried this run {done_new}, elapsed {time.time()-t0:.0f}s, mode {"RUN" if args.yes else "DRY-RUN"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
