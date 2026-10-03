# -*- coding: utf-8 -*-
"""G2: sample 200 questions and build four-arm contexts (A=original top-5 / B=no H2 prefix / C=no H1 backfilled / D=both removed)

Read-only: runs, chunks, L1Outline, QA, old answers (for consistency checks)
Writes: 01-sample/sample_200.json, 02-contexts/contexts_{A,B,C,D}.json, context_audit.txt
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    CONTEXTS_DIR, CHUNKS_DIR, MIN_BODY_CHUNKS, MIN_BOOKS, OLD_ANSWERS_DIR, PUNCT,
    QA_DRAFTS, RUNS_DIR, SAMPLE_DIR, SAMPLE_SIZE, SEED, SPLIT_DIR, TOPK, TOPN,
)

INFO_RE = re.compile(r'Region\s+(\d+)---Block\s+(\d+)')
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def write_json(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def ensure_prefix(outline: str) -> str:
    if not outline:
        return ''
    return outline if outline[-1] in PUNCT else outline + '。'


def load_all() -> tuple[dict, dict, dict, dict, dict]:
    """Returns qa_map, runs_by_query, chunk_map, prefix_map(book->(region,chunk)->prefix), old_ctx"""
    qa_rows = read_json(QA_DRAFTS)
    qa_map: dict[str, dict] = {}
    for row in qa_rows:
        book = row.get('book_title', '')
        for q in row.get('qas', []) or []:
            qid = q.get('qa_id', '')
            if not qid:
                continue
            qa_map[qid] = {
                'book_title': book,
                'chapter_id': row.get('chapter_id', ''),
                'question': q.get('question', ''),
                'ground_truth_answer': q.get('answer', ''),
                'question_type': q.get('question_type', ''),
                'difficulty': q.get('difficulty', ''),
            }

    runs_by_query: dict[str, dict] = {}
    for p in sorted(RUNS_DIR.glob('*.ours.top10.json')):
        for r in read_json(p):
            runs_by_query[r['query_id']] = r

    chunk_map: dict[str, dict] = {}
    for p in sorted(CHUNKS_DIR.glob('*.ours.json')):
        for c in read_json(p):
            chunk_map[c['chunk_id']] = c

    prefix_map: dict[str, dict] = {}
    for p in sorted(SPLIT_DIR.glob('*_L1Outline.json')):
        book = p.name.split('_MergeRule')[0]
        data = read_json(p)
        m: dict = {}
        for ri, region in enumerate(data.get('h2_regions', []), 1):
            for ci, ch in enumerate(region.get('chunks', []), 1):
                h2 = ch.get('h2_outline', '')
                if h2:
                    m[(ri, ci)] = ensure_prefix(h2)
        prefix_map[book] = m

    old_ctx: dict[str, list[tuple[str, str]]] = {}
    for p in sorted(OLD_ANSWERS_DIR.glob('*.in_book.ours.top5.json')):
        for row in read_json(p):
            old_ctx[row['query_id']] = [
                (c.get('chunk_id', ''), c.get('text', '')) for c in row.get('context_chunks', [])
            ]
    return qa_map, runs_by_query, chunk_map, prefix_map, old_ctx


def strip_prefix(text: str, prefix: str) -> tuple[str, bool]:
    if prefix and text.startswith(prefix):
        return text[len(prefix):], True
    return text, False


def build_arms(qid: str, run: dict, chunk_map: dict, pmap: dict) -> dict:
    items = run.get('top_k', [])[:TOPN]
    top5 = items[:TOPK]
    ctx = {}
    stats_by_arm = {arm: {'strip_ok': 0, 'strip_fallback': 0} for arm in 'ABCD'}

    def make_entry(item: dict, arm: str, rank: int, original_rank: int) -> dict:
        c = chunk_map[item['chunk_id']]
        text = c['text']
        is_outline = bool(c.get('is_outline'))
        if arm in ('B', 'D') and not is_outline:
            info = c.get('info', '')
            m = INFO_RE.search(info)
            prefix = pmap.get((int(m.group(1)), int(m.group(2))), '') if m else ''
            new_text, ok = strip_prefix(text, prefix)
            if ok:
                stats_by_arm[arm]['strip_ok'] += 1
                text = new_text
            else:
                stats_by_arm[arm]['strip_fallback'] += 1
        return {
            'chunk_id': item['chunk_id'], 'rank': rank, 'score': item.get('score', 0.0),
            'is_outline': is_outline, 'original_rank': original_rank, 'text': text,
        }

    # A / B: original top-5 (H1 slices kept; B strips the body prefix)
    for arm in ('A', 'B'):
        ctx[arm] = [make_entry(it, arm, i + 1, it['rank']) for i, it in enumerate(top5)]

    # C / D: first 5 body blocks from top-10 (backfill), renumbered 1..5 (original ranks kept)
    body_items = [it for it in items if not chunk_map[it['chunk_id']].get('is_outline')][:TOPK]
    for arm in ('C', 'D'):
        ctx[arm] = [make_entry(it, arm, i + 1, it['rank']) for i, it in enumerate(body_items)]

    return {'arms': ctx, 'stats': stats_by_arm}


def main() -> int:
    qa_map, runs_by_query, chunk_map, prefix_map, old_ctx = load_all()
    print(f'Loaded: QA {len(qa_map)} | runs {len(runs_by_query)} | chunks {len(chunk_map)} | old answers {len(old_ctx)}')

    # Sampling pool: top-10 body blocks >= 5
    excluded, pool = [], []
    for qid, run in runs_by_query.items():
        body_n = sum(1 for it in run.get('top_k', [])[:TOPN]
                     if not chunk_map.get(it['chunk_id'], {}).get('is_outline'))
        (pool if body_n >= MIN_BODY_CHUNKS else excluded).append(qid)
    print(f'Pool {len(pool)} | excluded (body<5) {len(excluded)}')

    # Sampling (fixed seed; must cover >=18 books, otherwise re-sample with a new seed)
    attempts = []
    chosen = None
    seed_used = SEED
    for k in range(10):
        s = SEED + k
        rng = random.Random(s)
        cand = rng.sample(pool, SAMPLE_SIZE)
        books = {qa_map[q]['book_title'] for q in cand}
        attempts.append({'seed': s, 'books': len(books)})
        if len(books) >= MIN_BOOKS:
            chosen, seed_used = cand, s
            break
    if chosen is None:
        print('[ERROR] 10 attempts failed to meet the coverage condition')
        return 1
    chosen.sort()
    per_book = Counter(qa_map[q]['book_title'] for q in chosen)
    print(f'Sampling done: seed={seed_used} | {len(chosen)} questions | {len(per_book)} books')
    for b, n in sorted(per_book.items()):
        print(f'  {b}: {n}')

    # Build four-arm contexts
    contexts = {arm: [] for arm in 'ABCD'}
    strip_stats = defaultdict(lambda: {'strip_ok': 0, 'strip_fallback': 0})
    a_text_mismatch = 0
    c_d_outline_left = 0
    chunk_count_bad = 0

    for qid in chosen:
        run = runs_by_query[qid]
        qa = qa_map[qid]
        pmap = prefix_map.get(qa['book_title'], {})
        built = build_arms(qid, run, chunk_map, pmap)
        for arm in 'ABCD':
            entries = built['arms'][arm]
            if len(entries) != TOPK:
                chunk_count_bad += 1
            if arm in ('C', 'D') and any(e['is_outline'] for e in entries):
                c_d_outline_left += 1
            contexts[arm].append({
                'query_id': qid, 'book_title': qa['book_title'], 'chapter_id': qa['chapter_id'],
                'question': qa['question'], 'ground_truth_answer': qa['ground_truth_answer'],
                'question_type': qa['question_type'], 'difficulty': qa['difficulty'],
                'chunks': entries,
                'meta': {'arm': arm, **built['stats'][arm]},
            })
        # A vs. old answers context consistency (block by block)
        old = old_ctx.get(qid, [])
        new_a = contexts['A'][-1]['chunks']
        if len(old) != len(new_a) or any(o[0] != n['chunk_id'] or o[1] != n['text'] for o, n in zip(old, new_a)):
            a_text_mismatch += 1

    # Summary statistics
    def wc(text: str) -> int:
        return len(WORD_RE.findall(text))

    audit = []
    audit.append('=== Four-arm context build audit ===')
    audit.append(f'sample seed={seed_used}, n={len(chosen)}, books={len(per_book)}, pool={len(pool)}, excluded={len(excluded)}')
    audit.append(f'Questions whose A context differs from old answers: {a_text_mismatch} (expect 0)')
    audit.append(f'Questions where C/D still contain H1 slices: {c_d_outline_left} (expect 0)')
    audit.append(f'Occurrences of an arm with block count != 5: {chunk_count_bad} (expect 0)')
    for arm in 'ABCD':
        lens = [sum(wc(c['text']) for c in row['chunks']) for row in contexts[arm]]
        audit.append(f'Arm {arm}: context word count min/median/max = {min(lens)}/{int(statistics.median(lens))}/{max(lens)}')
    for arm in 'BD':
        s = Counter()
        for row in contexts[arm]:
            s['strip_ok'] += row['meta']['strip_ok']
            s['strip_fallback'] += row['meta']['strip_fallback']
        audit.append(f'Arm {arm}: prefix stripped ok {s["strip_ok"]} / fallback {s["strip_fallback"]}')

    # 3 examples
    sample_qids = [q for q in chosen if any(c['is_outline'] for c in next(r['chunks'] for r in contexts['A'] if r['query_id'] == q))][:2]
    sample_qids += [q for q in chosen if q not in sample_qids][:1]
    for qid in sample_qids:
        audit.append(f'\n--- Example {qid} ---')
        audit.append('Question: ' + qa_map[qid]['question'][:120])
        for arm in 'ABCD':
            row = next(r for r in contexts[arm] if r['query_id'] == qid)
            head = ' | '.join(f"[{c['rank']}]{'H1' if c['is_outline'] else 'body'}:" + c['text'][:60].replace('\n', ' ') for c in row['chunks'])
            audit.append(f'  Arm {arm}: {head[:500]}')

    # Persist to disk
    write_json(SAMPLE_DIR / 'sample_200.json', {
        'seed_used': seed_used, 'sample_size': len(chosen), 'pool_size': len(pool),
        'excluded_count': len(excluded), 'coverage_books': len(per_book),
        'attempts': attempts, 'sampled_query_ids': chosen,
        'per_book_counts': dict(sorted(per_book.items())),
        'excluded_query_ids': sorted(excluded),
    })
    for arm in 'ABCD':
        write_json(CONTEXTS_DIR / f'contexts_{arm}.json', contexts[arm])
    (CONTEXTS_DIR / 'context_audit.txt').write_text('\n'.join(audit), encoding='utf-8')

    ok = (a_text_mismatch == 0 and c_d_outline_left == 0 and chunk_count_bad == 0)
    print('\n'.join(audit))
    print('\nG2 result:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
