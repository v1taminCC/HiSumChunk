# -*- coding: utf-8 -*-
"""Builds fill18 four-arm contexts for the 18 fill questions (writes only to 02-contexts-fill18/).

Protocol:
- 18 questions = excluded_query_ids in sample_200.json (5 books);
- A/B = first 5 blocks of the deep-ranked list for that question (B strips the H2 prefix) — runs_top20/ for 4 books, original top10 for Rime (its top10 already covers all 6 blocks of the book);
- C/D = first 5 body blocks of the deep-ranked list (top20) (all if fewer than 5; Rime=4 blocks), ranks renumbered 1..N with original_rank preserved;
- A cross-checked byte-for-byte against old-answer contexts; all sources read-only.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

FOUR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FOUR / 'scripts'))
from config import SAMPLE_DIR, TOPK  # noqa: E402

OUT_DIR = FOUR / '02-contexts-fill18'
RUNS_TOP20 = FOUR / 'runs_top20'
TOP20_BOOKS = ['Lord Jim', 'Sense and Sensibility', "Tess of the D'Urbervilles", 'The Prince']
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def write_json(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        'fourarm_base_01', FOUR / 'scripts' / '01_build_sample_contexts.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    base = load_base_module()
    qa_map, runs_by_query, chunk_map, prefix_map, old_ctx = base.load_all()
    sample = read_json(SAMPLE_DIR / 'sample_200.json')
    excluded = list(sample['excluded_query_ids'])

    deep = {}
    for book in TOP20_BOOKS:
        for r in read_json(RUNS_TOP20 / f'{book}.ours.top20.json'):
            deep[r['query_id']] = r['top_k']

    contexts = {arm: [] for arm in 'ABCD'}
    strip_stats = defaultdict(lambda: {'strip_ok': 0, 'strip_fallback': 0})
    a_mismatch = 0
    c_d_outline = 0
    missing = 0
    manifest = []

    for qid in excluded:
        if qid not in qa_map:
            missing += 1
            continue
        qa = qa_map[qid]
        book = qa['book_title']
        pmap = prefix_map.get(book, {})
        rank_list = deep.get(qid) or runs_by_query[qid]['top_k']
        used_deep = qid in deep
        q_stats = {arm: {'strip_ok': 0, 'strip_fallback': 0} for arm in 'BD'}

        def make_entry(item, arm, rank):
            c = chunk_map[item['chunk_id']]
            text, is_outline = c['text'], bool(c.get('is_outline'))
            if arm in ('B', 'D') and not is_outline:
                info = c.get('info', '')
                m = base.INFO_RE.search(info)
                prefix = pmap.get((int(m.group(1)), int(m.group(2))), '') if m else ''
                new_text, ok = base.strip_prefix(text, prefix)
                if ok:
                    q_stats[arm]['strip_ok'] += 1
                    strip_stats[arm]['strip_ok'] += 1
                    text = new_text
                else:
                    q_stats[arm]['strip_fallback'] += 1
                    strip_stats[arm]['strip_fallback'] += 1
            return {'chunk_id': item['chunk_id'], 'rank': rank, 'score': item['score'],
                    'is_outline': is_outline, 'original_rank': item['rank'], 'text': text}

        top5 = rank_list[:TOPK]
        body_items = [it for it in rank_list if not chunk_map[it['chunk_id']].get('is_outline')][:TOPK]
        arms = {
            'A': [make_entry(it, 'A', i + 1) for i, it in enumerate(top5)],
            'B': [make_entry(it, 'B', i + 1) for i, it in enumerate(top5)],
            'C': [make_entry(it, 'C', i + 1) for i, it in enumerate(body_items)],
            'D': [make_entry(it, 'D', i + 1) for i, it in enumerate(body_items)],
        }
        if any(e['is_outline'] for e in arms['C'] + arms['D']):
            c_d_outline += 1
        for arm in 'ABCD':
            contexts[arm].append({
                'query_id': qid, 'book_title': book, 'chapter_id': qa['chapter_id'],
                'question': qa['question'], 'ground_truth_answer': qa['ground_truth_answer'],
                'question_type': qa['question_type'], 'difficulty': qa['difficulty'],
                'chunks': arms[arm],
                'meta': {'arm': arm, 'subset': 'fill18', **(q_stats[arm] if arm in 'BD' else {})},
            })
        old = old_ctx.get(qid, [])
        new_a = contexts['A'][-1]['chunks']
        if len(old) != len(new_a) or any(o[0] != n['chunk_id'] or o[1] != n['text'] for o, n in zip(old, new_a)):
            a_mismatch += 1
            print(f'[A mismatch] {qid}')
        manifest.append({
            'query_id': qid, 'book_title': book, 'used_deep_top20': used_deep,
            'rank_list_len': len(rank_list), 'top5_len': len(top5), 'body_len': len(body_items),
            'max_original_rank_used': max([it['rank'] for it in body_items], default=0),
        })

    def wc(t):
        return len(WORD_RE.findall(t))

    audit = ['=== fill18 four-arm context audit ===',
             f'Questions: {len(contexts["A"])}/18 | missing qa {missing} (expect 0)',
             f'A differs from old-answer context: {a_mismatch} (expect 0)',
             f'C/D contain H1 slices: {c_d_outline} (expect 0)']
    for arm in 'ABCD':
        cnt = Counter(len(r['chunks']) for r in contexts[arm])
        lens = [sum(wc(c['text']) for c in r['chunks']) for r in contexts[arm]]
        audit.append(f'Arm {arm}: block-count distribution {dict(cnt)} | word count min/median/max = {min(lens)}/{int(statistics.median(lens))}/{max(lens)}')
    for arm in 'BD':
        audit.append(f'Arm {arm}: prefix stripped ok {strip_stats[arm]["strip_ok"]} / fallback {strip_stats[arm]["strip_fallback"]}')
    deep_n = sum(1 for m in manifest if m['used_deep_top20'])
    audit.append(f'Questions using top20: {deep_n}/18 (the rest use top10, Rime)')
    audit.append('Depth usage: ' + ', '.join(f"{m['query_id'].split('.')[0][:18]}..r{m['max_original_rank_used']}" for m in manifest if m['used_deep_top20'])[:500])

    for arm in 'ABCD':
        write_json(OUT_DIR / f'contexts_{arm}.json', contexts[arm])
    write_json(OUT_DIR / 'fill18_manifest.json', {
        'count': len(manifest), 'top20_books': TOP20_BOOKS,
        'items': manifest,
    })
    (OUT_DIR / 'context_audit.txt').write_text('\n'.join(audit), encoding='utf-8')

    body_expected = all((m['body_len'] == 5) or (m['book_title'] == 'The Rime of the Ancient Mariner' and m['body_len'] == 4)
                        for m in manifest)
    ok = (missing == 0 and a_mismatch == 0 and c_d_outline == 0 and body_expected and len(manifest) == 18)
    print('\n'.join(audit))
    print('Step2 result:', 'PASS' if ok else 'FAIL', '| output:', OUT_DIR)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
