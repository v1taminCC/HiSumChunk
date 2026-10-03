# -*- coding: utf-8 -*-
"""Source-chunk snapshot builder (one-off, read-only export from the main experiment)

- Takes the 17,271 body chunks of ours and strips the H2 prefix (replicating the four-arm prefix_map semantics)
- Exports data/source/source_chunks.jsonl + chapter_map.json (fully self-contained afterwards)

Usage:
  python scripts/00_build_source_chunks.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    L1_OUTLINE_DIR, MAIN_CHUNKS_DIR, PUNCT, REPORTS_DIR, SOURCE,
)

INFO_RE = re.compile(r"Region\s*(\d+)---Block\s*(\d+)---Subchunk\s*(\d+)")


def ensure_prefix(text: str) -> str:
    if not text:
        return ''
    return text if text[-1] in PUNCT else text + '。'


def build_prefix_map() -> dict[str, dict[tuple[int, int], str]]:
    """Replicates the four-arm loading semantics: sorted glob + book=split('_MergeRule')[0] override"""
    pmap: dict[str, dict[tuple[int, int], str]] = {}
    for p in sorted(L1_OUTLINE_DIR.glob('*_L1Outline.json')):
        book = p.name.split('_MergeRule')[0]
        data = json.loads(p.read_text(encoding='utf-8'))
        m: dict[tuple[int, int], str] = {}
        for ri, region in enumerate(data.get('h2_regions', []), 1):
            for ci, ch in enumerate(region.get('chunks', []), 1):
                h2 = ch.get('h2_outline', '')
                if h2:
                    m[(ri, ci)] = ensure_prefix(h2)
        pmap[book] = m
    return pmap


def main() -> int:
    SOURCE.mkdir(parents=True, exist_ok=True)
    pmap = build_prefix_map()

    stats = Counter()
    chapters: dict[str, dict[str, dict]] = defaultdict(dict)
    rows_out = []
    book_stats = {}
    for p in sorted(MAIN_CHUNKS_DIR.glob('*.ours.json')):
        book = p.name.replace('.ours.json', '')
        chunks = json.loads(p.read_text(encoding='utf-8'))
        m = pmap.get(book, {})
        b_ok = b_fb = 0
        for x in chunks:
            if x.get('is_outline'):
                continue
            stats['total'] += 1
            text = str(x.get('text', ''))
            info = str(x.get('info', ''))
            mi = INFO_RE.match(info)
            region = block = sub = None
            prefix = ''
            if mi:
                region, block, sub = int(mi.group(1)), int(mi.group(2)), int(mi.group(3))
                prefix = m.get((region, block), '')
            clean = text
            if not prefix:
                stats['no_prefix'] += 1
            elif text.startswith(prefix):
                clean = text[len(prefix):]
                stats['strip_ok'] += 1
                b_ok += 1
            else:
                stats['strip_mismatch'] += 1
                b_fb += 1
            chapter = str(x.get('active_chapter_id', '') or '')
            rows_out.append({
                'chunk_id': x['chunk_id'],
                'book': book,
                'chapter_id': chapter,
                'covered_chapter_ids': list(x.get('covered_chapter_ids', []) or []),
                'crosses_chapter': bool(x.get('crosses_chapter', False)),
                'region': region, 'block': block, 'sub': sub,
                'prefix_chars': len(prefix),
                'text': clean,
                'text_chars': len(clean),
                'source_start': x.get('source_start'),
                'source_end': x.get('source_end'),
                'info': info,
            })
            ch = chapters[book].setdefault(chapter, {'chunk_ids': [], 'chars': 0})
            ch['chunk_ids'].append(x['chunk_id'])
            ch['chars'] += len(clean)
        book_stats[book] = {'chunks': b_ok + b_fb, 'strip_ok': b_ok, 'strip_fallback': b_fb}

    with open(SOURCE / 'source_chunks.jsonl', 'w', encoding='utf-8') as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (SOURCE / 'chapter_map.json').write_text(
        json.dumps(chapters, ensure_ascii=False), encoding='utf-8')

    lines = ['=== CR source-chunk snapshot audit ===',
             f'Body chunks: {stats["total"]} (expect 17271)',
             f'Prefix stripped ok: {stats["strip_ok"]} | blocks without prefix: {stats["no_prefix"]} | '
             f'Prefix mismatches: {stats["strip_mismatch"]} (expect 0)',
             f'Chapters: {sum(len(v) for v in chapters.values())}']
    bad = {b: v for b, v in book_stats.items() if v['strip_fallback']}
    lines.append(f'Books with prefix mismatches: {bad if bad else "none"}')
    lines.append('Blocks per book (first 5 shown):')
    for b in sorted(book_stats)[:5]:
        lines.append(f'  {b}: {book_stats[b]}')
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = '\n'.join(lines)
    (REPORTS_DIR / 'cr_source_audit.txt').write_text(report, encoding='utf-8')
    print(report)

    ok = stats['strip_mismatch'] == 0 and stats['total'] == 17271
    print('\nG0 result:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
