# -*- coding: utf-8 -*-
"""G1: source data validation + fingerprint (read-only sources; writes 00-manifest/source_manifest.json)"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    ANSWERS_OUT_DIR, CHUNKS_DIR, JUDG_OUT_DIR, MANIFEST_DIR, OLD_ANSWERS_DIR,
    OLD_JUDG_DIR, PROMPTS_DIR, QA_DRAFTS, RUNS_DIR, SPLIT_DIR,
)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def list_files(dir_path: Path, pattern: str) -> list[Path]:
    return sorted(p for p in dir_path.glob(pattern) if p.is_file() and not p.name.startswith('.'))


def main() -> int:
    errors: list[str] = []
    manifest: dict = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'files': [],
        'counts': {},
    }

    # 1) QA
    qa_rows = json.loads(QA_DRAFTS.read_text(encoding='utf-8'))
    qa_total = sum(len(r.get('qas', []) or []) for r in qa_rows)
    manifest['counts']['qa_rows'] = len(qa_rows)
    manifest['counts']['qa_total'] = qa_total
    if qa_total != 577:
        errors.append(f'QA total should be 577, got {qa_total}')

    # 2) File and row counts per directory
    run_files = list_files(RUNS_DIR, '*.ours.top10.json')
    chunk_files = list_files(CHUNKS_DIR, '*.ours.json')
    ans_files = list_files(OLD_ANSWERS_DIR, '*.in_book.ours.top5.json')
    jud_files = list_files(OLD_JUDG_DIR, '*.3dim.json')
    split_files = list_files(SPLIT_DIR, '*_L1Outline.json')

    manifest['counts'].update({
        'run_files': len(run_files), 'chunk_files': len(chunk_files),
        'old_answer_files': len(ans_files), 'judg_files': len(jud_files),
        'l1outline_files': len(split_files),
    })
    for name, files, expect in [('runs', run_files, 21), ('chunks', chunk_files, 21),
                                ('answers', ans_files, 21), ('judgments', jud_files, 21),
                                ('l1outline', split_files, 21)]:
        if len(files) != expect:
            errors.append(f'{name} should have {expect} files, got {len(files)}')

    # File-level md5
    targets: list[Path] = [QA_DRAFTS] + run_files + chunk_files + ans_files + jud_files + split_files
    targets += list_files(PROMPTS_DIR, '*.txt')
    for p in targets:
        manifest['files'].append({
            'path': str(p), 'size': p.stat().st_size, 'md5': md5(p),
        })

    # Row counts
    ans_rows = sum(len(json.loads(p.read_text(encoding='utf-8'))) for p in ans_files)
    jud_rows = 0
    ours_rows = 0
    for p in jud_files:
        for j in json.loads(p.read_text(encoding='utf-8')):
            jud_rows += 1
            if j.get('scheme') == 'ours':
                ours_rows += 1
    manifest['counts']['old_answer_rows'] = ans_rows
    manifest['counts']['judg_rows'] = jud_rows
    manifest['counts']['judg_ours_rows'] = ours_rows
    if ans_rows != 577:
        errors.append(f'Old answers should have 577 rows, got {ans_rows}')
    if ours_rows != 577:
        errors.append(f'ours judgments should have 577 rows, got {ours_rows}')

    prompts = list_files(PROMPTS_DIR, '*.txt')
    if len(prompts) != 4:
        errors.append(f'Expected 4 prompts, got {len(prompts)}')

    manifest['errors'] = errors
    manifest['status'] = 'PASS' if not errors else 'FAIL'
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_DIR / 'source_manifest.json'
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Source validation: {manifest["status"]} | {len(manifest["files"])} files')
    for k, v in manifest['counts'].items():
        print(f'  {k}: {v}')
    for e in errors:
        print('  [ERROR]', e)
    print('manifest ->', out)
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
