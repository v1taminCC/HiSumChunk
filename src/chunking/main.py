#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from anchor_utils import (
    annotate_chunks_by_search,
    load_anchor_file,
    read_json,
    remap_anchors,
    save_anchor_file,
    strip_chapter_anchors_with_tracking,
    write_json,
)

RUN_MODE = "all"  # "single" | "all"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))
SINGLE_TXT_FILE = str(DEMO_BOOK_DIR / "Far From the Madding Crowd.txt")
INPUT_TXT_DIR = str(DEMO_BOOK_DIR)
OUTPUT_TXT_DIR = str(DEMO_BOOK_DIR / "work_txt")
OUTPUT_JSON_DIR = str(DEMO_BOOK_DIR)
TIME_REPORT_FILE = str(Path(__file__).resolve().parent / "time_report.txt")
SHOW_PROGRESS = True

CFG = {
    "step1": {"OUTPUT_SUFFIX": "_MergeRule", "ENCODING": "utf-8", "SHOW_PROGRESS": True},
    "step2": {"MODEL_PATH": os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507"), "RUNTIME_ENDPOINT": "http://127.0.0.1:30000", "TEMPERATURE": 0.1, "TOP_P": 0.5, "TOP_K": 15, "MAX_GENERATE_TOKENS": 16000, "MAX_INPUT_LENGTH": 50000, "SYSTEM_PROMPT": "", "BATCH_SIZE": 4, "PAGES_PER_BATCH": 4, "LEVEL1_WEIGHT": 2, "LEVEL2_WEIGHT": 1, "THRESHOLD": 0.2, "PAGE_MARKER_PREFIX": ">>>PAGE ", "PAGE_MARKER_SUFFIX": "<<<", "SEPARATOR_LENGTH": 60, "SHOW_PROGRESS": True},
    "step3": {"OUTPUT_SUFFIX": "_NoHeaderFooter", "OUTPUT_ENCODING": "utf-8", "MODEL_PATH": os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507"), "RUNTIME_ENDPOINT": "http://127.0.0.1:30000", "TEMPERATURE": 0.1, "TOP_P": 0.5, "TOP_K": 15, "MAX_GENERATE_TOKENS": 16000, "MAX_INPUT_LENGTH": 50000, "SYSTEM_PROMPT": "", "BATCH_SIZE": 4, "CATALOG_BATCH_SIZE": 16, "PAGES_PER_BATCH": 4, "PAGE_MARKER_PREFIX": ">>>PAGE ", "PAGE_MARKER_SUFFIX": "<<<", "DETECT_ALL_PAGES": True, "SEPARATOR_LENGTH": 60, "SHOW_PROGRESS": True},
    "step4a": {"OUTPUT_SUFFIX": "_MergeLineBreak.json", "OUTPUT_ENCODING": "utf-8", "MODEL_PATH": os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507"), "RUNTIME_ENDPOINT": "http://127.0.0.1:30000", "TEMPERATURE": 0.1, "TOP_P": 0.5, "TOP_K": 15, "MAX_GENERATE_TOKENS": 16000, "MAX_INPUT_LENGTH": 50000, "SYSTEM_PROMPT": "", "BATCH_SIZE": 16, "SEPARATOR_LENGTH": 60, "SHOW_PROGRESS": True, "SENTENCE_END_PUNCTUATIONS": {"."}},
    "step4b": {"OUTPUT_SUFFIX": "_Level.json", "MODEL_PATH": os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507"), "RUNTIME_ENDPOINT": "http://127.0.0.1:30000", "TEMPERATURE": 0.1, "TOP_P": 0.5, "TOP_K": 15, "MAX_GENERATE_TOKENS": 16000, "MAX_INPUT_LENGTH": 50000, "SYSTEM_PROMPT": "", "CHUNK_SIZE": 6000, "SENTENCE_END_PUNCTUATIONS": ["!", "?", ";", "."], "MERGE_THRESHOLD": 50, "BATCH_SIZE": 4, "SEPARATOR_LENGTH": 60, "SHOW_PROGRESS": True, "PRINT_FINAL_SEGMENTS": True},
    "step5": {"OUTPUT_SUFFIX": "_L2Outline.json", "MODEL_PATH": os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507"), "RUNTIME_ENDPOINT": "http://127.0.0.1:30000", "TEMPERATURE": 0.1, "TOP_P": 0.5, "TOP_K": 15, "MAX_GENERATE_TOKENS": 16000, "MAX_INPUT_LENGTH": 50000, "SYSTEM_PROMPT": "", "SEPARATOR_LENGTH": 60, "SHOW_PROGRESS": True},
    "step6": {"OUTPUT_SUFFIX": "_MergeH2.json", "MODEL_PATH": os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507"), "RUNTIME_ENDPOINT": "http://127.0.0.1:30000", "TEMPERATURE": 0.1, "TOP_P": 0.5, "TOP_K": 15, "MAX_GENERATE_TOKENS": 16000, "MAX_INPUT_LENGTH": 50000, "SYSTEM_PROMPT": "", "MAX_CHUNKS_PER_BATCH": 20, "SEPARATOR_LENGTH": 60, "SHOW_PROGRESS": True},
    "step7": {"OUTPUT_SUFFIX": "_L1Outline.json", "MODEL_PATH": os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507"), "RUNTIME_ENDPOINT": "http://127.0.0.1:30000", "TEMPERATURE": 0.1, "TOP_P": 0.5, "TOP_K": 15, "MAX_GENERATE_TOKENS": 16000, "MAX_INPUT_LENGTH": 50000, "SYSTEM_PROMPT": "", "WORD_LIMIT_0_50": 50, "WORD_LIMIT_50_200": 100, "WORD_LIMIT_201_500": 300, "WORD_LIMIT_501_2000": 1000, "WORD_LIMIT_2001_4000": 2000, "WORD_LIMIT_4001_12000": 3000, "BATCH_SIZE_0_50": 16, "BATCH_SIZE_50_200": 16, "BATCH_SIZE_201_500": 16, "BATCH_SIZE_501_2000": 16, "BATCH_SIZE_2001_4000": 8, "BATCH_SIZE_4001_12000": 3, "OUTLINE_SEPARATOR": ".\n", "SEPARATOR_LENGTH": 60, "SHOW_PROGRESS": True},
    "step8": {"OUTPUT_SUFFIX": "_split.json", "TARGET_LEN": 1000, "OVERLAP": 100, "BACKTRACK_RANGE": 150, "PUNCTUATIONS": ".!?", "SHOW_PROGRESS": True},
}


def collect_files() -> list[Path]:
    if RUN_MODE == "single":
        return [Path(SINGLE_TXT_FILE)]
    return sorted([p for p in Path(INPUT_TXT_DIR).glob("*.txt") if p.is_file()])


def run_script(script: Path, desc: str) -> None:
    if SHOW_PROGRESS:
        print(f"\n{'=' * 60}\nExecuting: {desc}\nScript: {script.name}\n{'=' * 60}")
    subprocess.run([sys.executable, str(script)], cwd=str(script.parent), check=True)


def update_params(script: Path, params: dict) -> None:
    import re

    lines = script.read_text(encoding="utf-8").splitlines(True)
    for key, value in params.items():
        if isinstance(value, str):
            rep = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            rep = "True" if value else "False"
        elif isinstance(value, (int, float)):
            rep = str(value)
        elif isinstance(value, (list, tuple)):
            rep = repr(list(value))
        elif isinstance(value, set):
            rep = repr(set(value))
        else:
            rep = repr(value)
        p1 = re.compile(rf"^(\s*){re.escape(key)}\s*:\s*([^=]+?)\s*=\s*([^#\n]+?)(\s*#.*)?$")
        p2 = re.compile(rf"^(\s*){re.escape(key)}\s*=\s*([^#\n]+?)(\s*#.*)?$")
        for i, line in enumerate(lines):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m = p1.match(line.rstrip())
            if m:
                lines[i] = f"{m.group(1)}{key}: {m.group(2).strip()} = {rep}{m.group(4) or ''}\n"
                break
            m = p2.match(line.rstrip())
            if m:
                lines[i] = f"{m.group(1)}{key} = {rep}{m.group(3) or ''}\n"
                break
    script.write_text("".join(lines), encoding="utf-8")


def save_initial(source_txt: Path, txt_dir: Path, json_dir: Path) -> tuple[Path, Path]:
    base = source_txt.stem
    initial_txt = txt_dir / f"{base}.txt"
    anchor_path = json_dir / f"{base}.anchors.json"
    clean_text, anchors = strip_chapter_anchors_with_tracking(source_txt.read_text(encoding="utf-8"))
    initial_txt.write_text(clean_text, encoding="utf-8")
    save_anchor_file(anchor_path, str(initial_txt), anchors)
    return initial_txt, anchor_path


def remap_save(old_txt: Path, new_txt: Path, old_anchor: Path, new_anchor: Path) -> None:
    old_text = old_txt.read_text(encoding="utf-8")
    new_text = new_txt.read_text(encoding="utf-8")
    anchors = load_anchor_file(old_anchor)
    save_anchor_file(new_anchor, str(new_txt), remap_anchors(old_text, new_text, anchors))


def annotate_step4(json_path: Path, txt_path: Path, anchor_path: Path) -> None:
    data = read_json(json_path)
    data["chunks"] = annotate_chunks_by_search(
        data.get("chunks", []),
        txt_path.read_text(encoding="utf-8"),
        load_anchor_file(anchor_path),
    )
    data["anchor_file"] = str(anchor_path)
    data["source_text_file"] = str(txt_path)
    write_json(json_path, data)


def format_seconds(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1_000)
    return f"{hours} hours {minutes} minutes {secs} seconds {milliseconds} milliseconds"


def write_time_report(report_path: Path, reports: list[dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    total_all_books = 0.0
    aggregate_steps: dict[str, float] = {}

    for report in reports:
        lines.append(f"Book: {report['book']}")
        for step in report["steps"]:
            lines.append(f"  {step['name']}: {format_seconds(step['seconds'])}")
            step_name = str(step["name"])
            aggregate_steps[step_name] = aggregate_steps.get(step_name, 0.0) + float(step["seconds"])
        lines.append(f"  Book total time: {format_seconds(report['total_seconds'])}")
        lines.append("")
        total_all_books += report["total_seconds"]

    if reports:
        lines.append(f"Books processed: {len(reports)}")
        lines.append(f"All books total time: {format_seconds(total_all_books)}")
        lines.append("")
        lines.append("All books step summary:")
        for step_name, step_seconds in aggregate_steps.items():
            percent = (step_seconds / total_all_books * 100) if total_all_books else 0.0
            lines.append(
                f"  {step_name}: {format_seconds(step_seconds)} ({percent:.2f}%)"
            )
    else:
        lines.append("No books to report")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_file(script_dir: Path, source_txt: Path) -> dict:
    txt_dir, json_dir = Path(OUTPUT_TXT_DIR), Path(OUTPUT_JSON_DIR)
    txt_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    base = source_txt.stem
    timings: list[dict[str, float | str]] = []
    book_start = perf_counter()

    def record_step(step_name: str, func):
        step_start = perf_counter()
        result = func()
        elapsed = perf_counter() - step_start
        timings.append({"name": step_name, "seconds": elapsed})
        if SHOW_PROGRESS:
            print(f"[{base}] {step_name} time: {format_seconds(elapsed)}")
        return result

    initial_txt, anchor0 = record_step(
        "Initial clean & anchor save",
        lambda: save_initial(source_txt, txt_dir, json_dir),
    )

    s1 = script_dir / "1-merge_punctuation_marks.py"
    txt1 = txt_dir / f"{base}{CFG['step1']['OUTPUT_SUFFIX']}.txt"
    anchor1 = json_dir / f"{txt1.stem}.anchors.json"
    record_step(
        "Step 1",
        lambda: (
            update_params(s1, {"INPUT_FILE": str(initial_txt), "OUTPUT_DIR": str(txt_dir), **CFG["step1"]}),
            run_script(s1, f"{base} - Step 1"),
            remap_save(initial_txt, txt1, anchor0, anchor1),
        ),
    )

    s2 = script_dir / "2-has_title.py"
    title_txt = txt_dir / f"{txt1.stem}_Title.txt"
    no_title_txt = txt_dir / f"{txt1.stem}_NoTitle.txt"

    def step2_func():
        update_params(s2, {"INPUT_FILE": str(txt1), **CFG["step2"]})
        run_script(s2, f"{base} - Step 2")
        txt2_local = title_txt if title_txt.exists() else no_title_txt
        anchor2_local = json_dir / f"{txt2_local.stem}.anchors.json"
        shutil.copy2(anchor1, anchor2_local)
        has_title_local = txt2_local == title_txt
        return txt2_local, anchor2_local, has_title_local

    txt2, anchor2, has_title = record_step("Step 2", step2_func)

    s3 = script_dir / "3-clear_noise.py"
    txt3 = txt_dir / f"{txt2.stem}{CFG['step3']['OUTPUT_SUFFIX']}.txt"
    anchor3 = json_dir / f"{txt3.stem}.anchors.json"
    record_step(
        "Step 3",
        lambda: (
            update_params(s3, {"INPUT_FILE": str(txt2), **CFG["step3"]}),
            run_script(s3, f"{base} - Step 3"),
            remap_save(txt2, txt3, anchor2, anchor3),
        ),
    )

    def step4_func():
        if has_title:
            s4 = script_dir / "4.2-Split_Level.py"
            update_params(s4, {"INPUT_FILE": str(txt3), "OUTPUT_DIR": str(json_dir), **CFG["step4b"]})
            run_script(s4, f"{base} - Step 4b")
            j4_local = json_dir / f"{txt3.stem}{CFG['step4b']['OUTPUT_SUFFIX']}"
        else:
            s4 = script_dir / "4.1-merge_line_break.py"
            update_params(s4, {"INPUT_FILE": str(txt3), "OUTPUT_DIR": str(json_dir), **CFG["step4a"]})
            run_script(s4, f"{base} - Step 4a")
            j4_local = json_dir / f"{txt3.stem}{CFG['step4a']['OUTPUT_SUFFIX']}"
        annotate_step4(j4_local, txt3, anchor3)
        return j4_local

    j4 = record_step("Step 4b" if has_title else "Step 4a", step4_func)

    s5 = script_dir / "5-Outline_A.py"
    j5 = j4.parent / f"{j4.stem}{CFG['step5']['OUTPUT_SUFFIX']}"
    record_step(
        "Step 5",
        lambda: (
            update_params(s5, {"INPUT_FILE": str(j4), **CFG["step5"]}),
            run_script(s5, f"{base} - Step 5"),
        ),
    )

    s6 = script_dir / "6-Merge_H2.py"
    j6 = j5.parent / f"{j5.stem}{CFG['step6']['OUTPUT_SUFFIX']}"
    record_step(
        "Step 6",
        lambda: (
            update_params(s6, {"INPUT_FILE": str(j5), **CFG["step6"]}),
            run_script(s6, f"{base} - Step 6"),
        ),
    )

    s7 = script_dir / "7-Outline_B.py"
    j7 = j6.parent / f"{j6.stem}{CFG['step7']['OUTPUT_SUFFIX']}"
    record_step(
        "Step 7",
        lambda: (
            update_params(s7, {"INPUT_FILE": str(j6), **CFG["step7"]}),
            run_script(s7, f"{base} - Step 7"),
        ),
    )

    s8 = script_dir / "8-split.py"
    final_json = j7.parent / f"{j7.stem}{CFG['step8']['OUTPUT_SUFFIX']}"
    record_step(
        "Step 8",
        lambda: (
            update_params(s8, {"INPUT_FILE": str(j7), **CFG["step8"]}),
            run_script(s8, f"{base} - Step 8"),
        ),
    )

    total_seconds = perf_counter() - book_start

    if SHOW_PROGRESS:
        print(f"\nComplete: {source_txt.name}\nFinal output: {final_json}\nTotal time: {format_seconds(total_seconds)}")

    return {
        "book": source_txt.name,
        "steps": timings,
        "total_seconds": total_seconds,
        "final_output": str(final_json),
    }


def main() -> None:
    files = collect_files()
    if not files:
        print("No files found to process")
        sys.exit(1)
    script_dir = Path(__file__).parent.absolute()
    report_path = Path(TIME_REPORT_FILE)
    reports: list[dict] = []
    write_time_report(report_path, reports)

    for file_path in files:
        reports.append(process_file(script_dir, file_path))
        write_time_report(report_path, reports)
        if SHOW_PROGRESS:
            print(f"\nTime report updated: {report_path}")


if __name__ == "__main__":
    main()
