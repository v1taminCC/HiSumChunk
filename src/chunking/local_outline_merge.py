#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import importlib.util
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
EXAMPLE_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))
from time import perf_counter
from types import ModuleType
from typing import Any, Callable

# ================================================================================================
# ================================ Configurable Parameters =======================================
# ================================================================================================

RUN_MODE = "all"  # "all" | "single"
SINGLE_INPUT_FILE = str(EXAMPLE_BOOK_DIR / "Far From the Madding Crowd_MergeRule_Title_NoHeaderFooter_Level.json")
INPUT_JSON_DIR = str(EXAMPLE_BOOK_DIR)
TIME_REPORT_FILE = str(BASE_DIR / "local_modify_time_report.txt")
SHOW_PROGRESS = True

START_FILE_SUFFIXES = ["_Level.json", "_MergeLineBreak.json"]
EXPECTED_FILE_COUNT = 21
OVERWRITE_OUTPUTS = True

SCRIPT_FILES = {
    "step5": "5-Outline_A.py",
    "step6": "6-Merge_H2.py",
    "step7": "7-Outline_B.py",
    "step8": "8-split.py",
}

CFG = {
    "step5": {
        "OUTPUT_SUFFIX": "_L2Outline.json",
        "MODEL_PATH": os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507"),
        "RUNTIME_ENDPOINT": "http://127.0.0.1:30000",
        "TEMPERATURE": 0.1,
        "TOP_P": 0.5,
        "TOP_K": 15,
        "MAX_GENERATE_TOKENS": 16000,
        "MAX_INPUT_LENGTH": 50000,
        "SYSTEM_PROMPT": "",
        "SUMMARY_LANGUAGE": "en",
        "SEPARATOR_LENGTH": 60,
        "SHOW_PROGRESS": True,
    },
    "step6": {
        "OUTPUT_SUFFIX": "_MergeH2.json",
        "MODEL_PATH": os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507"),
        "RUNTIME_ENDPOINT": "http://127.0.0.1:30000",
        "TEMPERATURE": 0.1,
        "TOP_P": 0.5,
        "TOP_K": 15,
        "MAX_GENERATE_TOKENS": 16000,
        "MAX_INPUT_LENGTH": 50000,
        "SYSTEM_PROMPT": "",
        "MAX_CHUNKS_PER_BATCH": 20,
        "SEPARATOR_LENGTH": 60,
        "SHOW_PROGRESS": True,
    },
    "step7": {
        "OUTPUT_SUFFIX": "_L1Outline.json",
        "MODEL_PATH": os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507"),
        "RUNTIME_ENDPOINT": "http://127.0.0.1:30000",
        "TEMPERATURE": 0.1,
        "TOP_P": 0.5,
        "TOP_K": 15,
        "MAX_GENERATE_TOKENS": 16000,
        "MAX_INPUT_LENGTH": 50000,
        "SYSTEM_PROMPT": "",
        "SUMMARY_LANGUAGE": "en",
        "WORD_LIMIT_0_50": 50,
        "WORD_LIMIT_50_200": 100,
        "WORD_LIMIT_201_500": 300,
        "WORD_LIMIT_501_2000": 1000,
        "WORD_LIMIT_2001_4000": 2000,
        "WORD_LIMIT_4001_12000": 3000,
        "BATCH_SIZE_0_50": 16,
        "BATCH_SIZE_50_200": 16,
        "BATCH_SIZE_201_500": 16,
        "BATCH_SIZE_501_2000": 16,
        "BATCH_SIZE_2001_4000": 8,
        "BATCH_SIZE_4001_12000": 3,
        "OUTLINE_SEPARATOR": ".\n",
        "SEPARATOR_LENGTH": 60,
        "SHOW_PROGRESS": True,
    },
    "step8": {
        "OUTPUT_SUFFIX": "_split.json",
        "TARGET_LEN": 1000,
        "OVERLAP": 100,
        "BACKTRACK_RANGE": 150,
        "PUNCTUATIONS": ".!?",
        "SHOW_PROGRESS": True,
    },
}

# ================================================================================================
# ========================================== Code =================================================
# ================================================================================================


def log(message: str) -> None:
    if SHOW_PROGRESS:
        print(message, flush=True)


def format_seconds(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1_000)
    return f"{hours} hours {minutes} minutes {secs} seconds {milliseconds} milliseconds"


def collect_input_files() -> list[Path]:
    if RUN_MODE == "single":
        single = Path(SINGLE_INPUT_FILE)
        return [single] if single.is_file() else []

    input_dir = Path(INPUT_JSON_DIR)
    files: list[Path] = []
    for suffix in START_FILE_SUFFIXES:
        files.extend(p for p in input_dir.glob(f"*{suffix}") if p.is_file())
    return sorted(set(files), key=lambda p: p.name)


def load_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def set_module_params(module: ModuleType, params: dict[str, Any]) -> None:
    for key, value in params.items():
        setattr(module, key, value)


def remove_if_overwrite(path: Path) -> None:
    if OVERWRITE_OUTPUTS and path.exists():
        path.unlink()


def run_step(
    step_name: str,
    script_dir: Path,
    input_file: Path,
    output_file: Path,
    display_name: str,
) -> None:
    script_path = script_dir / SCRIPT_FILES[step_name]
    if not script_path.exists():
        raise FileNotFoundError(f"Step script not found: {script_path}")

    remove_if_overwrite(output_file)

    params = dict(CFG[step_name])
    params["INPUT_FILE"] = str(input_file)

    module = load_module(script_path, f"temp_{step_name}_{abs(hash((str(input_file), step_name)))}")
    set_module_params(module, params)

    log("\n" + "=" * 60)
    log(f"Executing: {display_name}")
    log(f"Script: {script_path.name}")
    log(f"Input: {input_file}")
    log(f"Expected output: {output_file}")
    log("=" * 60)

    module.main()

    if not output_file.exists():
        raise FileNotFoundError(f"Step completed but output file not found: {output_file}")


def validate_start_file(path: Path) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to read start JSON: {path}, error: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Start JSON is not object format: {path}")
    chunks = data.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise RuntimeError(f"Start JSON missing non-empty chunks: {path}")

    required_chunk_fields = {"length", "text"}
    missing_basic = [idx + 1 for idx, chunk in enumerate(chunks) if not required_chunk_fields <= set(chunk)]
    if missing_basic:
        raise RuntimeError(f"Start JSON chunks missing length/text fields: {path}, problematic chunks: {missing_basic[:5]}")


def process_file(script_dir: Path, start_file: Path) -> dict[str, Any]:
    validate_start_file(start_file)

    timings: list[dict[str, Any]] = []
    book_start = perf_counter()
    base_name = start_file.stem

    def record_step(name: str, func: Callable[[], Path]) -> Path:
        step_start = perf_counter()
        result = func()
        elapsed = perf_counter() - step_start
        timings.append({"name": name, "seconds": elapsed})
        log(f"[{base_name}] {name} time: {format_seconds(elapsed)}")
        return result

    j4 = start_file
    j5 = j4.parent / f"{j4.stem}{CFG['step5']['OUTPUT_SUFFIX']}"
    record_step(
        "Step 5 - English L2 outline",
        lambda: (
            run_step("step5", script_dir, j4, j5, f"{base_name} - Step 5"),
            j5,
        )[1],
    )

    j6 = j5.parent / f"{j5.stem}{CFG['step6']['OUTPUT_SUFFIX']}"
    record_step(
        "Step 6 - Merge H2 regions",
        lambda: (
            run_step("step6", script_dir, j5, j6, f"{base_name} - Step 6"),
            j6,
        )[1],
    )

    j7 = j6.parent / f"{j6.stem}{CFG['step7']['OUTPUT_SUFFIX']}"
    record_step(
        "Step 7 - English L1 outline",
        lambda: (
            run_step("step7", script_dir, j6, j7, f"{base_name} - Step 7"),
            j7,
        )[1],
    )

    final_json = j7.parent / f"{j7.stem}{CFG['step8']['OUTPUT_SUFFIX']}"
    record_step(
        "Step 8 - Final split",
        lambda: (
            run_step("step8", script_dir, j7, final_json, f"{base_name} - Step 8"),
            final_json,
        )[1],
    )

    total_seconds = perf_counter() - book_start
    log(f"\nComplete: {start_file.name}")
    log(f"Final output: {final_json}")
    log(f"Total time: {format_seconds(total_seconds)}")

    return {
        "book": start_file.name,
        "steps": timings,
        "total_seconds": total_seconds,
        "final_output": str(final_json),
    }


def write_time_report(report_path: Path, reports: list[dict[str, Any]], failed: list[dict[str, str]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    total_all_books = 0.0
    aggregate_steps: dict[str, float] = {}

    for report in reports:
        lines.append(f"Book: {report['book']}")
        for step in report["steps"]:
            step_name = str(step["name"])
            step_seconds = float(step["seconds"])
            lines.append(f"  {step_name}: {format_seconds(step_seconds)}")
            aggregate_steps[step_name] = aggregate_steps.get(step_name, 0.0) + step_seconds
        lines.append(f"  Book total time: {format_seconds(float(report['total_seconds']))}")
        lines.append(f"  Final output: {report['final_output']}")
        lines.append("")
        total_all_books += float(report["total_seconds"])

    lines.append(f"Books successfully processed: {len(reports)}")
    lines.append(f"Failed books: {len(failed)}")
    lines.append(f"Total time for successful books: {format_seconds(total_all_books)}")
    lines.append("")

    if aggregate_steps:
        lines.append("Successful books step summary:")
        for step_name, step_seconds in aggregate_steps.items():
            percent = (step_seconds / total_all_books * 100) if total_all_books else 0.0
            lines.append(f"  {step_name}: {format_seconds(step_seconds)} ({percent:.2f}%)")
        lines.append("")

    if failed:
        lines.append("Failure details:")
        for item in failed:
            lines.append(f"  {item['book']}: {item['error']}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).parent.absolute()
    report_path = Path(TIME_REPORT_FILE)
    files = collect_input_files()

    if not files:
        print("No start JSON files found to process")
        sys.exit(1)

    log("=" * 60)
    log("Temporary English summary re-run workflow: Step 5 -> Step 8")
    log(f"Input directory: {INPUT_JSON_DIR}")
    log(f"Start file suffixes: {START_FILE_SUFFIXES}")
    log(f"Files found: {len(files)}")
    log(f"Expected file count: {EXPECTED_FILE_COUNT}")
    log(f"Overwrite existing outputs: {OVERWRITE_OUTPUTS}")
    log("=" * 60)

    if EXPECTED_FILE_COUNT and len(files) != EXPECTED_FILE_COUNT:
        log(f"[Warning] Found {len(files)} files, expected {EXPECTED_FILE_COUNT}, will continue with found files.")

    reports: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    write_time_report(report_path, reports, failed)

    for index, file_path in enumerate(files, start=1):
        log("\n" + "#" * 60)
        log(f"Processing progress: {index}/{len(files)}")
        log(f"Start file: {file_path}")
        log("#" * 60)
        try:
            reports.append(process_file(script_dir, file_path))
        except Exception as exc:
            failed.append({"book": file_path.name, "error": str(exc)})
            log(f"[Failed] {file_path.name}: {exc}")
        finally:
            write_time_report(report_path, reports, failed)
            log(f"Time report updated: {report_path}")

    if failed:
        log("\nSome files failed, check the time report for failure details.")
        sys.exit(1)

    log("\nAll processing complete.")


if __name__ == "__main__":
    main()
