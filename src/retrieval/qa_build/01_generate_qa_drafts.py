#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

from config import DEFAULT_QA_PER_CHAPTER, PROMPT_DIR, QA_DRAFT_BY_BOOK_DIR, QA_DRAFT_DIR, SOURCE_CHAPTER_DIR
from utils_io import ensure_dir, read_json, write_json
from utils_prompt import load_prompt, render_prompt
from utils_qwen_api import qwen_chat_batch, API_MODEL, DEFAULT_WORKERS

SAMPLING_STATE_FILE = QA_DRAFT_DIR / "qa_sampling_state.json"
CURRENT_SAMPLE_FILE = QA_DRAFT_DIR / "current_sampled_chapters.json"

QA_RANDOM_WINDOW_WORDS = 1024
QA_RETRY_SLEEP_INCOMPLETE = 3.0
REQUIRED_QA_FIELDS = ("question", "answer", "question_type", "difficulty")

# === Progress ===


def _term_width() -> int:
    try:
        import shutil
        return shutil.get_terminal_size().columns
    except Exception:
        return 120


def _elapsed_str(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _bar(filled: int, total: int, width: int = 20) -> str:
    f = int(width * filled / max(total, 1))
    return "\u2588" * f + "\u2591" * (width - f)


def _book_bar(book_idx: int, total_books: int) -> str:
    return _bar(book_idx, total_books, 12)


def _print_progress_line(done: int, total: int, label: str, status: str, elapsed: float) -> None:
    pct = done / max(total, 1) * 100
    bar = _bar(done, total)
    ela = _elapsed_str(elapsed)
    eta = _elapsed_str(elapsed / done * (total - done)) if 0 < done < total else "..."
    line = f"  {label:<12} {bar} {done:>4}/{total:<4} ({pct:5.1f}%)  {status}  elapsed {ela}"
    if 0 < done < total:
        line += f"  remaining {eta}"
    print(f"\r{line:<{_term_width()}}", end="", flush=True)
    sys.stdout.flush()


# === QA validation ===


def is_complete_qa(qa: dict) -> bool:
    for field in REQUIRED_QA_FIELDS:
        if not str(qa.get(field, "") or "").strip():
            return False
    evidence_units = qa.get("evidence_units", []) or []
    if not isinstance(evidence_units, list) or not evidence_units:
        return False
    for eu in evidence_units:
        if not str(eu.get("evidence_text", "") or "").strip():
            return False
        if not str(eu.get("reason", "") or "").strip():
            return False
    return True


def extract_qa_index(qa_id: str) -> int:
    match = re.search(r"\.qa_(\d+)$", str(qa_id or ""))
    return int(match.group(1)) if match else 0


# === Data loading ===


def normalize_existing_book_rows(rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for row in rows:
        row = deepcopy(row)
        row["qas"] = [qa for qa in row.get("qas", []) or [] if is_complete_qa(qa) and not qa.get("parse_failed", False)]
        normalized.append(row)
    return normalized


def load_existing_book_rows(book_title: str) -> list[dict]:
    path = QA_DRAFT_BY_BOOK_DIR / f"{book_title}.json"
    if not path.exists():
        return []
    rows = normalize_existing_book_rows(read_json(path))
    write_json(path, rows)
    return rows


def build_existing_index(rows: list[dict]) -> dict[str, dict]:
    return {str(row.get("chapter_id", "") or ""): row for row in rows}


def save_book_rows(book_title: str, rows: list[dict]) -> None:
    write_json(QA_DRAFT_BY_BOOK_DIR / f"{book_title}.json", rows)


def sync_all_drafts() -> None:
    all_rows: list[dict] = []
    files = sorted(p for p in QA_DRAFT_BY_BOOK_DIR.glob("*.json") if p.is_file() and not p.name.endswith(".bak"))
    for fp in files:
        all_rows.extend(read_json(fp))
    write_json(QA_DRAFT_DIR / "qa_drafts_all.json", all_rows)


def build_empty_draft(row: dict, qa_per_chapter: int) -> dict:
    chapter_id = row["chapter_id"]
    return {
        "draft_id": f"{chapter_id}.draft",
        "book_title": row["book_title"],
        "chapter_id": chapter_id,
        "chapter_text": row["text"],
        "text_source": row.get("text_source", "booksum_summary_text"),
        "summary_id": row.get("summary_id", ""),
        "qa_per_chapter_requested": qa_per_chapter,
        "generation_status": "in_progress",
        "generation_error": "",
        "generation_model": API_MODEL,
        "qas": [],
    }


def build_one_qa_for_storage(draft_id: str, result: dict, qa_index: int) -> dict | None:
    parsed = result.get("parsed") or {}
    qas = parsed.get("qas") or []
    if not qas:
        return None
    for qa in qas:
        normalized = {
            "qa_id": f"{draft_id}.qa_{qa_index}",
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "question_type": qa.get("question_type", ""),
            "difficulty": qa.get("difficulty", ""),
            "evidence_units": qa.get("evidence_units", []),
            "parse_failed": False,
            "qa_source": "booksum_summary_text",
            "raw_model_output": "",
        }
        if is_complete_qa(normalized):
            return normalized
    return None


def sample_random_word_window(text: str, window_words: int = QA_RANDOM_WINDOW_WORDS) -> tuple[str, int, int, int]:
    word_matches = list(re.finditer(r"\S+", text))
    total_words = len(word_matches)
    if total_words <= window_words:
        return text, 0, total_words, total_words
    start_word = random.randint(0, total_words - window_words)
    end_word = start_word + window_words
    start_char = word_matches[start_word].start()
    end_char = word_matches[end_word - 1].end()
    return text[start_char:end_char], start_word, end_word, total_words


# === Core generation ===


def process_book(
    book_title: str,
    source_rows: list[dict],
    system_prompt: str,
    user_template: str,
    qa_per_chapter: int,
    workers: int,
    book_idx: int,
    total_books: int,
) -> bool:
    book_rows = load_existing_book_rows(book_title)
    by_chapter = build_existing_index(book_rows)
    total_chapters = len(source_rows)

    lbl = f"\u25b8 {book_title[:30]}"
    start_time = time.time()

    print(f"\n{'─'*_term_width()}", flush=True)
    print(f"{_book_bar(book_idx, total_books)} {lbl}  book [{book_idx}/{total_books}]  {total_chapters} chapters  workers={workers}", flush=True)

    # 1) Create draft or skip completed
    for source_row in source_rows:
        chapter_id = source_row["chapter_id"]
        draft = by_chapter.get(chapter_id)
        if draft is None:
            draft = build_empty_draft(source_row, qa_per_chapter)
            book_rows.append(draft)
            by_chapter[chapter_id] = draft
            save_book_rows(book_title, book_rows)
            sync_all_drafts()
        draft["qa_per_chapter_requested"] = qa_per_chapter
        draft["generation_model"] = API_MODEL

    # 2) Collect chapters needing QA generation this round
    pending: list[dict] = []
    for source_row in source_rows:
        draft = by_chapter[source_row["chapter_id"]]
        if len(draft.get("qas", []) or []) >= qa_per_chapter:
            draft["generation_status"] = "ok"
            continue
        pending.append(source_row)

    # 3) Batch concurrent API calls
    batch_size = workers
    src_ids = {r["chapter_id"] for r in source_rows}
    completed = set()
    processed = 0
    for r in book_rows:
        if r.get("generation_status") == "ok" and r["chapter_id"] in src_ids:
            completed.add(r["chapter_id"])
    processed = len(completed)

    while pending:
        # Take a batch
        batch = pending[:batch_size]
        pending = pending[batch_size:]

        # Build API calls for this batch
        calls = []
        for source_row in batch:
            sampled_text, _, _, _ = sample_random_word_window(source_row["text"])
            user_prompt = render_prompt(
                user_template,
                book_title=source_row["book_title"],
                chapter_id=source_row["chapter_id"],
                chapter_text=sampled_text,
                qa_per_chapter=1,
            )
            calls.append({"system": system_prompt, "user": user_prompt})

        results = qwen_chat_batch(calls, workers=min(workers, len(calls)))

        # Process results
        for i, source_row in enumerate(batch):
            draft = by_chapter[source_row["chapter_id"]]
            result = results[i]

            if not result.get("ok"):
                error_msg = str(result.get("error", "") or "")
                if ("data_inspection_failed" in error_msg or "inappropriate content" in error_msg.lower()
                        or "400 Client Error" in error_msg):
                    draft["generation_status"] = "in_progress"
                    draft["generation_error"] = error_msg
                    save_book_rows(book_title, book_rows)
                    sync_all_drafts()
                    continue  # No more retries for this chapter

                # API failed, put back to pending
                pending.append(source_row)
                time.sleep(1.0)
                continue

            next_idx = max([extract_qa_index(qa.get("qa_id", "")) for qa in (draft.get("qas", []) or [])] + [0]) + 1
            qa = build_one_qa_for_storage(draft["draft_id"], result, next_idx)
            if qa is None:
                # QA incomplete, put back to pending for retry
                pending.append(source_row)
                time.sleep(QA_RETRY_SLEEP_INCOMPLETE)
                continue

            qas = draft.get("qas", []) or []
            qas.append(qa)
            draft["qas"] = qas
            draft["generation_status"] = "ok" if len(qas) >= qa_per_chapter else "in_progress"
            draft["generation_error"] = ""
            draft["generation_model"] = API_MODEL
            if draft["generation_status"] == "ok":
                mark_chapter_generated(book_title, source_row["chapter_id"])

            save_book_rows(book_title, book_rows)
            sync_all_drafts()
            if source_row["chapter_id"] not in completed:
                completed.add(source_row["chapter_id"])
                processed += 1

        elapsed = time.time() - start_time
        _print_progress_line(processed, total_chapters, lbl, f"\u2713{processed}", elapsed)

    save_book_rows(book_title, book_rows)
    sync_all_drafts()
    elapsed = time.time() - start_time
    ok_count = sum(1 for r in book_rows if r.get("generation_status") == "ok")
    _print_progress_line(total_chapters, total_chapters, lbl, f"\u2713{ok_count}", elapsed)
    print(flush=True)
    return True


def mark_chapter_generated(book_title: str, chapter_id: str) -> None:
    """Update sampling state, mark chapter as QA generated"""
    if not SAMPLING_STATE_FILE.exists():
        return
    state = read_json(SAMPLING_STATE_FILE)
    gen_ids = set(state.get("generated_ids", []))
    gen_ids.add(chapter_id)
    state["generated_ids"] = sorted(gen_ids)
    write_json(SAMPLING_STATE_FILE, state)


def load_sampled_chapters_only(rows: list[dict], qa_per_chapter: int) -> list[dict]:
    """Load only chapters from sampling file that have not yet generated QA"""
    if not CURRENT_SAMPLE_FILE.exists():
        raise FileNotFoundError(f"Sampling file not found: {CURRENT_SAMPLE_FILE}. Please run 00b_sample_chapters.py first")
    sampled = read_json(CURRENT_SAMPLE_FILE)
    sampled_ids = {ch["chapter_id"] for ch in sampled}

    gen_ids = set()
    if SAMPLING_STATE_FILE.exists():
        state = read_json(SAMPLING_STATE_FILE)
        gen_ids = set(state.get("generated_ids", []))
    pending_ids = sampled_ids - gen_ids

    result = []
    for row in rows:
        if row["chapter_id"] in pending_ids:
            result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA drafts via a chat LLM API, with checkpoint resume")
    parser.add_argument("--book", type=str, default=None, help="Process only specified book")
    parser.add_argument("--count", type=int, default=None, help="Randomly sample N chapters (requires 00b_sample_chapters.py)")
    parser.add_argument("--qa_per_chapter", type=int, default=DEFAULT_QA_PER_CHAPTER)
    parser.add_argument("--workers", type=int, default=5, help="Concurrent chapters (default 5)")
    args = parser.parse_args()

    ensure_dir(QA_DRAFT_DIR)
    ensure_dir(QA_DRAFT_BY_BOOK_DIR)

    use_sampled = args.count is not None
    if use_sampled and args.book:
        print("ERROR: --count and --book cannot be used together", flush=True)
        return

    rows = read_json(SOURCE_CHAPTER_DIR / "source_chapters_all.json")
    if args.book:
        rows = [row for row in rows if row["book_title"] == args.book]
    elif use_sampled:
        rows = load_sampled_chapters_only(rows, args.qa_per_chapter)

    if not rows:
        print("No chapters to process. All sampled chapters may be completed.", flush=True)
        return

    system_prompt = load_prompt("qa_generation_system.txt", PROMPT_DIR)
    user_template = load_prompt("qa_generation_user.txt", PROMPT_DIR)

    by_book: dict[str, list[dict]] = {}
    for row in rows:
        by_book.setdefault(row["book_title"], []).append(row)

    total_books = len(by_book)
    print(f"Pending: {total_books} books, {args.qa_per_chapter} QA per chapter, workers={args.workers}")
    print(f"Model: {API_MODEL}, API: configured via API_BASE_URL env var", flush=True)

    for book_idx, (book_title, book_rows) in enumerate(by_book.items(), start=1):
        ok = process_book(
            book_title, book_rows, system_prompt, user_template,
            args.qa_per_chapter, args.workers, book_idx, total_books,
        )
        if not ok:
            print("Processing stopped", flush=True)
            return

    print(f"\n{'═'*_term_width()}\nAll done. {total_books} books.\n{'═'*_term_width()}", flush=True)


if __name__ == "__main__":
    main()
