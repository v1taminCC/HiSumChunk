#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate answers from retrieval results (via the API model).
Supports per-QA checkpoint resume.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    ANSWER_TOP_K,
    ANSWERS_BY_BOOK_DIR,
    CHUNKS_BY_BOOK_DIR,
    DEFAULT_TOP_K,
    PROMPTS_DIR,
    QA_DRAFTS_DIR,
    RUNS_IN_BOOK_DIR,
    VARIANTS,
)
from utils_io import ensure_dir, read_json, write_json
from utils_simple_api import MODEL_GENERATE, chat_batch, DEFAULT_WORKERS


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_qa_drafts() -> dict[str, dict]:
    qa_path = QA_DRAFTS_DIR / "qa_drafts_all.json"
    if not qa_path.exists():
        return {}
    rows = read_json(qa_path)
    by_qa_id = {}
    for row in rows:
        for qa in row.get("qas", []) or []:
            qa_id = qa.get("qa_id", "")
            if qa_id:
                by_qa_id[qa_id] = {
                    "question": qa.get("question", ""),
                    "ground_truth_answer": qa.get("answer", ""),
                    "question_type": qa.get("question_type", ""),
                    "difficulty": qa.get("difficulty", ""),
                    "book_title": row.get("book_title", ""),
                    "chapter_id": row.get("chapter_id", ""),
                }
    return by_qa_id


def load_retrieval_results(book: str, variant: str) -> list[dict]:
    fp = RUNS_IN_BOOK_DIR / f"{book}.{variant}.top{DEFAULT_TOP_K}.json"
    if not fp.exists():
        return []
    return read_json(fp)


def load_chunks(book: str, variant: str) -> dict[str, str]:
    fp = CHUNKS_BY_BOOK_DIR / f"{book}.{variant}.json"
    if not fp.exists():
        return {}
    chunks = read_json(fp)
    return {c["chunk_id"]: c.get("text", "") for c in chunks}


def make_chunks_text(top_chunks: list[dict], chunk_texts: dict[str, str], top_k: int) -> tuple[list[dict], str]:
    chunks_info = []
    chunks_text_parts = []
    for item in top_chunks[:top_k]:
        cid = item["chunk_id"]
        ctext = chunk_texts.get(cid, "")
        if ctext:
            chunks_info.append({
                "chunk_id": cid,
                "text": ctext,
                "rank": item["rank"],
                "score": item["score"],
            })
            chunks_text_parts.append(f"[Chunk {item['rank']}] {ctext}")
    return chunks_info, "\n\n".join(chunks_text_parts)


def list_book_titles() -> list[str]:
    books = set()
    for fp in RUNS_IN_BOOK_DIR.glob(f"*.{VARIANTS[0]}.top{DEFAULT_TOP_K}.json"):
        name = fp.stem
        suffix = f".{VARIANTS[0]}.top{DEFAULT_TOP_K}"
        if name.endswith(suffix):
            books.add(name[: -len(suffix)])
    return sorted(books)


def load_existing_answers(out_path: Path) -> tuple[list[dict], set[str]]:
    if not out_path.exists():
        return [], set()
    existing = read_json(out_path)
    done_ids = {r["query_id"] for r in existing if r.get("generated_answer") or r.get("error")}
    return existing, done_ids


def generate_answers_for_book(
    book: str,
    variant: str,
    qa_map: dict[str, dict],
    workers: int,
    out_path: Path,
) -> tuple[int, int, int]:
    """Return (newly_generated, skipped, failed)"""
    existing, done_ids = load_existing_answers(out_path)

    run_rows = load_retrieval_results(book, variant)
    if not run_rows:
        return 0, 0, 0

    pending_rows = [r for r in run_rows if r["query_id"] not in done_ids]
    if not pending_rows:
        return 0, len(done_ids), 0

    chunk_texts = load_chunks(book, variant)
    system_prompt = load_prompt("answer_generation_system")
    user_template = load_prompt("answer_generation_user")

    calls = []
    call_to_meta_idx = []
    metas = []

    for run_row in pending_rows:
        query_id = run_row["query_id"]
        qa_info = qa_map.get(query_id)

        if qa_info is None:
            metas.append({
                "query_id": query_id,
                "question": "",
                "gt_answer": "",
                "qt": "",
                "diff": "",
                "chunks_info": [],
                "generated_answer": "",
                "error": "qa_not_found",
            })
            continue

        question = qa_info["question"]
        top_chunks = run_row.get("top_k", [])
        chunks_info, chunks_text = make_chunks_text(top_chunks, chunk_texts, ANSWER_TOP_K)

        if not chunks_text:
            metas.append({
                "query_id": query_id,
                "question": question,
                "gt_answer": qa_info["ground_truth_answer"],
                "qt": qa_info["question_type"],
                "diff": qa_info["difficulty"],
                "chunks_info": chunks_info,
                "generated_answer": "",
                "error": "no_chunks",
            })
            continue

        user_prompt = user_template.replace("{question}", question).replace("{chunks_text}", chunks_text)
        calls.append({"system": system_prompt, "user": user_prompt})
        call_to_meta_idx.append(len(metas))
        metas.append({
            "query_id": query_id,
            "question": question,
            "gt_answer": qa_info["ground_truth_answer"],
            "qt": qa_info["question_type"],
            "diff": qa_info["difficulty"],
            "chunks_info": chunks_info,
            "generated_answer": "",
            "error": "",
        })

    ok_count = [0]
    fail_count = [0]

    def on_result(idx: int, result: dict):
        real_idx = call_to_meta_idx[idx]
        if result["ok"]:
            metas[real_idx]["generated_answer"] = result["content"].strip()
            metas[real_idx]["error"] = ""
            ok_count[0] += 1
        else:
            metas[real_idx]["generated_answer"] = ""
            metas[real_idx]["error"] = result["error"]
            fail_count[0] += 1

    if calls:
        chat_batch(calls, model=MODEL_GENERATE, temperature=0.2, workers=workers, on_result=on_result)

    new_rows = []
    for meta in metas:
        new_rows.append({
            "query_id": meta["query_id"],
            "book_title": book,
            "question": meta["question"],
            "ground_truth_answer": meta["gt_answer"],
            "question_type": meta["qt"],
            "difficulty": meta["diff"],
            "context_chunks": meta["chunks_info"],
            "generated_answer": meta["generated_answer"],
            "generation_model": MODEL_GENERATE,
            "variant": variant,
            "error": meta["error"],
        })

    all_rows = existing + new_rows
    write_json(out_path, all_rows)

    return ok_count[0], len(done_ids), fail_count[0]


def main():
    print("=" * 60)
    print("Ablation Experiment - Step 4: Generate answers (per-QA checkpoint resume)")
    print("=" * 60)

    ensure_dir(ANSWERS_BY_BOOK_DIR)
    books = list_book_titles()
    qa_map = load_qa_drafts()

    print(f"Books: {len(books)}")
    print(f"Variants: {VARIANTS}")
    print(f"QA count: {len(qa_map)}")
    print(f"Model: {MODEL_GENERATE}")
    print(f"workers: {DEFAULT_WORKERS}")

    total_new = 0
    total_skipped = 0
    total_failed = 0
    t0 = time.time()

    for variant in VARIANTS:
        print(f"\n--- Variant: {variant} ---")

        for bi, book in enumerate(books, 1):
            out_path = ANSWERS_BY_BOOK_DIR / f"{book}.in_book.{variant}.top{ANSWER_TOP_K}.json"
            ok, skipped, failed = generate_answers_for_book(book, variant, qa_map, DEFAULT_WORKERS, out_path)
            total_new += ok
            total_skipped += skipped
            total_failed += failed

            if ok == 0 and skipped > 0 and failed == 0:
                print(f"  [{bi}/{len(books)}] {book} - all done ({skipped} records)")
            elif ok > 0 or failed > 0:
                print(f"  [{bi}/{len(books)}] {book} - new✓{ok} skip{skipped} fail✗{failed}")
            else:
                print(f"  [{bi}/{len(books)}] {book} - no retrieval results")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Answer generation complete")
    print(f"  New: {total_new}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Failed: {total_failed}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
