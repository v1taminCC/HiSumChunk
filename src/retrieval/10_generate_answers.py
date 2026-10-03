#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from utils_simple_api import chat_batch, MODEL_GENERATE, DEFAULT_WORKERS
from utils_io import ensure_dir, read_json, write_json
from utils_progress import ProgressTracker
from config import SUPPORTED_SCHEMES

RUNS_IN_BOOK_DIR = BASE_DIR / "runs" / "in_book"
RUNS_GLOBAL_DIR = BASE_DIR / "runs" / "global"
CHUNKS_BY_BOOK_DIR = BASE_DIR / "data" / "chunks" / "by_book"
QA_DRAFTS_ALL = BASE_DIR / "qa_build" / "data" / "qa_drafts" / "qa_drafts_all.json"
PROMPT_DIR = BASE_DIR / "prompts"
ANSWERS_DIR = BASE_DIR / "data" / "answers"
ANSWERS_BY_BOOK_DIR = ANSWERS_DIR / "by_book"

DEFAULT_TOP_K = 5


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_qa_drafts() -> dict[str, dict]:
    rows = read_json(QA_DRAFTS_ALL)
    by_qa_id: dict[str, dict] = {}
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


def load_retrieval_results_for_book(mode: str, scheme: str, book_title: str) -> list[dict]:
    if mode == "in_book":
        fp = RUNS_IN_BOOK_DIR / f"{book_title}.{scheme}.top10.json"
    else:
        fp = RUNS_GLOBAL_DIR / f"all_books.{scheme}.top10.json"
    if not fp.exists():
        return []
    all_rows = read_json(fp)
    if mode == "global":
        all_rows = [r for r in all_rows if r.get("book_title") == book_title]
    return all_rows


def load_chunks_for_book(scheme: str, book_title: str) -> dict[str, str]:
    fp = CHUNKS_BY_BOOK_DIR / f"{book_title}.{scheme}.json"
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
            chunks_info.append({"chunk_id": cid, "text": ctext, "rank": item["rank"], "score": item["score"]})
            chunks_text_parts.append(f"[Chunk {item['rank']}] {ctext}")
    return chunks_info, "\n\n".join(chunks_text_parts)


def generate_answers_for_book(
    book_title: str,
    mode: str,
    scheme: str,
    top_k: int,
    qa_map: dict[str, dict],
    tracker: ProgressTracker,
    existing_ids: set[str],
    workers: int,
) -> list[dict]:
    run_rows = load_retrieval_results_for_book(mode, scheme, book_title)
    if not run_rows:
        return []

    chunk_texts = load_chunks_for_book(scheme, book_title)
    pending = [r for r in run_rows if r["query_id"] not in existing_ids]

    output_rows: list[dict] = []
    total = len(run_rows)
    tracker.start_scheme(scheme, total, skipped=len(existing_ids))

    if not pending:
        tracker.finish_scheme()
        return output_rows

    system_prompt = load_prompt("answer_generation_system")
    user_template = load_prompt("answer_generation_user")

    calls = []
    meta_by_idx: dict[int, dict] = {}
    for idx, run_row in enumerate(pending):
        query_id = run_row["query_id"]
        qa_info = qa_map.get(query_id)
        if qa_info is None:
            meta_by_idx[idx] = {
                "query_id": query_id, "question": "", "gt_answer": "", "qt": "", "diff": "",
                "chunks_info": [], "generated_answer": "", "error": "qa_not_found",
            }
            calls.append({"system": system_prompt, "user": ""})
            continue

        question = qa_info["question"]
        top_chunks = run_row.get("top_k", [])
        chunks_info, chunks_text = make_chunks_text(top_chunks, chunk_texts, top_k)

        if not chunks_text:
            meta_by_idx[idx] = {
                "query_id": query_id, "question": question,
                "gt_answer": qa_info["ground_truth_answer"],
                "qt": qa_info["question_type"], "diff": qa_info["difficulty"],
                "chunks_info": chunks_info, "generated_answer": "", "error": "no_chunks",
            }
            calls.append({"system": system_prompt, "user": ""})
            continue

        user_prompt = user_template.replace("{question}", question).replace("{chunks_text}", chunks_text)
        calls.append({"system": system_prompt, "user": user_prompt})
        meta_by_idx[idx] = {
            "query_id": query_id, "question": question,
            "gt_answer": qa_info["ground_truth_answer"],
            "qt": qa_info["question_type"], "diff": qa_info["difficulty"],
            "chunks_info": chunks_info, "generated_answer": "", "error": "",
        }

    def on_result(idx: int, result: dict) -> None:
        meta = meta_by_idx[idx]
        if result["ok"]:
            meta["generated_answer"] = result["content"].strip()
            meta["error"] = ""
            tracker.update(ok=True)
        else:
            meta["generated_answer"] = ""
            if not meta["error"]:
                meta["error"] = result["error"]
            tracker.update(ok=False)

    results = chat_batch(calls, model=MODEL_GENERATE, temperature=0.2, workers=workers, on_result=on_result)

    for idx, meta in meta_by_idx.items():
        if meta["generated_answer"] == "" and meta["error"] == "" and idx < len(results):
            if results[idx]["ok"]:
                meta["generated_answer"] = results[idx]["content"].strip()
            else:
                meta["error"] = results[idx]["error"]

    for idx in sorted(meta_by_idx.keys()):
        meta = meta_by_idx[idx]
        output_rows.append({
            "query_id": meta["query_id"],
            "book_title": book_title,
            "question": meta["question"],
            "ground_truth_answer": meta["gt_answer"],
            "question_type": meta["qt"],
            "difficulty": meta["diff"],
            "context_chunks": meta["chunks_info"],
            "generated_answer": meta["generated_answer"],
            "generation_model": MODEL_GENERATE,
            "scheme": scheme,
            "error": meta["error"],
        })

    tracker.finish_scheme()
    return output_rows


def list_books(mode: str) -> list[str]:
    if mode == "in_book":
        names = set()
        for p in RUNS_IN_BOOK_DIR.glob("*.top10.json"):
            if p.is_file():
                # Extract book name: everything before the last ".{scheme}.top10.json"
                name = p.name
                for s in SUPPORTED_SCHEMES:
                    suffix = f".{s}.top10.json"
                    if name.endswith(suffix):
                        names.add(name[:-len(suffix)])
                        break
        return sorted(names)
    else:
        fp = RUNS_GLOBAL_DIR / "all_books.ours.top10.json"
        if fp.exists():
            return sorted({r.get("book_title", "") for r in read_json(fp) if r.get("book_title")})
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers from retrieved chunks (concurrent, per-book)")
    parser.add_argument("--mode", type=str, default="in_book", choices=["in_book", "global"])
    parser.add_argument("--scheme", type=str, default="all", 
                        choices=["all"] + [s for s in SUPPORTED_SCHEMES if s not in ("booksum_split",)],
                        help="Split scheme; all=all available schemes")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--book", type=str, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Concurrent threads (default {DEFAULT_WORKERS})")
    args = parser.parse_args()

    ensure_dir(ANSWERS_DIR)
    ensure_dir(ANSWERS_BY_BOOK_DIR)

    books = [args.book] if args.book else list_books(args.mode)
    if args.scheme == "all":
        schemes = [s for s in SUPPORTED_SCHEMES if s != "booksum_split"]
    else:
        schemes = [args.scheme]

    print(f"Pending: {len(books)} books, mode={args.mode}, workers={args.workers}, top_k={args.top_k}")

    qa_map = load_qa_drafts()
    tracker = ProgressTracker(len(books))

    all_output: dict[str, list[dict]] = {s: [] for s in schemes}
    global_start = time.time()
    total_queries = 0

    for book_idx, book_title in enumerate(books, start=1):
        tracker.start_book(book_idx, book_title)
        for scheme in schemes:
            out_path = ANSWERS_BY_BOOK_DIR / f"{book_title}.{args.mode}.{scheme}.top{args.top_k}.json"
            if out_path.exists():
                existing = read_json(out_path)
                existing_ids = {r["query_id"] for r in existing}
            else:
                existing = []
                existing_ids = set()

            new_rows = generate_answers_for_book(
                book_title, args.mode, scheme, args.top_k, qa_map, tracker, existing_ids, args.workers,
            )
            existing.extend(new_rows)
            write_json(out_path, existing)
            all_output[scheme].extend(existing)
            total_queries += len(existing)

        tracker.finish_book()

    for scheme in schemes:
        out_all = ANSWERS_DIR / f"answers_all.{args.mode}.{scheme}.top{args.top_k}.json"
        write_json(out_all, all_output[scheme])

    print(tracker.total_summary(len(books), total_queries, global_start))


if __name__ == "__main__":
    main()
