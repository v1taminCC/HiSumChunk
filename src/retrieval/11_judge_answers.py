#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from utils_simple_api import chat_batch, MODEL_JUDGE, DEFAULT_WORKERS
from utils_io import ensure_dir, read_json, write_json
from utils_progress import ProgressTracker

ANSWERS_DIR = BASE_DIR / "data" / "answers"
ANSWERS_BY_BOOK_DIR = ANSWERS_DIR / "by_book"
QA_DRAFTS_ALL = BASE_DIR / "qa_build" / "data" / "qa_drafts" / "qa_drafts_all.json"
PROMPT_DIR = BASE_DIR / "prompts"
JUDGMENTS_DIR = BASE_DIR / "results" / "answer_judgments"
JUDGMENTS_BY_BOOK_DIR = JUDGMENTS_DIR / "by_book"

DEFAULT_TOP_K = 5
JUDGE_TEMPERATURE = 0.0

JUDGE_DIMENSIONS = ["grounding", "correctness", "hallucination_absence"]


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()


def load_qa_evidence() -> dict[str, str]:
    """Load evidence_text for each query from QA drafts for judge reference"""
    if not QA_DRAFTS_ALL.exists():
        return {}
    rows = read_json(QA_DRAFTS_ALL)
    evidence_map = {}
    for row in rows:
        for qa in row.get("qas", []) or []:
            qid = qa.get("qa_id", "")
            if not qid:
                continue
            units = qa.get("evidence_units", []) or []
            texts = []
            for u in units:
                t = str(u.get("evidence_text", "")).strip()
                if t:
                    texts.append(f"• {t}")
            if texts:
                evidence_map[qid] = "\n".join(texts)
    return evidence_map


def load_answers_for_book(book_title: str, mode: str, scheme: str, top_k: int) -> dict[str, dict]:
    fp = ANSWERS_BY_BOOK_DIR / f"{book_title}.{mode}.{scheme}.top{top_k}.json"
    if not fp.exists():
        return {}
    rows = read_json(fp)
    return {r["query_id"]: r for r in rows}


def format_chunks_text(chunks: list[dict], max_chars: int | None = None) -> str:
    parts = []
    for c in chunks:
        text = c['text']
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + f"... (truncated, original {len(c['text'])} chars)"
        parts.append(f"--- Chunk (rank={c['rank']}, score={c['score']:.4f}) ---\n{text}")
    return "\n\n".join(parts)


def build_judge_call(
    question: str,
    ground_truth: str,
    evidence: str,
    generated_answer: str,
    chunks_text: str,
) -> dict:
    system_prompt = load_prompt("judge_system")
    user_template = load_prompt("judge_user")
    user_prompt = (
        user_template.replace("{question}", question)
        .replace("{ground_truth}", ground_truth)
        .replace("{evidence}", evidence)
        .replace("{chunks}", chunks_text)
        .replace("{answer}", generated_answer)
    )
    return {"system": system_prompt, "user": user_prompt}


def parse_judge_result(raw_result: dict) -> dict | None:
    if not raw_result.get("ok"):
        return None
    content = raw_result["content"]
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        parsed = json.loads(cleaned)
        scores = {}
        for dim in JUDGE_DIMENSIONS:
            scores[dim] = int(parsed.get(dim, 0))
            scores[f"{dim}_reason"] = str(parsed.get(f"{dim}_reason", ""))
        return scores
    except Exception:
        return None


def make_empty_scores(reason: str) -> dict:
    scores = {}
    for dim in JUDGE_DIMENSIONS:
        scores[dim] = 0
        scores[f"{dim}_reason"] = reason
    return scores


def output_path_for_book(book_title: str, mode: str, top_k: int) -> Path:
    return JUDGMENTS_BY_BOOK_DIR / f"{book_title}.eval.json"


def compact_judgment(row: dict) -> dict:
    return {
        "query_id": row.get("query_id", ""),
        "book_title": row.get("book_title", ""),
        "scheme": row.get("scheme", ""),
        "question": row.get("question", ""),
        "ground_truth_answer": row.get("ground_truth_answer", ""),
        "question_type": row.get("question_type", ""),
        "difficulty": row.get("difficulty", ""),
        "grounding": int(row.get("grounding", 0) or 0),
        "correctness": int(row.get("correctness", 0) or 0),
        "hallucination_absence": int(row.get("hallucination_absence", 0) or 0),
        "context_chunks": row.get("context_chunks", []),
        "generated_answer": row.get("generated_answer", ""),
        "judge_model": MODEL_JUDGE,
    }


def compact_judgments(rows: list[dict]) -> list[dict]:
    return [compact_judgment(row) for row in rows]


def list_books_from_answers(mode: str, top_k: int) -> list[str]:
    suffix = f".{mode}.ours.top{top_k}.json"
    names = set()
    for p in ANSWERS_BY_BOOK_DIR.glob(f"*{suffix}"):
        if p.is_file():
            book_title = p.name.replace(suffix, "")
            if book_title:
                names.add(book_title)
    return sorted(names)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge independent scoring of each scheme's answers")
    parser.add_argument("--mode", type=str, default="in_book", choices=["in_book", "global"])
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--book", type=str, default=None)
    parser.add_argument("--scheme", type=str, default=None, help="Judge only specified scheme; judge all if not specified")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Concurrent threads (default {DEFAULT_WORKERS})")
    parser.add_argument("--truncate_chunks", type=str, default="retry", choices=["retry", "always", "never", "skip_failed_first"],
                        help="retry=full first, truncate on 400 | always=truncate directly | never=no truncation | skip_failed_first=skip failures first round, retry truncated at end")
    args = parser.parse_args()

    ensure_dir(JUDGMENTS_DIR)
    ensure_dir(JUDGMENTS_BY_BOOK_DIR)

    # Discover all available (book, scheme) combinations
    books = set()
    schemes_available = set()
    if args.book:
        books = {args.book}
    for fp in ANSWERS_BY_BOOK_DIR.glob(f"*.in_book.*.top{args.top_k}.json"):
        if not fp.is_file():
            continue
        # filename: {book}.{mode}.{scheme}.top{k}.json
        name = fp.name
        # Extract scheme: between .in_book. and .top{k}.json
        mode_suffix = f".{args.mode}."
        top_suffix = f".top{args.top_k}.json"
        if mode_suffix in name and top_suffix in name:
            middle = name.split(mode_suffix, 1)[1].rsplit(top_suffix, 1)[0]
            schemes_available.add(middle)
            book_title = name.rsplit(mode_suffix + middle + top_suffix, 1)[0]
            if not args.book:
                books.add(book_title)

    if args.scheme:
        schemes_available = {args.scheme}

    books = sorted(books)

    print(f"Pending: {len(books)} books, {len(schemes_available)} schemes, mode={args.mode}, workers={args.workers}")
    print(f"Schemes: {schemes_available}")

    evidence_map = load_qa_evidence()
    print(f"Loaded {len(evidence_map)} QA evidence entries", flush=True)

    tracker = ProgressTracker(len(books))
    all_judgments: list[dict] = []
    global_start = time.time()
    total_judged = 0

    for book_idx, book_title in enumerate(books, start=1):
        tracker.start_book(book_idx, book_title)

        out_path = output_path_for_book(book_title, args.mode, args.top_k)
        existing = read_json(out_path) if out_path.exists() else []
        # Record old entries' 400 status for Phase 1 direct truncation
        prev_400 = set()
        for j in existing:
            if "400" in j.get("error", ""):
                prev_400.add((j["query_id"], j.get("scheme", "")))
        # Filter old failed entries (403 skip, other errors re-judge)
        existing = [j for j in existing if not j.get("error", "") or "403" in j.get("error", "")]  # 403 permanent skip, other errors re-judge
        existing_query_ids = set()
        for j in existing:
            existing_query_ids.add((j["query_id"], j.get("scheme", "")))

        to_judge = []
        for scheme in sorted(schemes_available):
            answers = load_answers_for_book(book_title, args.mode, scheme, args.top_k)
            if not answers:
                print(f"    Missing answer: {scheme}", flush=True)
                continue
            for qid, row in answers.items():
                if (qid, scheme) not in existing_query_ids:
                    to_judge.append((qid, scheme, row))

        if not to_judge:
            tracker.finish_book()
            all_judgments.extend(existing)
            total_judged += len(existing)
            continue

        total_pending = len(existing) + len(to_judge)
        tracker.start_scheme("judge", total_pending, skipped=len(existing))

        calls = []
        metas: list[dict] = []
        for qid, scheme, answer_row in to_judge:
            generated = answer_row.get("generated_answer", "")
            chunks = answer_row.get("context_chunks", [])[: args.top_k]
            question = answer_row.get("question", "")
            ground_truth = answer_row.get("ground_truth_answer", "")
            qt = answer_row.get("question_type", "")
            diff = answer_row.get("difficulty", "")

            if not generated:
                metas.append({
                    "query_id": qid, "book_title": book_title, "question": question,
                    "ground_truth_answer": ground_truth, "question_type": qt, "difficulty": diff,
                    "scheme": scheme, "generated_answer": generated, "context_chunks": chunks,
                    **make_empty_scores("empty_answer"), "judge_model": MODEL_JUDGE, "error": "empty_answer",
                })
                continue

            # Old entry had 400 -> truncate directly this time (no wasted full chunk call)
            use_truncate = args.truncate_chunks == "always" or (qid, scheme) in prev_400
            chunks_text = format_chunks_text(chunks, max_chars=1500 if use_truncate else None)
            ref_evidence = evidence_map.get(qid, "(no reference evidence available)")
            call = build_judge_call(question, ground_truth, ref_evidence, generated, chunks_text)
            calls.append(call)
            # Record prompt length for terminal diagnostics
            prompt_len = len(call["system"]) + len(call["user"])
            metas.append({
                "query_id": qid, "book_title": book_title, "question": question,
                "ground_truth_answer": ground_truth, "question_type": qt, "difficulty": diff,
                "scheme": scheme, "generated_answer": generated, "context_chunks": chunks,
                "_prompt_len": prompt_len,
            })

        def on_result(idx: int, result: dict) -> None:
            scores = parse_judge_result(result)
            if scores is None:
                scores = make_empty_scores(result.get("error", "parse_failed"))
                metas[idx].update(scores)
                metas[idx]["error"] = result.get("error", "parse_failed")
                # Terminal diagnostics: print prompt length and error on failure
                plen = metas[idx].get("_prompt_len", "?")
                err = result.get("error", "?")[:80]
                print(f"\n  ✗ [{metas[idx]['scheme']}] {metas[idx]['query_id'][-40:]}  prompt {plen} chars  {err}", flush=True)
            else:
                metas[idx].update(scores)
                metas[idx]["error"] = ""
            metas[idx]["judge_model"] = MODEL_JUDGE
            ok = scores[JUDGE_DIMENSIONS[0]] > 0
            tracker.update(ok=ok)

        if calls:
            # Phase 1: all with full chunks (no truncation)
            CHUNK = 20
            book_judgments = list(existing)
            for chunk_start in range(0, len(calls), CHUNK):
                chunk_calls = calls[chunk_start:chunk_start + CHUNK]

                def make_callback(offset: int):
                    def _cb(idx: int, result: dict) -> None:
                        return on_result(offset + idx, result)
                    return _cb

                chat_batch(
                    chunk_calls,
                    model=MODEL_JUDGE,
                    temperature=JUDGE_TEMPERATURE,
                    workers=args.workers,
                    on_result=make_callback(chunk_start),
                )
                for m in metas:
                    if m is not None and JUDGE_DIMENSIONS[0] not in m:
                        m.update(make_empty_scores("no_result"))
                        m["error"] = "no_result"
                        m["judge_model"] = MODEL_JUDGE

            # Phase 2: retry 400-error entries with truncated chunks (retry mode only)
            if args.truncate_chunks == "retry":
                retry_calls = []
                retry_indices = []
                for idx, meta in enumerate(metas):
                    if meta and "400" in meta.get("error", ""):
                        if idx < len(to_judge):
                            _, scheme, answer_row = to_judge[idx]
                            chunks = answer_row.get("context_chunks", [])[: args.top_k]
                            chunks_text = format_chunks_text(chunks, max_chars=1500)
                            question = answer_row.get("question", "")
                            ground_truth = answer_row.get("ground_truth_answer", "")
                            generated = answer_row.get("generated_answer", "")
                            ref_evidence = evidence_map.get(to_judge[idx][0], "(no reference evidence available)")
                            call = build_judge_call(question, ground_truth, ref_evidence, generated, chunks_text)
                            retry_calls.append(call)
                            # Store truncated prompt length for diagnostics
                            metas[idx]["_prompt_len_retry"] = len(call["system"]) + len(call["user"])
                            retry_indices.append(idx)

                if retry_calls:
                    def retry_callback(idx: int, result: dict) -> None:
                        real_idx = retry_indices[idx]
                        scores = parse_judge_result(result)
                        if scores is None:
                            scores = make_empty_scores(result.get("error", "parse_failed"))
                            metas[real_idx].update(scores)
                            metas[real_idx]["error"] = result.get("error", "parse_failed")
                            plen = metas[real_idx].get("_prompt_len_retry", "?")
                            err = result.get("error", "?")[:80]
                            print(f"\n  ✗[truncated retry] [{metas[real_idx]['scheme']}] {metas[real_idx]['query_id'][-40:]}  prompt {plen} chars  {err}", flush=True)
                        else:
                            metas[real_idx].update(scores)
                            metas[real_idx]["error"] = ""
                            print(f"\n  ✓[truncated retry OK] [{metas[real_idx]['scheme']}] {metas[real_idx]['query_id'][-40:]}", flush=True)
                        metas[real_idx]["judge_model"] = MODEL_JUDGE
                        ok = scores[JUDGE_DIMENSIONS[0]] > 0
                        tracker.update(ok=ok)

                    for chunk_start in range(0, len(retry_calls), CHUNK):
                        chunk_calls = retry_calls[chunk_start:chunk_start + CHUNK]
                        chat_batch(
                            chunk_calls,
                            model=MODEL_JUDGE,
                            temperature=JUDGE_TEMPERATURE,
                            workers=args.workers,
                            on_result=lambda idx, r, offset=chunk_start: retry_callback(offset + idx, r),
                        )

            # Write to disk
            current = list(existing)
            for m in metas:
                if m is not None and JUDGE_DIMENSIONS[0] not in m:
                    m.update(make_empty_scores("no_result"))
                    m["error"] = "no_result"
                    m["judge_model"] = MODEL_JUDGE
            current.extend([m for m in metas if m is not None])
            write_json(out_path, compact_judgments(current))
        else:
            book_judgments = list(existing)
            for meta in metas:
                if JUDGE_DIMENSIONS[0] not in meta:
                    meta.update(make_empty_scores("no_result"))
                    meta["error"] = "no_result"
                    meta["judge_model"] = MODEL_JUDGE
            book_judgments.extend([m for m in metas if m is not None])
            write_json(out_path, compact_judgments(book_judgments))
        # Re-read disk file to ensure accurate count
        book_judgments = read_json(out_path) if out_path.exists() else []
        all_judgments.extend(book_judgments)
        total_judged += len(book_judgments)

        tracker.finish_book()

    # skip_failed_first mode: after first round, truncate-retry all failed entries
    if args.truncate_chunks == "skip_failed_first":
        print("\n=== Second pass: truncate-retry all failed entries ===", flush=True)
        for book_title in books:
            out_path = output_path_for_book(book_title, args.mode, args.top_k)
            if not out_path.exists():
                continue
            all_entries = read_json(out_path)
            failed_entries = [j for j in all_entries if j.get("error", "")]
            if not failed_entries:
                continue
            print(f"  {book_title}: {len(failed_entries)} failed, truncating retry...", flush=True)

            # Rebuild truncated calls for failed entries
            calls = []
            metas = []
            for j in failed_entries:
                qid = j["query_id"]
                scheme = j["scheme"]
                # Re-fetch chunks from answer file
                answers = load_answers_for_book(book_title, args.mode, scheme, args.top_k)
                row = answers.get(qid)
                if not row:
                    continue
                generated = row.get("generated_answer", "")
                question = row.get("question", "")
                ground_truth = row.get("ground_truth_answer", "")
                chunks = row.get("context_chunks", [])[: args.top_k]
                chunks_text = format_chunks_text(chunks, max_chars=1500)
                ref_evidence = evidence_map.get(qid, "(no reference evidence available)")
                call = build_judge_call(question, ground_truth, ref_evidence, generated, chunks_text)
                calls.append(call)
                metas.append({
                    "query_id": qid, "book_title": book_title, "scheme": scheme,
                    "question": question, "ground_truth_answer": ground_truth,
                    "question_type": j.get("question_type", ""), "difficulty": j.get("difficulty", ""),
                    "generated_answer": generated, "context_chunks": chunks,
                })

            if not calls:
                continue

            # Send truncated version in batches
            CHUNK = 20
            for chunk_start in range(0, len(calls), CHUNK):
                chunk_calls = calls[chunk_start:chunk_start + CHUNK]

                def make_retry_callback(offset: int):
                    def _cb(idx: int, result: dict) -> None:
                        real_idx = offset + idx
                        scores = parse_judge_result(result)
                        if scores is None:
                            scores = make_empty_scores(result.get("error", "parse_failed"))
                            metas[real_idx]["error"] = result.get("error", "parse_failed")
                        else:
                            metas[real_idx].update(scores)
                            metas[real_idx]["error"] = ""
                        metas[real_idx]["judge_model"] = MODEL_JUDGE
                    return _cb

                chat_batch(
                    chunk_calls, model=MODEL_JUDGE, temperature=JUDGE_TEMPERATURE,
                    workers=args.workers, on_result=make_retry_callback(chunk_start),
                )

            # Replace old failed entries with successful ones, write to disk
            success_entries = [j for j in all_entries if not j.get("error", "")]
            success_entries.extend(metas)
            write_json(out_path, compact_judgments(success_entries))
            print(f"    {book_title}: retry done", flush=True)

        # Rebuild global summary
        all_judgments = []
        for book_title in books:
            out_path = output_path_for_book(book_title, args.mode, args.top_k)
            if out_path.exists():
                all_judgments.extend(read_json(out_path))
        total_ok = sum(1 for j in all_judgments if not j.get("error", ""))
        total_fail = sum(1 for j in all_judgments if j.get("error", ""))
        print(f"Second pass done: ✓{total_ok} ✗{total_fail}", flush=True)

    all_output_name = "judgments_all.eval.json"
    write_json(JUDGMENTS_DIR / all_output_name, compact_judgments(all_judgments))
    print(tracker.total_summary(len(books), total_judged, global_start))


if __name__ == "__main__":
    main()
