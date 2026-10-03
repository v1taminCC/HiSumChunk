#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use the judge model to judge ablation variant answers.
Dimensions: grounding, correctness, hallucination_absence
Supports per-QA checkpoint resume.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    ANSWER_TOP_K,
    ANSWERS_BY_BOOK_DIR,
    JUDGMENTS_BY_BOOK_DIR,
    PROMPTS_DIR,
    QA_DRAFTS_DIR,
    VARIANTS,
)
from utils_io import ensure_dir, read_json, write_json
from utils_simple_api import MODEL_JUDGE, chat_batch, DEFAULT_WORKERS

JUDGE_DIMENSIONS = ["grounding", "correctness", "hallucination_absence"]
JUDGE_TEMPERATURE = 0.0


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()


def load_qa_evidence() -> dict[str, str]:
    qa_path = QA_DRAFTS_DIR / "qa_drafts_all.json"
    if not qa_path.exists():
        return {}
    rows = read_json(qa_path)
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


def format_chunks_text(chunks: list[dict], max_chars: int | None = None) -> str:
    parts = []
    for c in chunks:
        text = c["text"]
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + f"... (truncated, original {len(c['text'])} chars)"
        parts.append(f"--- Chunk (rank={c['rank']}, score={c['score']:.4f}) ---\n{text}")
    return "\n\n".join(parts)


def build_judge_call(question: str, ground_truth: str, evidence: str,
                     generated_answer: str, chunks_text: str) -> dict:
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
        return scores
    except Exception:
        return None


def make_empty_scores(reason: str) -> dict:
    scores = {}
    for dim in JUDGE_DIMENSIONS:
        scores[dim] = 0
    return scores


def list_book_titles() -> list[str]:
    books = set()
    for fp in ANSWERS_BY_BOOK_DIR.glob(f"*.in_book.{VARIANTS[0]}.top{ANSWER_TOP_K}.json"):
        name = fp.stem
        suffix = f".in_book.{VARIANTS[0]}.top{ANSWER_TOP_K}"
        if name.endswith(suffix):
            books.add(name[: -len(suffix)])
    return sorted(books)


def load_existing_judgments(out_path: Path) -> tuple[list[dict], set[str]]:
    if not out_path.exists():
        return [], set()
    existing = read_json(out_path)
    done_ids = {j["query_id"] for j in existing if j.get(JUDGE_DIMENSIONS[0], 0) > 0}
    return existing, done_ids


def judge_book_variant(
    book: str,
    variant: str,
    evidence_map: dict[str, str],
    workers: int,
    out_path: Path,
) -> tuple[int, int, int]:
    """Return (new_judged, skipped, failed)"""
    existing, done_ids = load_existing_judgments(out_path)

    answer_path = ANSWERS_BY_BOOK_DIR / f"{book}.in_book.{variant}.top{ANSWER_TOP_K}.json"
    if not answer_path.exists():
        return 0, 0, 0

    answers = read_json(answer_path)
    pending = [row for row in answers if row["query_id"] not in done_ids]

    if not pending:
        return 0, len(done_ids), 0

    calls = []
    metas = []
    call_to_meta_idx = []

    for row in pending:
        qid = row["query_id"]
        generated = row.get("generated_answer", "")
        question = row.get("question", "")
        ground_truth = row.get("ground_truth_answer", "")
        chunks = row.get("context_chunks", [])[:ANSWER_TOP_K]

        meta = {
            "query_id": qid,
            "book_title": book,
            "question": question,
            "ground_truth_answer": ground_truth,
            "question_type": row.get("question_type", ""),
            "difficulty": row.get("difficulty", ""),
            "variant": variant,
            "generated_answer": generated,
            "context_chunks": chunks,
        }

        if not generated:
            meta.update(make_empty_scores("empty_answer"))
            meta["judge_model"] = MODEL_JUDGE
            meta["error"] = "empty_answer"
            metas.append(meta)
            continue

        chunks_text = format_chunks_text(chunks)
        ref_evidence = evidence_map.get(qid, "(no reference evidence available)")
        call = build_judge_call(question, ground_truth, ref_evidence, generated, chunks_text)
        calls.append(call)
        call_to_meta_idx.append(len(metas))
        metas.append(meta)

    ok_count = [0]
    fail_count = [0]

    def on_result(idx: int, result: dict):
        real_idx = call_to_meta_idx[idx]
        scores = parse_judge_result(result)
        if scores is None:
            scores = make_empty_scores(result.get("error", "parse_failed"))
            metas[real_idx].update(scores)
            metas[real_idx]["error"] = result.get("error", "parse_failed")
            fail_count[0] += 1
        else:
            metas[real_idx].update(scores)
            metas[real_idx]["error"] = ""
            ok_count[0] += 1
        metas[real_idx]["judge_model"] = MODEL_JUDGE

    if calls:
        chat_batch(
            calls,
            model=MODEL_JUDGE,
            temperature=JUDGE_TEMPERATURE,
            workers=workers,
            on_result=on_result,
        )

        # Retry 400 errors with truncated chunks
        retry_calls = []
        retry_meta_indices = []
        for i, meta in enumerate(metas):
            if "400" in meta.get("error", ""):
                chunks = meta.get("context_chunks", [])[:ANSWER_TOP_K]
                chunks_text = format_chunks_text(chunks, max_chars=1500)
                ref_evidence = evidence_map.get(meta["query_id"], "(no reference evidence available)")
                call = build_judge_call(
                    meta["question"], meta["ground_truth_answer"],
                    ref_evidence, meta["generated_answer"], chunks_text,
                )
                retry_calls.append(call)
                retry_meta_indices.append(i)

        if retry_calls:
            print(f"    Retrying {len(retry_calls)} records (truncated chunks)...", flush=True)

            def on_retry(idx: int, result: dict):
                real_idx = retry_meta_indices[idx]
                scores = parse_judge_result(result)
                if scores is None:
                    scores = make_empty_scores(result.get("error", "parse_failed"))
                    metas[real_idx].update(scores)
                    metas[real_idx]["error"] = result.get("error", "retry_failed")
                else:
                    metas[real_idx].update(scores)
                    metas[real_idx]["error"] = ""
                    ok_count[0] += 1
                    fail_count[0] -= 1

            chat_batch(
                retry_calls,
                model=MODEL_JUDGE,
                temperature=JUDGE_TEMPERATURE,
                workers=workers,
                on_result=on_retry,
            )

    all_rows = existing + metas
    write_json(out_path, all_rows)

    return ok_count[0], len(done_ids), fail_count[0]


def main():
    print("=" * 60)
    print("Ablation Experiment - Step 5: Judge answers (per-QA checkpoint resume)")
    print("=" * 60)

    ensure_dir(JUDGMENTS_BY_BOOK_DIR)
    books = list_book_titles()
    evidence_map = load_qa_evidence()

    print(f"Books: {len(books)}")
    print(f"Variants: {VARIANTS}")
    print(f"Judge model: {MODEL_JUDGE}")
    print(f"Dimensions: {JUDGE_DIMENSIONS}")
    print(f"Evidence entries: {len(evidence_map)}")

    total_new = 0
    total_skipped = 0
    total_failed = 0
    t0 = time.time()

    for variant in VARIANTS:
        print(f"\n--- Variant: {variant} ---")

        for bi, book in enumerate(books, 1):
            out_path = JUDGMENTS_BY_BOOK_DIR / f"{book}.{variant}.eval.json"
            ok, skipped, failed = judge_book_variant(book, variant, evidence_map, DEFAULT_WORKERS, out_path)
            total_new += ok
            total_skipped += skipped
            total_failed += failed

            if ok == 0 and skipped > 0 and failed == 0:
                print(f"  [{bi}/{len(books)}] {book} - all done ({skipped} records)")
            elif ok > 0 or failed > 0:
                print(f"  [{bi}/{len(books)}] {book} - new✓{ok} skip{skipped} fail✗{failed}")
            else:
                print(f"  [{bi}/{len(books)}] {book} - no answer file")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Judging complete")
    print(f"  New: {total_new}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Failed: {total_failed}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
