from __future__ import annotations

from pathlib import Path

from utils_embedding import load_meta
from utils_io import read_json

SUPPORTED_QUERY_VARIANTS = ["default", "en", "default_en"]


def query_variant_suffix(query_variant: str) -> str:
    if query_variant not in SUPPORTED_QUERY_VARIANTS:
        raise ValueError(f"Unsupported query_variant: {query_variant}. Choices: {SUPPORTED_QUERY_VARIANTS}")
    return "" if query_variant == "default" else f".{query_variant}"


def build_query_text(row: dict, query_variant: str, query_id: str) -> str:
    question = str(row.get("question", "") or "").strip()
    retrieval_question_en = str(row.get("retrieval_question_en", "") or "").strip()
    if query_variant == "default":
        return question
    if query_variant == "en":
        if not retrieval_question_en:
            raise ValueError(f"query_variant=en requires retrieval_question_en, but field is empty: query_id={query_id}")
        return retrieval_question_en
    if query_variant == "default_en":
        if not retrieval_question_en:
            raise ValueError(f"query_variant=default_en requires retrieval_question_en, but field is empty: query_id={query_id}")
        if not question:
            return retrieval_question_en
        return f"{question}\n{retrieval_question_en}"
    raise ValueError(f"Unsupported query_variant: {query_variant}")


def iter_query_rows(source_rows: list[dict], default_book: str | None = None, query_variant: str = "default") -> list[dict]:
    filtered_rows: list[dict] = []
    for row_idx, row in enumerate(source_rows, start=1):
        if isinstance(row, dict) and "question" in row:
            query_id = str(row.get("query_id", "") or f"query_{row_idx}")
            text = build_query_text(row, query_variant, query_id).strip()
            if not text:
                continue
            filtered_rows.append(
                {
                    "query_id": query_id,
                    "book_title": str(row.get("book_title", "") or default_book or ""),
                    "chapter_id": str(row.get("chapter_id", "") or ""),
                    "question": text,
                    "query_variant": query_variant,
                }
            )
            continue

        book_title = str(row.get("book_title", "") or default_book or "").strip()
        chapter_id = str(row.get("chapter_id", "") or "").strip()
        for qa_idx, qa in enumerate(row.get("qas", []) or [], start=1):
            if qa.get("parse_failed", False):
                continue
            query_id = str(qa.get("qa_id", "") or f"{chapter_id}.qa_{qa_idx}" or f"query_{row_idx}_{qa_idx}")
            text = build_query_text(qa, query_variant, query_id).strip()
            if not text:
                continue
            filtered_rows.append(
                {
                    "query_id": query_id,
                    "book_title": book_title,
                    "chapter_id": chapter_id,
                    "question": text,
                    "query_variant": query_variant,
                }
            )
    return filtered_rows


def validate_in_book_query_embeddings(book: str, query_path: Path, meta_path: Path, query_variant: str = "default") -> None:
    source_rows = read_json(query_path)
    current_rows = iter_query_rows(source_rows, default_book=book, query_variant=query_variant)
    current_ids = [row["query_id"] for row in current_rows]
    meta_rows = load_meta(meta_path)
    meta_ids = [row["query_id"] for row in meta_rows]
    meta_variants = {str(row.get("query_variant", "default") or "default") for row in meta_rows}

    current_set = set(current_ids)
    meta_set = set(meta_ids)
    if current_ids == meta_ids and meta_variants <= {query_variant}:
        return

    missing_in_meta = sorted(current_set - meta_set)
    extra_in_meta = sorted(meta_set - current_set)
    raise RuntimeError(
        "in-book query embeddings do not match current queries, please re-run 03_embed_queries.py.\n"
        f"book={book}\n"
        f"query_variant={query_variant}\n"
        f"meta_variants={sorted(meta_variants)}\n"
        f"queries_count={len(current_ids)} meta_count={len(meta_ids)}\n"
        f"missing_in_meta_count={len(missing_in_meta)} sample={missing_in_meta[:5]}\n"
        f"extra_in_meta_count={len(extra_in_meta)} sample={extra_in_meta[:5]}"
    )


def validate_global_query_embeddings(query_path: Path, meta_path: Path, query_variant: str = "default") -> None:
    source_rows = read_json(query_path)
    current_rows = iter_query_rows(source_rows, query_variant=query_variant)
    current_ids = [row["query_id"] for row in current_rows]
    meta_rows = load_meta(meta_path)
    meta_ids = [row["query_id"] for row in meta_rows]
    meta_variants = {str(row.get("query_variant", "default") or "default") for row in meta_rows}

    current_set = set(current_ids)
    meta_set = set(meta_ids)
    if current_ids == meta_ids and meta_variants <= {query_variant}:
        return

    missing_in_meta = sorted(current_set - meta_set)
    extra_in_meta = sorted(meta_set - current_set)
    raise RuntimeError(
        "global query embeddings do not match current queries_all, please re-run 03_embed_queries.py.\n"
        f"query_variant={query_variant}\n"
        f"meta_variants={sorted(meta_variants)}\n"
        f"queries_count={len(current_ids)} meta_count={len(meta_ids)}\n"
        f"missing_in_meta_count={len(missing_in_meta)} sample={missing_in_meta[:5]}\n"
        f"extra_in_meta_count={len(extra_in_meta)} sample={extra_in_meta[:5]}"
    )


def validate_run_query_variant(run_rows: list[dict], query_variant: str, *, source: str) -> None:
    variants = {str(row.get("query_variant", "default") or "default") for row in run_rows}
    if variants <= {query_variant}:
        return
    raise RuntimeError(
        "query_variant in run file does not match current parameters, check 05_retrieve.py output file.\n"
        f"source={source}\n"
        f"expected_query_variant={query_variant}\n"
        f"actual_query_variants={sorted(variants)}"
    )


def validate_run_vs_relevance(run_rows: list[dict], relevance_rows: list[dict], *, book: str, scheme: str) -> None:
    run_ids = [row["query_id"] for row in run_rows]
    relevance_ids = [row["query_id"] for row in relevance_rows]
    if run_ids == relevance_ids:
        return

    run_set = set(run_ids)
    rel_set = set(relevance_ids)
    missing_in_relevance = sorted(run_set - rel_set)
    extra_in_relevance = sorted(rel_set - run_set)
    raise RuntimeError(
        "run and relevance query_id sets do not match, re-run related artifacts.\n"
        f"book={book} scheme={scheme}\n"
        f"run_count={len(run_ids)} relevance_count={len(relevance_ids)}\n"
        f"missing_in_relevance_count={len(missing_in_relevance)} sample={missing_in_relevance[:5]}\n"
        f"extra_in_relevance_count={len(extra_in_relevance)} sample={extra_in_relevance[:5]}"
    )
