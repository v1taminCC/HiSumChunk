#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
WORK_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))
INPUT_DIR = WORK_BOOK_DIR
OUTPUT_DIR = WORK_BOOK_DIR
DEFAULT_WINDOW_WORDS = 512
DEFAULT_OVERLAP_WORDS = 0
ANCHOR_PATTERN = re.compile(r"^<<<CHAPTER_ID:(.*?)>>>\s*$", re.MULTILINE)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


class ChapterSpan(dict):
    pass


class WordEntry(dict):
    pass


def build_output_suffix(window_words: int, overlap_words: int) -> str:
    overlap_part = f"Overlap{overlap_words}W" if overlap_words > 0 else "NoOverlap"
    return f"_Fixed{window_words}W_{overlap_part}_split.json"


def parse_book_text(raw_text: str) -> Tuple[str, List[ChapterSpan], List[WordEntry]]:
    matches = list(ANCHOR_PATTERN.finditer(raw_text))
    if not matches:
        raise ValueError("No chapter anchors found: <<<CHAPTER_ID:...>>>")

    chapters: List[ChapterSpan] = []
    words: List[WordEntry] = []
    plain_cursor = 0

    for idx, match in enumerate(matches):
        chapter_id = match.group(1).strip()
        content_start = match.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        chapter_text = raw_text[content_start:content_end]

        plain_start = plain_cursor
        for word_match in WORD_PATTERN.finditer(chapter_text):
            words.append(
                WordEntry(
                    text=word_match.group(0),
                    chapter_id=chapter_id,
                    plain_start=plain_cursor + word_match.start(),
                    plain_end=plain_cursor + word_match.end(),
                )
            )
        plain_cursor += len(chapter_text)
        plain_end = plain_cursor

        chapters.append(
            ChapterSpan(
                chapter_id=chapter_id,
                text=chapter_text,
                plain_start=plain_start,
                plain_end=plain_end,
            )
        )

    return raw_text, chapters, words


def chunk_plain_text_from_chapters(chapters: List[ChapterSpan], start: int, end: int) -> str:
    parts: List[str] = []
    for chapter in chapters:
        overlap_start = max(start, int(chapter["plain_start"]))
        overlap_end = min(end, int(chapter["plain_end"]))
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - int(chapter["plain_start"])
        local_end = overlap_end - int(chapter["plain_start"])
        chapter_text = str(chapter["text"])
        parts.append(chapter_text[local_start:local_end])
    return "".join(parts)


def build_chunk_text(chapters: List[ChapterSpan], covered_ids: List[str], start: int, end: int) -> str:
    return chunk_plain_text_from_chapters(chapters, start, end).strip()


def build_items_for_book(book_txt_path: Path, window_words: int, overlap_words: int) -> List[dict]:
    raw_text = book_txt_path.read_text(encoding="utf-8")
    _, chapters, words = parse_book_text(raw_text)

    if not words:
        return []

    chapter_order = [str(ch["chapter_id"]) for ch in chapters]
    items: List[dict] = []
    step_words = max(1, window_words - overlap_words)

    chunk_index = 0
    for word_start_idx in range(0, len(words), step_words):
        chunk_words = words[word_start_idx:word_start_idx + window_words]
        if not chunk_words:
            continue

        chunk_index += 1
        first_word = chunk_words[0]
        last_word = chunk_words[-1]
        source_start = int(first_word["plain_start"])
        source_end = int(last_word["plain_end"])

        covered_ids: List[str] = []
        seen = set()
        for word in chunk_words:
            cid = str(word["chapter_id"])
            if cid not in seen:
                seen.add(cid)
                covered_ids.append(cid)

        active_chapter_id = covered_ids[0]
        crosses_chapter = len(covered_ids) > 1
        chunk_text = build_chunk_text(chapters, covered_ids, source_start, source_end)
        plain_text = chunk_plain_text_from_chapters(chapters, source_start, source_end)
        word_count = len(chunk_words)

        items.append(
            {
                "block_id": chunk_index,
                "length": word_count,
                "text": chunk_text,
                "char_length": len(plain_text),
                "active_chapter_id": active_chapter_id,
                "covered_chapter_ids": covered_ids,
                "crosses_chapter": crosses_chapter,
                "source_start": source_start,
                "source_end": source_end,
            }
        )

        if word_start_idx + window_words >= len(words):
            break

    return items


def write_book_json(book_txt_path: Path, window_words: int, overlap_words: int) -> Path:
    items = build_items_for_book(book_txt_path, window_words=window_words, overlap_words=overlap_words)
    data = {
        "window_words": window_words,
        "overlap_words": overlap_words,
            "split_method": "Fixed-1024W",
        "max_length": max((int(item["length"]) for item in items), default=0),
        "min_length": min((int(item["length"]) for item in items), default=0),
        "items": items,
    }

    output_name = f"{book_txt_path.stem}{build_output_suffix(window_words, overlap_words)}"
    output_path = OUTPUT_DIR / output_name
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-window chunking and JSON export")
    parser.add_argument("--window_words", type=int, default=DEFAULT_WINDOW_WORDS, help="Window size in English words")
    parser.add_argument("--overlap_words", type=int, default=DEFAULT_OVERLAP_WORDS, help="Overlap size in words between adjacent windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_words <= 0:
        raise ValueError("window_words must be greater than 0")
    if args.overlap_words < 0:
        raise ValueError("overlap_words must not be negative")
    if args.overlap_words >= args.window_words:
        raise ValueError("overlap_words must be less than window_words")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(p for p in INPUT_DIR.glob("*.txt") if p.is_file())
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found: {INPUT_DIR}")

    total_chunks = 0
    for txt_path in txt_files:
        output_path = write_book_json(txt_path, window_words=args.window_words, overlap_words=args.overlap_words)
        data = json.loads(output_path.read_text(encoding="utf-8"))
        chunk_count = len(data.get("items", []))
        total_chunks += chunk_count
        print(f"[OK] {output_path.name} <- {chunk_count} chunks")

    print(f"Done: wrote {len(txt_files)} book JSON files, {total_chunks} chunks total")
    print(f"window_words: {args.window_words}")
    print(f"overlap_words: {args.overlap_words}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
