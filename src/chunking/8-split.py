#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text chunking script based on H2 region areas (overlapping greedy sliding window)

Revision notes:
  - Preserves original anchor fields (active_chapter_id / covered_chapter_ids / crosses_chapter / source_start / source_end)
  - Chunk length, overlap length, backtrack range, and output length all counted as "English words + Chinese characters"
  - Can be run standalone to re-chunk a given *_L1Outline.json and output a new *_split.json

Input structure:
  {
    "max_length": ...,
    "min_length": ...,
    "h2_regions": [
      {
        "h1_outline": "...",
        "chunks": [
          {
            "length": ...,
            "h2_outline": "...",
            "text": "...original text..."
          },
          ...
        ]
      },
      ...
    ]
  }

Output structure:
  {
    "split_method": "HiSumChunk",
    "model": os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507").split("/")[-1],
    "max_length": ...,
    "min_length": ...,
    "items": [
      {
        "region_id": "1",
        "is_region_summary": true/false,
        "chunk_info": "...",
        "length": 123,
        "text": "..."
      },
      ...
    ]
  }
"""

import os
import sys
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
EXAMPLE_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))

# ================================================================================================
# ==================================== Configurable Parameters ============================================
# ================================================================================================

# ---------------------------------- Chunking parameters -----------------------------------
TARGET_LEN: int = 1000
OVERLAP: int = 100
BACKTRACK_RANGE: int = 150

# Sentence-ending punctuation set (Chinese + English)
PUNCTUATIONS: str = ".!?"

# ---------------------------------- File path config -----------------------------------
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
EXAMPLE_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))
INPUT_FILE: str = str(EXAMPLE_BOOK_DIR / "Far From the Madding Crowd_MergeRule_Title_NoHeaderFooter_Level_L2Outline_MergeH2_L1Outline.json")
OUTPUT_SUFFIX: str = "_split.json"

# ---------------------------------- Other config ---------------------------------------
SHOW_PROGRESS: bool = True

# ================================================================================================
# ========================================== Code ============================================
# ================================================================================================

import argparse
import bisect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
UnitSpan = Tuple[int, int]


def _log(msg: str) -> None:
    if SHOW_PROGRESS:
        print(msg)


def ensure_outline_with_period(outline: str) -> str:
    if not outline:
        return ""
    outline = str(outline)
    if outline[-1] in PUNCTUATIONS:
        return outline
    return outline + "."


def is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0x2CEB0 <= code <= 0x2EBEF
        or 0x30000 <= code <= 0x3134F
    )


def extract_count_units(text: str) -> List[UnitSpan]:
    """
    Extract count units:
      - CJK character: each counts as 1
      - English/numeric sequence: each word counts as 1
      - Other symbols/whitespace: not counted
    Each element in the return value is a (start_char, end_char) character range.
    """
    units: List[UnitSpan] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if is_cjk_char(ch):
            units.append((i, i + 1))
            i += 1
            continue

        match = WORD_RE.match(text, i)
        if match:
            units.append((match.start(), match.end()))
            i = match.end()
            continue

        i += 1

    return units


def count_length_units(text: str) -> int:
    return len(extract_count_units(text))


def unit_index_to_char_start(units: Sequence[UnitSpan], unit_index: int, text_len: int) -> int:
    if unit_index <= 0:
        return 0
    if unit_index >= len(units):
        return text_len
    return units[unit_index][0]


def greedy_sliding_window(
    text: str,
    target_len: int,
    overlap: int,
    backtrack_range: int,
    punctuations: str,
) -> List[str]:
    """
    Overlapping greedy sliding window chunking.

    Counting units:
      - English: word-level
      - Chinese: character-level

    Behavior:
      - Try to split at sentence-ending punctuation near target_len
      - Segments overlap by `overlap` count units
      - If no punctuation found, force-split at the target_len boundary
    """
    if not text:
        return []

    units = extract_count_units(text)
    if not units:
        return [text] if text.strip() else []

    total_units = len(units)
    text_len = len(text)
    unit_ends = [end for _, end in units]
    segments: List[str] = []
    start_unit = 0

    while start_unit < total_units:
        start_char = units[start_unit][0]
        preferred_end_unit = start_unit + target_len

        if preferred_end_unit >= total_units:
            segments.append(text[start_char:])
            break

        search_start_unit = max(start_unit, preferred_end_unit - backtrack_range)
        preferred_end_char = units[preferred_end_unit - 1][1]
        search_start_char = unit_index_to_char_start(units, search_start_unit, text_len)

        cut_char = None
        for i in range(preferred_end_char - 1, search_start_char - 1, -1):
            if text[i] in punctuations:
                cut_char = i + 1
                break

        if cut_char is None:
            cut_unit = preferred_end_unit
            cut_char = units[cut_unit - 1][1]
        else:
            cut_unit = bisect.bisect_right(unit_ends, cut_char)
            if cut_unit <= start_unit:
                cut_unit = preferred_end_unit
                cut_char = units[cut_unit - 1][1]

        if cut_unit <= start_unit:
            cut_unit = min(start_unit + target_len, total_units)
            if cut_unit <= start_unit:
                cut_unit = min(start_unit + 1, total_units)
            cut_char = units[cut_unit - 1][1]

        segment = text[start_char:cut_char]
        segments.append(segment)

        next_start_unit = cut_unit - overlap
        if next_start_unit <= start_unit:
            next_start_unit = cut_unit
        start_unit = max(0, next_start_unit)

    return segments


def load_input(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"[ERROR] Input file does not exist: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to read JSON: {e}")
            sys.exit(1)

    return data


def build_flat_items(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the hierarchical h2_regions structure into a flat items list with chunking.

    Conventions:
      - region_id: numbered as "1", "2", ... following h2_regions order
      - Region summary items are placed before their content chunks
      - Anchor fields are fully passed through, no loss
      - Output length is "English words + Chinese characters"
    """
    max_length = data.get("max_length", 0)
    min_length = data.get("min_length", 0)
    h2_regions = data.get("h2_regions", [])

    if not isinstance(h2_regions, list):
        print("[ERROR] h2_regions in input JSON is not a list.")
        sys.exit(1)

    all_text_entries: List[Dict[str, Any]] = []

    for region_idx, region in enumerate(h2_regions, start=1):
        h1_outline = region.get("h1_outline", "")
        chunks = region.get("chunks", [])
        if not isinstance(chunks, list):
            chunks = []

        for chunk_local_index, chunk in enumerate(chunks, start=1):
            text = chunk.get("text", "") or ""
            all_text_entries.append(
                {
                    "region_idx": region_idx,
                    "chunk_local_index": chunk_local_index,
                    "h1_outline": h1_outline,
                    "h2_outline": chunk.get("h2_outline", ""),
                    "length": chunk.get("length", 0),
                    "text": text,
                    "active_chapter_id": chunk.get("active_chapter_id", ""),
                    "covered_chapter_ids": chunk.get("covered_chapter_ids", []),
                    "crosses_chapter": chunk.get("crosses_chapter", False),
                    "source_start": chunk.get("source_start"),
                    "source_end": chunk.get("source_end"),
                }
            )

    segments_by_region: Dict[int, List[Dict[str, Any]]] = {
        region_idx: [] for region_idx, _ in enumerate(h2_regions, start=1)
    }

    for entry in all_text_entries:
        region_idx = entry["region_idx"]
        chunk_local_index = entry["chunk_local_index"]
        h2_outline = entry["h2_outline"]
        original_length = entry["length"]
        raw_text = entry["text"]
        active_chapter_id = entry["active_chapter_id"]
        covered_chapter_ids = entry["covered_chapter_ids"]
        crosses_chapter = entry["crosses_chapter"]
        source_start = entry["source_start"]
        source_end = entry["source_end"]

        segments = greedy_sliding_window(
            text=raw_text,
            target_len=TARGET_LEN,
            overlap=OVERLAP,
            backtrack_range=BACKTRACK_RANGE,
            punctuations=PUNCTUATIONS,
        )

        prefix = ensure_outline_with_period(h2_outline)

        for sub_idx, seg in enumerate(segments, start=1):
            full_text = f"{prefix}{seg}"
            chunk_info = f"Region {region_idx}---Block {chunk_local_index}---Subchunk {sub_idx}"
            item: Dict[str, Any] = {
                "region_id": str(region_idx),
                "is_region_summary": False,
                "chunk_info": chunk_info,
                "length": count_length_units(full_text),
                "text": full_text,
                "active_chapter_id": active_chapter_id,
                "covered_chapter_ids": covered_chapter_ids,
                "crosses_chapter": crosses_chapter,
                "source_start": source_start,
                "source_end": source_end,
            }
            segments_by_region[region_idx].append(item)

    flat_items: List[Dict[str, Any]] = []

    for region_idx, region in enumerate(h2_regions, start=1):
        h1_outline = region.get("h1_outline", "") or ""
        chunks = region.get("chunks", [])
        if not isinstance(chunks, list):
            chunks = []

        region_active_chapter_id = ""
        region_covered_chapter_ids: List[str] = []
        seen_covered = set()
        for chunk in chunks:
            active = str(chunk.get("active_chapter_id", "") or "").strip()
            if active and not region_active_chapter_id:
                region_active_chapter_id = active
            for chapter_id in chunk.get("covered_chapter_ids", []) or []:
                chapter_id = str(chapter_id or "").strip()
                if chapter_id and chapter_id not in seen_covered:
                    seen_covered.add(chapter_id)
                    region_covered_chapter_ids.append(chapter_id)

        summary_item: Dict[str, Any] = {
            "region_id": str(region_idx),
            "is_region_summary": True,
            "chunk_info": "",
            "length": count_length_units(h1_outline),
            "text": h1_outline,
            "active_chapter_id": region_active_chapter_id,
            "covered_chapter_ids": region_covered_chapter_ids,
            "crosses_chapter": len(region_covered_chapter_ids) > 1,
            "source_start": None,
            "source_end": None,
        }
        flat_items.append(summary_item)
        flat_items.extend(segments_by_region.get(region_idx, []))

    if flat_items:
        lengths = [item.get("length", count_length_units(item.get("text", ""))) for item in flat_items]
        max_length = max(lengths)
        min_length = min(lengths)

    return {
        "split_method": "HiSumChunk",
        "model": os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507").split("/")[-1],
        "max_length": max_length,
        "min_length": min_length,
        "items": flat_items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunking script with English word + Chinese character counting")
    parser.add_argument("--input", default=INPUT_FILE, help="Input JSON file path")
    parser.add_argument("--output", default="", help="Output JSON file path (auto-generated from input filename if not specified)")
    parser.add_argument("--target-len", type=int, default=TARGET_LEN, help="Target chunk length")
    parser.add_argument("--overlap", type=int, default=OVERLAP, help="Overlap length")
    parser.add_argument("--backtrack-range", type=int, default=BACKTRACK_RANGE, help="Backtrack range length")
    return parser.parse_args()


def main() -> None:
    global TARGET_LEN, OVERLAP, BACKTRACK_RANGE

    args = parse_args()
    TARGET_LEN = args.target_len
    OVERLAP = args.overlap
    BACKTRACK_RANGE = args.backtrack_range

    script_dir = Path(__file__).parent.absolute()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / output_path
    else:
        output_path = input_path.parent / f"{input_path.stem}{OUTPUT_SUFFIX}"

    _log("=" * 60)
    _log("Chunking script with English word + Chinese character counting")
    _log(f"Input file: {input_path}")
    _log(f"Output file: {output_path}")
    _log(f"Target length: {TARGET_LEN}")
    _log(f"Overlap: {OVERLAP}")
    _log(f"Backtrack range: {BACKTRACK_RANGE}")
    _log("=" * 60)

    data = load_input(input_path)
    result = build_flat_items(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    items_count = len(result.get("items", []))
    _log(f"Chunking complete, {items_count} items generated.")
    _log(f"max_length = {result.get('max_length', 0)}")
    _log(f"min_length = {result.get('min_length', 0)}")
    _log("=" * 60)


if __name__ == "__main__":
    main()
