#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rule-based (non-LLM) merging of lines that should not have been broken.
These rules only fix "typesetting splits" and never touch "semantically separated line breaks".
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

# ==================== Configurable Parameters ====================
# Input txt file path (relative to script directory, or absolute)
INPUT_FILE = str(EXAMPLE_BOOK_DIR / "Far From the Madding Crowd.txt")

# Output txt file directory (relative to script directory)
OUTPUT_DIR = str(EXAMPLE_BOOK_DIR / "work_txt")

# Output file suffix (between filename and extension)
OUTPUT_SUFFIX = "_MergeRule"

# File encoding
ENCODING = "utf-8"

# Whether to show progress info
SHOW_PROGRESS = True
# ================================================================

import os
import sys
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
EXAMPLE_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))


PAIR_SYMBOLS = [
    ("(", ")"),
    ("[", "]"),
    (""", """),
]

# Regex for detecting number + unit / time / numeric continuation
END_WITH_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?\s*$")
START_WITH_UNIT_OR_NUMBER = re.compile(
    r"^\s*(?:"
    r"[0-9]+(?:\.[0-9]+)?"           # number continuation
    r"|%|‰|°|deg|celsius|sq|m2|m²|m3|m³|km|kV|MW|GW|kg|t|T"
    r")"
)

# Mid-sentence punctuation set (not sentence-ending)
MID_PUNCTUATIONS = (",", ";", ":", "—", "–", "…")


def has_unclosed_pair(line: str) -> bool:
    """Check if current line has unclosed paired symbols"""
    for left, right in PAIR_SYMBOLS:
        if line.count(left) > line.count(right):
            return True
    return False


def is_numeric_unit_split(line: str, next_line: str) -> bool:
    """Check if number/unit/date is split"""
    if not END_WITH_NUMBER.search(line):
        return False
    if START_WITH_UNIT_OR_NUMBER.search(next_line):
        return True
    return False


def is_mid_punctuation_split(line: str) -> bool:
    """Check if mid-sentence punctuation is followed by a line break"""
    stripped = line.rstrip()
    for p in MID_PUNCTUATIONS:
        if stripped.endswith(p):
            # For ellipsis: if single or odd "…" consider incomplete
            if p == "…":
                if stripped.endswith("……"):
                    continue
            return True
    return False


def should_merge(line: str, next_line: str) -> bool:
    """Combined rule to determine whether to remove current line break"""
    if not next_line.strip():
        return False

    # 1) Paired symbols unclosed
    if has_unclosed_pair(line):
        return True

    # 2) Number/unit/date split
    if is_numeric_unit_split(line, next_line):
        return True

    # 3) Mid-sentence punctuation after line break
    if is_mid_punctuation_split(line):
        return True

    return False


def merge_file(input_file: Path, output_dir: Path):
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (input_file.stem + OUTPUT_SUFFIX + input_file.suffix)

    with open(input_file, "r", encoding=ENCODING) as f:
        lines = f.read().splitlines()

    if not lines:
        print("Error: Input file is empty")
        return

    merged_lines = []
    current = lines[0]
    merge_count = 0

    for i in range(len(lines) - 1):
        cur_line = current
        next_line = lines[i + 1]

        if should_merge(cur_line, next_line):
            current = cur_line + next_line
            merge_count += 1
        else:
            merged_lines.append(cur_line)
            current = next_line

    merged_lines.append(current)

    with open(output_file, "w", encoding=ENCODING) as f:
        f.write("\n".join(merged_lines))

    if SHOW_PROGRESS:
        print(f"Processing complete: {input_file.name}")
        print(f"Original line count: {len(lines)}")
        print(f"Line count after merge: {len(merged_lines)}")
        print(f"Removed line breaks: {merge_count}")
        print(f"Output file: {output_file}")


def main():
    script_dir = Path(__file__).parent.absolute()
    if Path(INPUT_FILE).is_absolute():
        input_path = Path(INPUT_FILE)
    else:
        input_path = script_dir / INPUT_FILE
    output_dir = script_dir / OUTPUT_DIR

    merge_file(input_path, output_dir)


if __name__ == "__main__":
    main()
