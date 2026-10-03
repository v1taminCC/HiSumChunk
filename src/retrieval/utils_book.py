from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from config import (
    BOOKSUM_BOOK_TXT_DIR, BOOKSUM_SPLIT_JSON_DIR,
    FIXED_SPLIT_JSON_DIR, FIXED_SPLIT_SUFFIX,
    LUMBER_SPLIT_JSON_DIR, LUMBER_SPLIT_SUFFIX,
    LBD_SPLIT_JSON_DIR, LBD_SPLIT_SUFFIX,
    SEMANTIC_SPLIT_JSON_DIR, SEMANTIC_SPLIT_SUFFIX,
    RAPTOR_SPLIT_JSON_DIR, RAPTOR_SPLIT_SUFFIX,
    SPLIT_GLOB,
)


PIPELINE_MARKERS = [
    "_MergeRule",
    "_NoTitle",
    "_NoHeaderFooter",
    "_MergeLineBreak",
    "_Level",
    "_L2Outline",
    "_MergeH2",
    "_L1Outline",
    "_Fixed1024W",
    "_NoOverlap",
]


def extract_book_title(book_id: Any) -> str:
    s = "" if book_id is None else str(book_id).strip()
    patterns = [
        r"\.book\s*[ivxlcdm\d]+\s*\.chapter\s*[ivxlcdm\d]+\s*-\s*chapter\s*[ivxlcdm\d]+.*$",
        r"\.book\s*[ivxlcdm\d]+\s*\.chapter\s*[ivxlcdm\d]+.*$",
        r"\.chapters?\s*[ivxlcdm\d]+\s*-\s*[ivxlcdm\d]+.*$",
        r"\.chapter\s*[ivxlcdm\d]+.*$",
        r"\.book\s*[ivxlcdm\d]+.*$",
        r"\.part\s*[ivxlcdm\d]+.*$",
        r"\.act\s*[ivxlcdm\d]+.*$",
        r"\.scene\s*[ivxlcdm\d]+.*$",
        r"\.epilogue.*$",
    ]
    for p in patterns:
        s = re.sub(p, "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s


def strip_split_suffix(filename: str) -> str:
    name = filename
    if name.endswith("_split.json"):
        name = name[:-11]
    elif name.endswith(".json"):
        name = name[:-5]
    for marker in PIPELINE_MARKERS:
        pos = name.find(marker)
        if pos != -1:
            name = name[:pos]
            break
    return name.strip()


def list_book_titles() -> list[str]:
    if BOOKSUM_BOOK_TXT_DIR.exists():
        return sorted(p.stem for p in BOOKSUM_BOOK_TXT_DIR.glob("*.txt") if p.is_file())
    return sorted({strip_split_suffix(p.name) for p in BOOKSUM_SPLIT_JSON_DIR.glob(SPLIT_GLOB) if p.is_file()})


def find_booksum_split_file(book_title: str) -> Path:
    matches = sorted(BOOKSUM_SPLIT_JSON_DIR.glob(f"{book_title}_*_split.json"))
    if not matches:
        raise FileNotFoundError(f"HiSumChunk split file not found: {book_title}")
    if len(matches) > 1:
        exact = [p for p in matches if strip_split_suffix(p.name) == book_title]
        if exact:
            return exact[0]
    return matches[0]


def find_fixed_split_file(book_title: str) -> Path:
    path = FIXED_SPLIT_JSON_DIR / f"{book_title}{FIXED_SPLIT_SUFFIX}"
    if path.exists():
        return path
    matches = sorted(FIXED_SPLIT_JSON_DIR.glob(f"{book_title}_*_split.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Fixed split file not found: {book_title}")


def find_lumber_split_file(book_title: str) -> Path:
    path = LUMBER_SPLIT_JSON_DIR / f"{book_title}{LUMBER_SPLIT_SUFFIX}"
    if path.exists():
        return path
    matches = sorted(LUMBER_SPLIT_JSON_DIR.glob(f"{book_title}_*_split.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"LumberChunker split file not found: {book_title}")


def find_lbd_split_file(book_title: str) -> Path:
    path = LBD_SPLIT_JSON_DIR / f"{book_title}{LBD_SPLIT_SUFFIX}"
    if path.exists():
        return path
    matches = sorted(LBD_SPLIT_JSON_DIR.glob(f"{book_title}_*_split.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"LLMSemantic split file not found: {book_title}")


def find_semantic_split_file(book_title: str) -> Path:
    path = SEMANTIC_SPLIT_JSON_DIR / f"{book_title}{SEMANTIC_SPLIT_SUFFIX}"
    if path.exists():
        return path
    matches = sorted(SEMANTIC_SPLIT_JSON_DIR.glob(f"{book_title}_*_split.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"SemanticEmbed split file not found: {book_title}")


def find_raptor_nodes_file(book_title: str) -> Path:
    path = RAPTOR_SPLIT_JSON_DIR / f"{book_title}{RAPTOR_SPLIT_SUFFIX}"
    if path.exists():
        return path
    matches = sorted(RAPTOR_SPLIT_JSON_DIR.glob(f"{book_title}*_RAPTOR_nodes.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"RAPTOR nodes file not found: {book_title}")
