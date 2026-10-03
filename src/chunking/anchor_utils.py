#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

CHAPTER_ANCHOR_RE = re.compile(r"^<<<CHAPTER_ID:(.*?)>>>\s*$", re.M)
PAGE_MARKER_RE = re.compile(r"^>>>PAGE (\d+)<<<\s*$", re.M)


def strip_chapter_anchors_with_tracking(text: str) -> tuple[str, List[Dict[str, Any]]]:
    anchors: List[Dict[str, Any]] = []
    clean_parts: List[str] = []
    clean_len = 0
    pending_anchor: str | None = None
    anchor_index = 0
    current_page: int | None = None

    for raw_line in text.splitlines(keepends=True):
        page_match = PAGE_MARKER_RE.match(raw_line.strip())
        if page_match:
            current_page = int(page_match.group(1))
            clean_parts.append(raw_line)
            clean_len += len(raw_line)
            continue

        chapter_match = CHAPTER_ANCHOR_RE.match(raw_line.strip())
        if chapter_match:
            pending_anchor = chapter_match.group(1).strip()
            continue

        if pending_anchor is not None and raw_line.strip():
            anchor_index += 1
            anchors.append(
                {
                    "anchor_index": anchor_index,
                    "chapter_id": pending_anchor,
                    "pos": clean_len,
                    "page_no": current_page,
                }
            )
            pending_anchor = None

        clean_parts.append(raw_line)
        clean_len += len(raw_line)

    if pending_anchor is not None:
        anchor_index += 1
        anchors.append(
            {
                "anchor_index": anchor_index,
                "chapter_id": pending_anchor,
                "pos": clean_len,
                "page_no": current_page,
            }
        )

    return "".join(clean_parts), anchors


def _context_candidates(text: str, pos: int) -> List[str]:
    raw = text[pos: pos + 240]
    stripped = raw.lstrip()
    candidates: List[str] = []
    for size in (160, 120, 80, 48, 24):
        piece = stripped[:size].strip()
        if piece and piece not in candidates:
            candidates.append(piece)
    return candidates


def _regex_search_with_flexible_space(haystack: str, needle: str, start_pos: int) -> int:
    compact = re.sub(r"\s+", "", needle)
    if not compact:
        return -1
    pattern = r"\s*".join(re.escape(ch) for ch in compact[:40])
    m = re.search(pattern, haystack[start_pos:], flags=re.M)
    if not m:
        return -1
    return start_pos + m.start()


def remap_anchors(old_text: str, new_text: str, anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    remapped: List[Dict[str, Any]] = []
    cursor = 0
    length_ratio = (len(new_text) / len(old_text)) if old_text else 1.0

    for anchor in sorted(anchors, key=lambda x: int(x.get("pos", 0))):
        old_pos = int(anchor.get("pos", 0))
        estimate = max(cursor, int(old_pos * length_ratio))
        new_pos = -1

        for candidate in _context_candidates(old_text, old_pos):
            new_pos = new_text.find(candidate, cursor)
            if new_pos != -1:
                break
            window_start = max(cursor, estimate - 4000)
            window_end = min(len(new_text), estimate + 4000)
            new_pos = new_text.find(candidate, window_start, window_end)
            if new_pos != -1:
                break
            new_pos = _regex_search_with_flexible_space(new_text, candidate, window_start)
            if new_pos != -1:
                break

        if new_pos == -1:
            new_pos = min(max(cursor, estimate), len(new_text))

        item = dict(anchor)
        item["pos"] = new_pos
        remapped.append(item)
        cursor = new_pos

    return remapped


def save_anchor_file(path: Path, text_file: str, anchors: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"text_file": text_file, "anchors": anchors}, f, ensure_ascii=False, indent=2)


def load_anchor_file(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("anchors", [])


def get_active_and_covered_chapters(
    anchors: List[Dict[str, Any]], start_pos: int, end_pos: int
) -> tuple[str, List[str], bool]:
    sorted_anchors = sorted(anchors, key=lambda x: (int(x.get("pos", 0)), int(x.get("anchor_index", 0))))

    active = ""
    covered: List[str] = []
    for anchor in sorted_anchors:
        pos = int(anchor.get("pos", 0))
        chapter_id = str(anchor.get("chapter_id", "") or "")
        if pos <= start_pos and chapter_id:
            active = chapter_id
        if start_pos <= pos < end_pos and chapter_id:
            covered.append(chapter_id)

    if active and active not in covered:
        covered.insert(0, active)

    deduped: List[str] = []
    for chapter_id in covered:
        if chapter_id and chapter_id not in deduped:
            deduped.append(chapter_id)

    return active, deduped, len(deduped) > 1


def annotate_chunks_by_search(
    chunks: List[Dict[str, Any]], current_text: str, anchors: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    search_cursor = 0

    for chunk in chunks:
        text = str(chunk.get("text", "") or "")
        start_pos = current_text.find(text, search_cursor)
        if start_pos == -1:
            start_pos = current_text.find(text)
        if start_pos == -1:
            start_pos = search_cursor
        end_pos = start_pos + len(text)
        active, covered, crosses = get_active_and_covered_chapters(anchors, start_pos, end_pos)

        new_chunk = dict(chunk)
        new_chunk["source_start"] = start_pos
        new_chunk["source_end"] = end_pos
        new_chunk["active_chapter_id"] = active
        new_chunk["covered_chapter_ids"] = covered
        new_chunk["crosses_chapter"] = crosses
        annotated.append(new_chunk)

        search_cursor = max(search_cursor, start_pos)
        if end_pos > search_cursor:
            search_cursor = end_pos

    return annotated


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
