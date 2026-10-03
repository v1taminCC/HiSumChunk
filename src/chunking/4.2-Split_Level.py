#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split single text into chunks and detect level-1 / level-2 titles.
Uses SGLang for concurrent model calls.
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
# ================================ Configurable Parameters =======================================
# ================================================================================================

# ------------------------------- Model Configuration -------------------------------
MODEL_PATH: str = os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507")
RUNTIME_ENDPOINT: str = "http://127.0.0.1:30000"

# ---------------------------- Generation Parameter Config ---------------------------
TEMPERATURE: float = 0.1
TOP_P: float = 0.5
TOP_K = 15
MAX_GENERATE_TOKENS: int = 16000
MAX_INPUT_LENGTH: int = 50000

# ------------------------------ System Prompt Config --------------------------------
SYSTEM_PROMPT: str = ""

# ---------------------------------- Chunk Config ------------------------------------
CHUNK_SIZE: int = 6000
# Sentence-ending punctuation (forward search for split points), adjust as needed
SENTENCE_END_PUNCTUATIONS = ['!', '?', ';', '.']
MERGE_THRESHOLD: int = 50

# ----------------------------- Parallel Processing Config ---------------------------
BATCH_SIZE: int = 4

# ----------------------------------- File Config ------------------------------------
INPUT_FILE: str = str(EXAMPLE_BOOK_DIR / "work_txt" / "Far From the Madding Crowd_MergeRule_Title_NoHeaderFooter.txt")
OUTPUT_SUFFIX: str = "_Level.json"
# Output directory (relative to script directory)
OUTPUT_DIR: str = str(EXAMPLE_BOOK_DIR)

# ------------------------------- Prompt Template ------------------------------------
PROMPT_TEMPLATE: str = """
Identify all level-1 and level-2 titles in the following text and output them sequentially in JSON.
Given text:
"{text}"
Return the following JSON format:
{{
    "has_titles": "yes or no",
    "title_1": "title content, empty string if not present",
    "title_2": "title content, empty string if not present",
    "title_3": "title content, empty string if not present",
    "title_4": "title content, empty string if not present",
    If there are more titles, continue until none remain.
    "reasoning": "Why you think the above titles exist or not, 500 words explanation"
}}
"""

# ---------------------------------- Other Config ------------------------------------
SEPARATOR_LENGTH: int = 60
SHOW_PROGRESS: bool = True
# Whether to print final chunk contents in terminal (may be long)
PRINT_FINAL_SEGMENTS: bool = True

# ================================================================================================
# ========================================== Code =================================================
# ================================================================================================

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
EXAMPLE_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))
from typing import List, Tuple

from sglang import (
    RuntimeEndpoint,
    assistant,
    function,
    gen,
    set_default_backend,
    system,
    user,
)

_backend_initialized = False


def _init_sglang_backend():
    """Initialize SGLang runtime connection (once)"""
    global _backend_initialized
    if _backend_initialized:
        return
    if SHOW_PROGRESS:
        print("=" * SEPARATOR_LENGTH)
        print(f"Connecting to SGLang runtime: {RUNTIME_ENDPOINT}")
    try:
        set_default_backend(RuntimeEndpoint(RUNTIME_ENDPOINT))
    except Exception as exc:
        print("Failed to connect to SGLang runtime. Please ensure the server is started, e.g.:")
        print(f"  python -m sglang.launch_server --model-path {MODEL_PATH} --port 30000")
        print("Detailed error:")
        print(exc)
        raise
    if SHOW_PROGRESS:
        print("SGLang runtime connection successful!")
        print("=" * SEPARATOR_LENGTH)
    _backend_initialized = True


@function
def _sglang_single_chat(s, prompt: str, max_tokens: int):
    """SGLang single chat function (using @function decorator)"""
    truncated_prompt = prompt[:MAX_INPUT_LENGTH] if len(prompt) > MAX_INPUT_LENGTH else prompt

    s += system(SYSTEM_PROMPT)
    s += user(truncated_prompt)
    gen_kwargs = {"max_tokens": max_tokens}
    if TEMPERATURE is not None:
        gen_kwargs["temperature"] = TEMPERATURE
    if TOP_P is not None:
        gen_kwargs["top_p"] = TOP_P
    if TOP_K is not None:
        gen_kwargs["top_k"] = TOP_K
    s += assistant(gen("response", **gen_kwargs))


def call_model(prompt: str, max_new_tokens: int) -> str:
    """Call model for Q&A (no memory)"""
    _init_sglang_backend()
    generate_max_tokens = max_new_tokens if max_new_tokens is not None else MAX_GENERATE_TOKENS
    try:
        res = _sglang_single_chat(prompt=prompt, max_tokens=generate_max_tokens)
        if isinstance(res, dict):
            raw_answer = res.get("response", "")
        elif hasattr(res, "response"):
            raw_answer = res.response
        elif hasattr(res, "__getitem__"):
            try:
                raw_answer = res["response"]
            except (KeyError, TypeError):
                raw_answer = str(res)
        elif isinstance(res, str):
            raw_answer = res
        else:
            raw_answer = str(res)

        if not isinstance(raw_answer, str):
            raw_answer = str(raw_answer)

        # Output raw model response (no processing)
        print(f"[Raw model output] {raw_answer}", flush=True)

        answer = raw_answer

        answer = answer.strip() if answer else ""
        if "USER:" in answer:
            answer = answer.split("USER:")[0].strip()
        if "ASSISTANT:" in answer and answer.count("ASSISTANT:") > 1:
            parts = answer.split("ASSISTANT:")
            if len(parts) > 1:
                answer = parts[1].strip()
                if "USER:" in answer:
                    answer = answer.split("USER:")[0].strip()

        return answer
    except Exception as exc:
        print(f"[Error] Model call failed: {exc}")
        return ""


def split_by_length_with_punct(text: str) -> List[str]:
    """Split text by length + sentence-ending punctuation"""
    if not text:
        return []
    segments: List[str] = []
    length = len(text)
    start = 0
    while start < length:
        target = min(start + CHUNK_SIZE, length)
        if target >= length:
            segments.append(text[start:].strip())
            break

        # Look backward for nearest sentence-ending punctuation
        cut_pos = -1
        for i in range(target, start, -1):
            if text[i - 1] in SENTENCE_END_PUNCTUATIONS:
                cut_pos = i
                break

        if cut_pos == -1:
            cut_pos = target

        segment = text[start:cut_pos].strip()
        if segment:
            segments.append(segment)
        start = cut_pos
    return segments


def build_prompt(segment: str) -> str:
    """Build prompt"""
    return PROMPT_TEMPLATE.format(text=segment)


def judge_segments_batch(segments: List[str]) -> List[str]:
    """Concurrently evaluate all chunks"""
    _init_sglang_backend()
    prompts = [build_prompt(seg) for seg in segments]
    results = [None] * len(prompts)
    total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        batch_indices = list(range(i, min(i + BATCH_SIZE, len(prompts))))
        batch_num = i // BATCH_SIZE + 1
        if SHOW_PROGRESS:
            print(f"  Batch progress: {batch_num}/{total_batches} batches - processing concurrently...", flush=True)

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            future_to_index = {
                executor.submit(call_model, prompt=prompt, max_new_tokens=MAX_GENERATE_TOKENS): idx
                for idx, prompt in zip(batch_indices, batch_prompts)
            }
            batch_results = {}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    ans = future.result()
                    batch_results[idx] = ans
                except Exception as exc:
                    if SHOW_PROGRESS:
                        print(f"    [Error] Call failed for index {idx} in batch {batch_num}: {exc}", flush=True)
                    batch_results[idx] = ""

            for idx in batch_indices:
                results[idx] = batch_results.get(idx, "")
    return results


def _extract_json_from_text(text: str) -> str:
    """
    Extract a complete JSON object from text (matching paired braces)
    """
    if not text:
        return ""
    
    # Find the first {
    start_idx = text.find('{')
    if start_idx == -1:
        return ""
    
    # Starting from the first {, match paired braces
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(start_idx, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found complete JSON object
                    return text[start_idx:i+1]
    
    return ""


def parse_result(model_response: str) -> Tuple[dict, bool]:
    """
    Strictly parse model response, returns (result dict, is_format_valid)
    """
    result = {
        "reasoning": "",
        "title_list": [],   # All titles in this chunk, in order from the model (empty strings removed)
    }
    ok = False

    if not model_response:
        return result, ok

    try:
        # Extract complete JSON object
        json_str = _extract_json_from_text(model_response)
        if not json_str:
            return result, ok
        
        # Strict JSON parsing, no tolerance
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return result, ok
        
        # Process "reasoning"
        if "reasoning" in data:
            result["reasoning"] = str(data.get("reasoning", "") or "").strip()

        # Extract "title_1, title_2..." list from JSON (by number order, remove empty)
        title_items = []
        for k, v in data.items():
            if re.match(r"^title_\d+$", str(k)):
                v_str = str(v or "").strip()
                if v_str:
                    title_items.append((k, v_str))
        
        if title_items:
            title_items.sort(
                key=lambda x: int(re.findall(r"\d+", x[0])[0])
                if re.findall(r"\d+", x[0])
                else 999999
            )
            result["title_list"] = [v for _, v in title_items]
        
        # Consider format valid if at least one title or reasoning is non-empty
        ok = bool(result["title_list"] or result["reasoning"])
        return result, ok
    except Exception:
        # Parse failed, return False directly
        pass

    return result, ok


def _build_flexible_title_pattern(title: str) -> str:
    """
    Build a "whitespace-ignoring" title matching regex:
    - Model may output "II. Bidding Documents", while original text is "II.  Bid  ding  Documents"
    - May also have full-width spaces, multiple spaces, newlines
    Strategy: first remove all whitespace from title, then insert \\s* between each char for matching.
    """
    compact = re.sub(r"\s+", "", (title or "").strip())
    if not compact:
        return ""
    return r"\s*".join(re.escape(ch) for ch in compact)


def find_title_pos(text: str, title: str) -> int:
    """Find the position of title in text (ignoring whitespace differences), return start index, -1 if not found."""
    if not text or not title:
        return -1

    # 1) Try exact match first (fastest, avoids false matches)
    pos = text.find(title)
    if pos != -1:
        return pos

    # 2) Flexible match: ignore whitespace differences
    pattern = _build_flexible_title_pattern(title)
    if not pattern:
        return -1
    m = re.search(pattern, text, flags=re.MULTILINE)
    return m.start() if m else -1


def split_segment_by_title(text: str, title: str) -> List[str]:
    """Split by title string, keep title at the beginning of the segment"""
    title = title.strip()
    if not title:
        return [text]

    pattern = re.escape(title)
    parts = re.split(pattern, text)
    if len(parts) == 1:
        return [text]

    segments = []
    first = parts[0].strip()
    if first:
        segments.append(first)

    for piece in parts[1:]:
        combined = f"{title}{piece}"
        combined = combined.strip()
        if combined:
            segments.append(combined)

    return segments if segments else [text]


def format_duration(seconds: float) -> str:
    """Format seconds into readable string"""
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h:02d}h-{m:02d}m-{s:02d}s-{ms:03d}ms"


def strip_title_prefix(segment: str, titles: List[str]) -> str:
    """
    Remove title prefix from the beginning of segment (ignoring whitespace differences).
    - If segment becomes only whitespace/symbols, return empty string (discard).
    - Only processes the first matched title.
    """
    if not segment:
        return ""
    for title in titles:
        pattern = _build_flexible_title_pattern(title)
        if not pattern:
            continue
        m = re.match(rf"^\s*{pattern}", segment, flags=re.MULTILINE)
        if m:
            # Remove title portion, keep the rest
            remainder = segment[m.end():].lstrip()
            return remainder
    return segment


def main():
    if SHOW_PROGRESS:
        print("=" * SEPARATOR_LENGTH)
        print("Title Splitting System")
        print("=" * SEPARATOR_LENGTH)

    script_dir = Path(__file__).parent.absolute()
    input_file_path = Path(INPUT_FILE)
    if not input_file_path.is_absolute():
        input_file_path = script_dir / INPUT_FILE
    if not input_file_path.exists():
        print(f"[Error] File not found: {input_file_path}")
        sys.exit(1)

    # Output written to the specified directory
    output_dir_path = script_dir / OUTPUT_DIR
    output_file_path = output_dir_path / (input_file_path.stem + OUTPUT_SUFFIX)
    if SHOW_PROGRESS:
        print(f"Input file: {input_file_path}")
        print(f"Output file: {output_file_path}")
        print("=" * SEPARATOR_LENGTH)

    try:
        _init_sglang_backend()
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("\nPlease start the SGLang server first, e.g.:")
        print(f"python -m sglang.launch_server --model-path {MODEL_PATH} --port 30000")
        sys.exit(1)

    # Read file
    if SHOW_PROGRESS:
        print("Reading file and splitting into chunks...")
    with open(input_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    segments = split_by_length_with_punct(text)
    if not segments:
        print("[Error] No chunk content.")
        sys.exit(1)
    if SHOW_PROGRESS:
        print(f"Generated {len(segments)} chunks, each target length ~{CHUNK_SIZE} chars")
        print("=" * SEPARATOR_LENGTH)

    # First round: concurrent model calls to extract titles
    if SHOW_PROGRESS:
        print("Starting batch concurrent title extraction...")
        print("-" * SEPARATOR_LENGTH)
    start_time = time.time()
    raw_results = judge_segments_batch(segments)
    duration = time.time() - start_time
    if SHOW_PROGRESS:
        print("\nFirst round complete, parsing results...")
        print(f"Total evaluation time: {format_duration(duration)}")
        print("=" * SEPARATOR_LENGTH)

    # First parse round: collect success and failure indices
    parsed_results: List[dict] = [None] * len(raw_results)
    failed_indices = []
    
    for idx, res in enumerate(raw_results):
        parsed, ok = parse_result(res)
        if ok:
            parsed_results[idx] = parsed
        else:
            failed_indices.append(idx)
            if SHOW_PROGRESS:
                print(f"[Warning] Chunk {idx + 1} format invalid, will retry once")
    
    # Retry format-invalid ones (only once)
    if failed_indices:
        if SHOW_PROGRESS:
            print(f"\nFound {len(failed_indices)} chunks with invalid format, retrying (once each)...")
        
        for idx in failed_indices:
            if SHOW_PROGRESS:
                print(f"  [Retry] Chunk {idx + 1}...")
            
            # Regenerate prompt and call model
            retry_prompt = build_prompt(segments[idx])
            retry_response = call_model(retry_prompt, MAX_GENERATE_TOKENS)
            parsed, ok = parse_result(retry_response)
            
            if ok:
                parsed_results[idx] = parsed
                if SHOW_PROGRESS:
                    print(f"  [Success] Chunk {idx + 1} retry succeeded")
            else:
                # Still failed after retry, use default values
                parsed_results[idx] = {
                    "reasoning": "",
                    "title_list": []
                }
                if SHOW_PROGRESS:
                    print(f"  [Skipped] Chunk {idx + 1} still failed after retry, using defaults")

    # ----------------------------------------
    # Aggregate "title_list" from all chunk model responses (deduplicate, preserve order)
    # ----------------------------------------
    all_titles_ordered: List[str] = []
    seen_titles = set()
    for parsed in parsed_results:
        if parsed:
            local_titles = parsed.get("title_list") or []
            for t in local_titles:
                t = str(t).strip()
                if not t:
                    continue
                if t not in seen_titles:
                    seen_titles.add(t)
                    all_titles_ordered.append(t)

    if SHOW_PROGRESS:
        print("\n" + "=" * SEPARATOR_LENGTH)
        print(f"Titles aggregated from model (deduplicated, total {len(all_titles_ordered)}):")
        for idx, t in enumerate(all_titles_ordered, 1):
            print(f"  [{idx}] {t}")
        print("=" * SEPARATOR_LENGTH)

    # ----------------------------------------
    # Split the full original text by titles sequentially (starting from single-element full text list)
    # ----------------------------------------
    pieces: List[str] = [text]
    for title in all_titles_ordered:
        found = False
        for i, piece in enumerate(pieces):
            pos = find_title_pos(piece, title)
            if pos != -1:
                # Split at this position: content before title, title and everything after in next piece
                before = piece[:pos]
                after = piece[pos:]

                new_pieces: List[str] = []
                # First part of original list
                new_pieces.extend(pieces[:i])
                # Content before title (if any) kept as a new segment
                if before:
                    new_pieces.append(before)
                # Title and its following content as a new segment
                new_pieces.append(after)
                # Rest of original list
                new_pieces.extend(pieces[i + 1 :])

                pieces = new_pieces
                found = True
                break

        if not found:
            # Current title not matched in any piece, print for later manual review
            print(f"[Title not matched] {title}", flush=True)

    # Remove empty/whitespace-only segments
    final_segments: List[str] = [p for p in pieces if p and p.strip()]

    # Remove title-only segments and strip title text from segment start
    cleaned_segments: List[str] = []
    for seg in final_segments:
        body = strip_title_prefix(seg, all_titles_ordered)
        # If after removing title the segment is empty (or only whitespace/symbols), discard
        if body and body.strip():
            cleaned_segments.append(body.strip())

    # Merge segments shorter than threshold into previous segment
    merged_segments: List[str] = []
    for i, seg in enumerate(cleaned_segments):
        if len(seg) < MERGE_THRESHOLD and merged_segments:
            # Merge into previous segment
            merged_segments[-1] = merged_segments[-1] + seg
        else:
            # Keep current segment
            merged_segments.append(seg)
    cleaned_segments = merged_segments

    # Count characters
    chunks_with_meta = [
        {"length": len(chunk), "text": chunk}
        for chunk in cleaned_segments
    ]
    max_length = max((s["length"] for s in chunks_with_meta), default=0)
    min_length = min((s["length"] for s in chunks_with_meta), default=0)

    if SHOW_PROGRESS and PRINT_FINAL_SEGMENTS:
        print("\n" + "=" * SEPARATOR_LENGTH)
        print(f"Final segment list content (total {len(chunks_with_meta)} segments):")
        print("=" * SEPARATOR_LENGTH)
        for idx, seg in enumerate(chunks_with_meta, 1):
            print("=" * SEPARATOR_LENGTH)
            print(f"[Segment {idx}/{len(chunks_with_meta)}] length={seg['length']}")
            print(seg["text"])
        print("=" * SEPARATOR_LENGTH)

    # Save as JSON object with global max/min character counts and chunk details
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "max_length": max_length,
        "min_length": min_length,
        "chunks": chunks_with_meta,
    }
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    if SHOW_PROGRESS:
        print("Processing complete!")
        print(f"Final segment count: {len(cleaned_segments)}")
        print("=" * SEPARATOR_LENGTH)
        print(f"Results saved to: {output_file_path}")


if __name__ == "__main__":
    main()
