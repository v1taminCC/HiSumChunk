#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-round parallel detection and merging of line breaks caused by PDF-to-txt conversion.

Requirements:
1) Code style similar to 3-clear_noise.py (SGLang RuntimeEndpoint + concurrent batch processing + log format).
2) Read txt, split by \n, filter empty lines first.
3) For adjacent lines (1,2), (2,3), ... parallel model calls to judge if they belong to the same sentence ("yes" means merge).
4) Apply merges sequentially: if (1,2)=yes and (2,3)=yes, then merge 1,2,3.
5) If a round has merges, continue next round with the merged list.
6) "no" boundaries are permanently skipped: boundaries that didn't merge in previous rounds won't be judged again; if A+B merged and B+C is no, then AB+C is also not judged.

New restriction:
- If the end of {first_segment} is a sentence-ending punctuation, skip model judgment, directly treat as no-merge.
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
MODEL_PATH: str = os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507")  # Model path (for startup hint only)
RUNTIME_ENDPOINT: str = "http://127.0.0.1:30000"            # SGLang runtime address

# ---------------------------- Generation Parameter Config ---------------------------
TEMPERATURE: float = 0.1
TOP_P: float = 0.5
TOP_K = 15
MAX_GENERATE_TOKENS: int = 16000
MAX_INPUT_LENGTH: int = 50000

# ------------------------------ System Prompt Config --------------------------------
SYSTEM_PROMPT: str = ""

# ----------------------------- Parallel Processing Config ---------------------------
BATCH_SIZE: int = 16  # Concurrent request count (as in your log example)

# ----------------------------------- File Config ------------------------------------
INPUT_FILE: str = str(EXAMPLE_BOOK_DIR / "work_txt" / "Far From the Madding Crowd_MergeRule_Title_NoHeaderFooter.txt")  # Relative to script directory
OUTPUT_SUFFIX: str = "_MergeLineBreak.json"
OUTPUT_ENCODING: str = "utf-8"

# Output directory (relative to script directory; if empty, defaults to input file directory)
OUTPUT_DIR: str = str(EXAMPLE_BOOK_DIR)

# ------------------------------- Prompt Template ------------------------------------
PROMPT_TEMPLATE: str = """
The following two segments are broken sentences caused by PDF-to-txt conversion line breaks (they may or may not belong together). Please judge whether these two text segments can be merged.
Rule: Whether merging would be natural (as long as the logical connection at the join point is coherent).
Given text separated by =====:
=====First segment=====
"{first_segment}"
=====Second segment=====
"{second_segment}"
Return the following JSON format:
{{
    "reasoning": "Why you think they can or cannot be merged, 300 word explanation",
    "answer": "yes or no"
}}
""".strip()

# --------------------------- Sentence End Punctuation Config ------------------------
# Define sentence-ending punctuation set, adjust as needed
SENTENCE_END_PUNCTUATIONS = {'.'}
# ---------------------------------- Other Config ------------------------------------
SEPARATOR_LENGTH: int = 60
SHOW_PROGRESS: bool = True

# ================================================================================================
# ================================================================================================

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
EXAMPLE_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def call_model(prompt: str, max_new_tokens: Optional[int] = None) -> str:
    """Call model (no context memory)"""
    _init_sglang_backend()
    generate_max_tokens = max_new_tokens if max_new_tokens is not None else MAX_GENERATE_TOKENS
    res = _sglang_single_chat(prompt=prompt, max_tokens=generate_max_tokens)
    if isinstance(res, dict):
        return str(res.get("response", "") or "").strip()
    if hasattr(res, "response"):
        return str(res.response or "").strip()
    if hasattr(res, "__getitem__"):
        try:
            return str(res["response"] or "").strip()
        except Exception:
            return str(res).strip()
    return str(res).strip()


def format_prompt(first_segment: str, second_segment: str) -> str:
    return PROMPT_TEMPLATE.format(first_segment=first_segment, second_segment=second_segment)


def truncate_segments(first_segment: str, second_segment: str, max_len: int = 2000) -> Tuple[str, str]:
    """
    Truncation mechanism to prevent exceeding token limit:
    - First segment: keep last max_len characters
    - Second segment: keep first max_len characters
    """
    first_trunc = first_segment[-max_len:] if len(first_segment) > max_len else first_segment
    second_trunc = second_segment[:max_len] if len(second_segment) > max_len else second_segment
    return first_trunc, second_trunc


def judge_batches(prompts: List[str]) -> List[str]:
    """Parallel model calls, preserving order (style matching 3-clear_noise.py)"""
    _init_sglang_backend()
    results: List[Optional[str]] = [None] * len(prompts)
    total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(prompts), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch_prompts = prompts[i:i + BATCH_SIZE]
        batch_indices = list(range(i, min(i + BATCH_SIZE, len(prompts))))
        if SHOW_PROGRESS:
            print(f"  Batch progress: {batch_num}/{total_batches} batches - processing concurrently...", flush=True)
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            future_to_index = {
                executor.submit(call_model, prompt=p, max_new_tokens=MAX_GENERATE_TOKENS): idx
                for idx, p in zip(batch_indices, batch_prompts)
            }
            batch_results: Dict[int, str] = {}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    batch_results[idx] = future.result()
                except Exception:
                    batch_results[idx] = ""
            for idx in batch_indices:
                results[idx] = batch_results.get(idx, "")
    return [r or "" for r in results]


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


def parse_answer_no_tolerance(model_response: str) -> Tuple[str, bool]:
    """
    Strictly parse the standard JSON format returned by the model:
    {
        "reasoning": "...",
        "answer": "yes" or "no"
    }
    Returns (answer, valid):
      - answer: "yes"/"no", defaults to "no" on parse failure
      - valid:   bool, whether format strictly meets requirements (can be json.loads'd and contains valid answer)
    """
    try:
        # Extract complete JSON object
        json_str = _extract_json_from_text(model_response)
        if not json_str:
            return "no", False
        
        # Strict JSON parsing, no tolerance
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return "no", False
        
        # Strictly check required fields
        if "answer" not in data:
            return "no", False
        
        # Extract value
        ans = str(data.get("answer", "no") or "no").strip()
        if ans not in ("yes", "no"):
            return "no", False
        
        return ans, True
    except Exception:
        # Parse failed, return False directly
        return "no", False


@dataclass
class Segment:
    """A segment: composed of consecutive lines from original range [start, end] (1-based), text is these lines merged"""

    start: int
    end: int
    text: str


def _merge_text(left: str, right: str) -> str:
    """Merge two segment texts: remove line break (core purpose)"""
    return (left or "") + (right or "")


def read_lines_filter_empty(file_path: Path) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()
    # Filter empty lines (before all functionality)
    return [ln for ln in raw if ln.strip() != ""]


def merge_iteratively(lines: List[str], record_file) -> Tuple[List[Segment], int, float]:
    """
    Multi-round iterative merging:
    - Each round: judge all adjacent segment boundaries, apply merges sequentially
    - If any round has merges, enter next round; next round re-judges all adjacent boundaries
    - record_file: file object for recording model responses
    """
    segments: List[Segment] = [Segment(i + 1, i + 1, lines[i]) for i in range(len(lines))]
    total_merge_times = 0
    judgment_count = 0  # Judgment group counter
    blocked_no_boundaries: Set[int] = set()  # Permanently skipped "no" boundaries (including format-failed defaults)

    if SHOW_PROGRESS:
        print("Starting multi-round iterative merging (each round: batch parallel judgment + sequential merge application)...")

    start_time = time.time()

    while True:
        if len(segments) <= 1:
            break

        # Generate boundaries to judge: boundary_id = segments[i].end (original boundary)
        pending: List[Tuple[int, int]] = []  # (segment_index i, boundary_id)
        prompts: List[str] = []
        segment_pairs: List[Tuple[str, str]] = []  # Record each pair of text segments
        decisions: Dict[int, str] = {}  # Initialize early, store skipped boundary judgment results
        
        for i in range(len(segments) - 1):
            boundary_id = segments[i].end
            first_text = segments[i].text
            second_text = segments[i + 1].text

            # Boundaries already judged as permanently no-merge, skip directly in subsequent rounds
            if boundary_id in blocked_no_boundaries:
                decisions[boundary_id] = "no"
                continue
            
            # Core modification: check if first segment ends with sentence-ending punctuation
            # 1. Strip trailing whitespace chars (spaces, newlines, etc.)
            stripped_first = first_text.strip()
            # 2. Check non-empty and last char is sentence-ending punctuation
            if stripped_first and stripped_first[-1] in SENTENCE_END_PUNCTUATIONS:
                # Ends with sentence-ending punctuation, skip model judgment, directly mark as "no"
                decisions[boundary_id] = "no"
                continue
            
            # Non-sentence-ending punctuation ending, add to pending judgment list
            pending.append((i, boundary_id))
            first_trunc, second_trunc = truncate_segments(first_text, second_text)
            segment_pairs.append((first_text, second_text))
            prompts.append(format_prompt(first_trunc, second_trunc))

        # If no judgeable boundaries in this round, terminate
        if not prompts and not decisions:
            break

        if SHOW_PROGRESS:
            print("-" * SEPARATOR_LENGTH)

        # Only batch call for prompts needing model judgment
        if prompts:
            responses = judge_batches(prompts)
            
            # First parse round: collect success and failure indices
            pending_results = [None] * len(pending)
            failed_indices = []
            
            for idx, ((i, boundary_id), resp, (first_text, second_text)) in enumerate(zip(pending, responses, segment_pairs)):
                ans, valid = parse_answer_no_tolerance(resp)
                if valid:
                    pending_results[idx] = ans
                    decisions[boundary_id] = ans
                    if ans == "no":
                        blocked_no_boundaries.add(boundary_id)
                    
                    # Only record groups where answer is "no"
                    if ans == "no":
                        judgment_count += 1
                        record_file.write(f"========== Judgment Group {judgment_count} ==========\n")
                        record_file.write("First segment text:\n")
                        record_file.write(f"{first_text}\n\n")
                        record_file.write("Second segment text:\n")
                        record_file.write(f"{second_text}\n\n")
                        record_file.write("Model response:\n")
                        record_file.write(f"{resp}\n\n\n")
                        record_file.flush()
                else:
                    failed_indices.append(idx)
                    if SHOW_PROGRESS:
                        print(f"[Warning] Boundary {boundary_id} format invalid, will retry once")
            
            # Retry format-invalid ones (only once)
            if failed_indices:
                if SHOW_PROGRESS:
                    print(f"  Found {len(failed_indices)} boundaries with invalid format, retrying (once each)...")
                
                for idx in failed_indices:
                    (i, boundary_id), _, (first_text, second_text) = pending[idx], responses[idx], segment_pairs[idx]
                    if SHOW_PROGRESS:
                        print(f"  [Retry] Boundary {boundary_id}...")
                    
                    # Regenerate prompt and call model
                    first_trunc, second_trunc = truncate_segments(first_text, second_text)
                    retry_prompt = format_prompt(first_trunc, second_trunc)
                    retry_response = call_model(retry_prompt, MAX_GENERATE_TOKENS)
                    ans, valid = parse_answer_no_tolerance(retry_response)
                    
                    if valid:
                        pending_results[idx] = ans
                        decisions[boundary_id] = ans
                        if ans == "no":
                            blocked_no_boundaries.add(boundary_id)
                        if SHOW_PROGRESS:
                            print(f"  [Success] Boundary {boundary_id} retry succeeded")
                        
                        # Only record groups where answer is "no"
                        if ans == "no":
                            judgment_count += 1
                            record_file.write(f"========== Judgment Group {judgment_count} ==========\n")
                            record_file.write("First segment text:\n")
                            record_file.write(f"{first_text}\n\n")
                            record_file.write("Second segment text:\n")
                            record_file.write(f"{second_text}\n\n")
                            record_file.write("Model response:\n")
                            record_file.write(f"{retry_response}\n\n\n")
                            record_file.flush()
                    else:
                        # Still failed after retry, use default "no"
                        pending_results[idx] = "no"
                        decisions[boundary_id] = "no"
                        blocked_no_boundaries.add(boundary_id)
                        if SHOW_PROGRESS:
                            print(f"  [Skipped] Boundary {boundary_id} still failed after retry, using default \"no\"")
            
            invalid_exists = False
        else:
            invalid_exists = False  # No model calls, naturally no invalid responses

        # Apply merges sequentially
        new_segments: List[Segment] = []
        merged_any = False
        i = 0
        while i < len(segments):
            if i == len(segments) - 1:
                new_segments.append(segments[i])
                break

            boundary_id = segments[i].end
            ans = decisions.get(boundary_id, "no")
            if ans == "yes":
                # Merge i and i+1
                merged = Segment(
                    start=segments[i].start,
                    end=segments[i + 1].end,
                    text=_merge_text(segments[i].text, segments[i + 1].text),
                )
                new_segments.append(merged)
                total_merge_times += 1
                merged_any = True
                i += 2
            else:
                new_segments.append(segments[i])
                i += 1

        segments = new_segments

        # If no merge in this round and no invalid responses needing retry, terminate
        if not merged_any and not invalid_exists:
            break

    duration = time.time() - start_time
    return segments, total_merge_times, duration


def format_duration(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h:02d}h-{m:02d}m-{s:02d}s-{ms:03d}ms"


def main():
    script_dir = Path(__file__).parent.absolute()
    input_path = Path(INPUT_FILE)
    if not input_path.is_absolute():
        input_path = script_dir / INPUT_FILE
    if not input_path.exists():
        print(f"[Error] File not found: {input_path}")
        sys.exit(1)

    _init_sglang_backend()

    if SHOW_PROGRESS:
        print("Reading file and filtering empty lines...")

    lines = read_lines_filter_empty(input_path)
    if SHOW_PROGRESS:
        print(f"Found {len(lines)} lines (empty lines filtered)")
        print(f"Will perform up to {max(len(lines) - 1, 0)} adjacent line judgments (some will be skipped due to sentence-ending punctuation)")
        print(f"Concurrent processing count: {BATCH_SIZE}")
        print("=" * SEPARATOR_LENGTH)
        print("")

    original_line_count = len(lines)

    # Prepare output path and record file path
    if OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR)
        if not output_dir.is_absolute():
            output_dir = script_dir / OUTPUT_DIR
    else:
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (input_path.stem + OUTPUT_SUFFIX)
    record_path = output_dir / "model_responses.txt"

    total_start = time.time()
    # Open record file, use append mode (if file already exists)
    with open(record_path, "w", encoding=OUTPUT_ENCODING) as record_file:
        segments, merge_times, judge_duration = merge_iteratively(lines, record_file)
    total_duration = time.time() - total_start

    # Generate JSON results
    chunks = [
        {"length": len(seg.text), "text": seg.text}
        for seg in segments
        if seg.text.strip() != ""
    ]
    max_length = max((c["length"] for c in chunks), default=0)
    min_length = min((c["length"] for c in chunks), default=0)

    output_data = {
        "max_length": max_length,
        "min_length": min_length,
        "chunks": chunks,
    }

    # Write JSON file
    with open(out_path, "w", encoding=OUTPUT_ENCODING) as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    if SHOW_PROGRESS:
        print("")
        print("Multi-round merge complete!")
        print("=" * SEPARATOR_LENGTH)
        print(f"Total judgment time: {format_duration(judge_duration)}")
        print("-" * SEPARATOR_LENGTH)
        print("")
        print("Merge complete!")
        print(f"Original line count: {original_line_count}")
        print(f"Lines after merge: {len(segments)}")
        print(f"Output file: {out_path}")
        print(f"Model response record file: {record_path}")
        print("")
        print("Merge statistics:")
        print(f"  Total merges (line reduction): {merge_times}")
        print("=" * SEPARATOR_LENGTH)
        print(f"Total processing time: {format_duration(total_duration)}")
        print("All processing complete!")
        print("=" * SEPARATOR_LENGTH)


if __name__ == "__main__":
    main()
