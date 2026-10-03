#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate outlines for chunked text.
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

# ---------------------------- Outline Language Config -------------------------------
# en: generate English outlines
SUMMARY_LANGUAGE: str = "en"

# ----------------------------------- File Config ------------------------------------
INPUT_FILE: str = str(EXAMPLE_BOOK_DIR / "Far From the Madding Crowd_MergeRule_Title_NoHeaderFooter_Level.json")
OUTPUT_SUFFIX: str = "_L2Outline.json"

# ----------------------------- Chunk Punctuation ------------------------------------
SENTENCE_END_PUNCTUATIONS = ["!", "?", ";", "."]

# ------------------------------- Prompt Template ------------------------------------
PROMPT_0_6000: str = """
Based on the following content, summarize an outline of {word_limit} words or fewer. The outline must accurately capture the information in the content, strictly based on the content, without adding any subjective content.
Given text:
"{content}"
Return the following JSON format:
{{
    "reasoning": "Do you think the summarized outline has hallucinations? Explain in 300 words or fewer",
    "outline": "outline content in {word_limit} words or fewer"
}}
"""

PROMPT_6000_first_agent: str = """
You are a multi-agent outline generation expert. The following content is the first (upper) part of a continuous text. Please summarize an outline of 500 words or fewer based on this content to pass to the next agent.
The outline must accurately capture the information, strictly based on the content, without adding any subjective content.
Given text:
"{content}"
Return the following JSON format:
{{
    "reasoning": "Do you think the summarized outline has hallucinations? Explain in 300 words or fewer",
    "outline": "outline content in 500 words or fewer"
}}
"""

PROMPT_6000_next_agent: str = """
You are a multi-agent outline generation expert. The following outline was summarized by the previous agent.
Previous agent's outline:
"{previous_outline}"
Based on the above outline and the following content, please summarize an outline of 500 words or fewer to pass to the next agent.
The outline must accurately capture the information from both the previous outline and the following content, strictly based on both, without adding any subjective content.
Given text:
"{content}"
Return the following JSON format:
{{
    "reasoning": "Do you think the summarized outline has hallucinations? Explain in 300 words or fewer",
    "outline": "outline content in 500 words or fewer"
}}
"""

PROMPT_6000_last_agent: str = """
You are the final agent in the multi-agent outline generation chain. The following outline was summarized by the previous agent.
Previous agent's outline:
"{previous_outline}"
Based on the above outline and the following content, please summarize a final outline of 600 words or fewer as the full-text summary.
The outline must accurately capture the information from both the previous outline and the following content, strictly based on both, without adding any subjective content.
Given text:
"{content}"
Return the following JSON format:
{{
    "reasoning": "Do you think the summarized outline has hallucinations? Explain in 300 words or fewer",
    "outline": "outline content in 500 words or fewer"
}}
"""

# ---------------------------------- Other Config ------------------------------------
SEPARATOR_LENGTH: int = 60
SHOW_PROGRESS: bool = True

# ================================================================================================
# ========================================== Code =================================================
# ================================================================================================

import json
import math
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
from typing import List, Tuple, Dict, Any

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


def apply_summary_language_instruction(prompt: str) -> str:
    language = str(SUMMARY_LANGUAGE or "en").strip().lower()
    if language in {"en", "english", "English"}:
        return (
            prompt.rstrip()
            + "\n\nAdditional requirement: Please keep the above JSON field names unchanged, but the content of \"reasoning\" and \"outline\" fields must be output in English."
            + "The summary must be written in English only.\n"
        )
    return prompt


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
        print("\n" + "=" * 80, flush=True)
        print("[Raw model output - Start]", flush=True)
        print("-" * 80, flush=True)
        print(raw_answer, flush=True)
        print("-" * 80, flush=True)
        print("[Raw model output - End]", flush=True)
        print("=" * 80 + "\n", flush=True)

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


def parse_outline_result(model_response: str) -> Tuple[Dict[str, str], bool]:
    """
    Strictly parse model response, returns (result dict, is_format_valid)
    result dict contains: {"reasoning": "", "outline": ""}
    """
    result = {
        "reasoning": "",
        "outline": "",
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
        
        # Strictly check required fields
        if "outline" not in data:
            return result, ok
        
        # Extract values
        result["reasoning"] = str(data.get("reasoning", "") or "").strip()
        result["outline"] = str(data.get("outline", "") or "").strip()
        
        # If outline field exists (even if empty), consider format valid
        ok = True
        return result, ok
    except Exception:
        # Parse failed, return False directly
        pass

    return result, ok


def determine_word_limit(text_length: int) -> Tuple[int, int]:
    """
    Determine outline word limit and reasoning word limit based on text length
    Returns: (outline word limit, reasoning word limit)
    """
    if text_length <= 100:
        return 50, 30
    elif text_length <= 1000:
        return 100, 50
    elif text_length <= 2000:
        return 200, 100
    elif text_length <= 3000:
        return 300, 150
    elif text_length <= 4000:
        return 400, 200
    elif text_length <= 5000:
        return 500, 250
    elif text_length <= 6000:
        return 600, 300
    else:
        return 600, 300


def split_text_by_position(text: str, target_pos: int) -> Tuple[str, str]:
    """
    Split text at target position by looking backward for sentence-ending punctuation
    Returns: (first half, second half)
    """
    if target_pos >= len(text):
        return text, ""
    
    # Look backward for nearest sentence-ending punctuation
    cut_pos = -1
    for i in range(target_pos, 0, -1):
        if text[i - 1] in SENTENCE_END_PUNCTUATIONS:
            cut_pos = i
            break
    
    if cut_pos == -1:
        cut_pos = target_pos
    
    return text[:cut_pos].strip(), text[cut_pos:].strip()


def split_long_text(text: str, text_length: int) -> List[str]:
    """
    Split text over 6000 chars into multiple segments
    Returns segment list
    """
    num_segments = math.ceil(text_length / 6000)
    segment_length = text_length // num_segments
    
    segments = []
    remaining = text
    
    for i in range(num_segments - 1):
        part1, part2 = split_text_by_position(remaining, segment_length)
        if part1:
            segments.append(part1)
        remaining = part2
        if not remaining:
            break
    
    if remaining:
        segments.append(remaining)
    
    return segments


def generate_outline_for_long_text(text: str) -> str:
    """
    Use multi-agent approach to generate outline for text over 6000 chars
    """
    text_length = len(text)
    segments = split_long_text(text, text_length)
    
    if SHOW_PROGRESS:
        print(f"  Long text split into {len(segments)} segments")
    
    if len(segments) == 0:
        return ""
    
    # First segment
    prompt = apply_summary_language_instruction(PROMPT_6000_first_agent.format(content=segments[0]))
    previous_outline = ""
    
    response = call_model(prompt, MAX_GENERATE_TOKENS)
    result, ok = parse_outline_result(response)
    if ok and result["outline"]:
        previous_outline = result["outline"]
    else:
        # Retry once
        if SHOW_PROGRESS:
            print(f"  [Warning] First segment format invalid, will retry once...")
        retry_response = call_model(prompt, MAX_GENERATE_TOKENS)
        result, ok = parse_outline_result(retry_response)
        if ok and result["outline"]:
            previous_outline = result["outline"]
        else:
            # Still failed after retry, return empty text
            if SHOW_PROGRESS:
                print(f"  [Skipped] First segment still failed after retry, using default")
            return ""
    
    # Middle segments
    for i in range(1, len(segments) - 1):
        prompt = apply_summary_language_instruction(PROMPT_6000_next_agent.format(
            previous_outline=previous_outline,
            content=segments[i]
        ))
        
        response = call_model(prompt, MAX_GENERATE_TOKENS)
        result, ok = parse_outline_result(response)
        if ok and result["outline"]:
            previous_outline = result["outline"]
        else:
            # Retry once
            if SHOW_PROGRESS:
                print(f"  [Warning] Segment {i+1} format invalid, will retry once...")
            retry_response = call_model(prompt, MAX_GENERATE_TOKENS)
            result, ok = parse_outline_result(retry_response)
            if ok and result["outline"]:
                previous_outline = result["outline"]
            else:
                # Still failed after retry, return empty text
                if SHOW_PROGRESS:
                    print(f"  [Skipped] Segment {i+1} still failed after retry, using default")
                return ""
    
    # Last segment
    if len(segments) > 1:
        prompt = apply_summary_language_instruction(PROMPT_6000_last_agent.format(
            previous_outline=previous_outline,
            content=segments[-1]
        ))
        
        response = call_model(prompt, MAX_GENERATE_TOKENS)
        result, ok = parse_outline_result(response)
        if ok and result["outline"]:
            return result["outline"]
        else:
            # Retry once
            if SHOW_PROGRESS:
                print(f"  [Warning] Last segment format invalid, will retry once...")
            retry_response = call_model(prompt, MAX_GENERATE_TOKENS)
            result, ok = parse_outline_result(retry_response)
            if ok and result["outline"]:
                return result["outline"]
            else:
                # Still failed after retry, return empty text
                if SHOW_PROGRESS:
                    print(f"  [Skipped] Last segment still failed after retry, using default")
                return ""
    else:
        # Only one segment
        return previous_outline


def generate_outline_batch(items: List[Dict[str, Any]], batch_size: int) -> List[str]:
    """
    Batch generate outlines
    items: list of dicts containing {"index", "text", "length", "prompt"}
    Returns outline list (matching items order)
    """
    _init_sglang_backend()
    results = [""] * len(items)
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        # Local indices within current items list
        batch_local_indices = list(range(i, min(i + batch_size, len(items))))
        # Original indices for log output
        batch_original_indices = [items[local_idx]["index"] for local_idx in batch_local_indices]
        batch_prompts = [items[local_idx]["prompt"] for local_idx in batch_local_indices]
        batch_num = i // batch_size + 1
        
        if SHOW_PROGRESS:
            print(f"  Batch progress: {batch_num}/{total_batches} batches - processing concurrently...", flush=True)
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_local_index = {
                executor.submit(call_model, prompt=prompt, max_new_tokens=MAX_GENERATE_TOKENS): local_idx
                for local_idx, prompt in zip(batch_local_indices, batch_prompts)
            }
            batch_results = {}
            for future in as_completed(future_to_local_index):
                local_idx = future_to_local_index[future]
                original_idx = items[local_idx]["index"]
                try:
                    ans = future.result()
                    batch_results[local_idx] = ans
                except Exception as exc:
                    if SHOW_PROGRESS:
                        print(f"    [Error] Call failed for original index {original_idx} in batch {batch_num}: {exc}", flush=True)
                    batch_results[local_idx] = ""
            
            for local_idx in batch_local_indices:
                results[local_idx] = batch_results.get(local_idx, "")
    
    return results


def process_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all chunks, generate outlines
    Returns chunks list with outline field
    """
    # Group by length
    group_0_2000 = []
    group_2001_4000 = []
    group_4001_6000 = []
    group_6000_plus = []
    
    for i, chunk in enumerate(chunks):
        length = chunk["length"]
        text = chunk["text"]
        
        item = {
            "index": i,
            "text": text,
            "length": length,
        }
        
        if length <= 2000:
            group_0_2000.append(item)
        elif length <= 4000:
            group_2001_4000.append(item)
        elif length <= 6000:
            group_4001_6000.append(item)
        else:
            group_6000_plus.append(item)
    
    # Prepare prompts for each group
    for item in group_0_2000:
        word_limit, think_limit = determine_word_limit(item["length"])
        item["prompt"] = apply_summary_language_instruction(PROMPT_0_6000.format(
            word_limit=word_limit,
            think_limit=think_limit,
            content=item["text"]
        ))
    
    for item in group_2001_4000:
        word_limit, think_limit = determine_word_limit(item["length"])
        item["prompt"] = apply_summary_language_instruction(PROMPT_0_6000.format(
            word_limit=word_limit,
            think_limit=think_limit,
            content=item["text"]
        ))
    
    for item in group_4001_6000:
        word_limit, think_limit = determine_word_limit(item["length"])
        item["prompt"] = apply_summary_language_instruction(PROMPT_0_6000.format(
            word_limit=word_limit,
            think_limit=think_limit,
            content=item["text"]
        ))
    
    # Initialize results array
    outlines = [""] * len(chunks)
    
    # Process 0-2000 char group
    if group_0_2000:
        if SHOW_PROGRESS:
            print(f"\nProcessing 0-2000 char group ({len(group_0_2000)} segments, BATCH_SIZE=16)...")
        raw_results = generate_outline_batch(group_0_2000, batch_size=16)
        
        # First parse round: collect success and failure indices
        group_parsed_list = [None] * len(group_0_2000)
        failed_indices = []
        
        for idx, (item, raw_result) in enumerate(zip(group_0_2000, raw_results)):
            result, ok = parse_outline_result(raw_result)
            if ok:
                group_parsed_list[idx] = result
                outlines[item["index"]] = result["outline"]
            else:
                failed_indices.append(idx)
                if SHOW_PROGRESS:
                    print(f"[Warning] Original index {item['index']} format invalid, will retry once")
        
        # Retry format-invalid ones (only once)
        if failed_indices:
            if SHOW_PROGRESS:
                print(f"  Found {len(failed_indices)} segments with invalid format, retrying (once each)...")
            
            for idx in failed_indices:
                item = group_0_2000[idx]
                if SHOW_PROGRESS:
                    print(f"  [Retry] Original index {item['index']}...")
                
                retry_response = call_model(item["prompt"], MAX_GENERATE_TOKENS)
                result, ok = parse_outline_result(retry_response)
                
                if ok:
                    group_parsed_list[idx] = result
                    outlines[item["index"]] = result["outline"]
                    if SHOW_PROGRESS:
                        print(f"  [Success] Original index {item['index']} retry succeeded")
                else:
                    # Still failed after retry, use default (empty text)
                    outlines[item["index"]] = ""
                    if SHOW_PROGRESS:
                        print(f"  [Skipped] Original index {item['index']} still failed after retry, using default")
    
    # Process 2001-4000 char group
    if group_2001_4000:
        if SHOW_PROGRESS:
            print(f"\nProcessing 2001-4000 char group ({len(group_2001_4000)} segments, BATCH_SIZE=8)...")
        raw_results = generate_outline_batch(group_2001_4000, batch_size=8)
        
        # First parse round: collect success and failure indices
        group_parsed_list = [None] * len(group_2001_4000)
        failed_indices = []
        
        for idx, (item, raw_result) in enumerate(zip(group_2001_4000, raw_results)):
            result, ok = parse_outline_result(raw_result)
            if ok:
                group_parsed_list[idx] = result
                outlines[item["index"]] = result["outline"]
            else:
                failed_indices.append(idx)
                if SHOW_PROGRESS:
                    print(f"[Warning] Original index {item['index']} format invalid, will retry once")
        
        # Retry format-invalid ones (only once)
        if failed_indices:
            if SHOW_PROGRESS:
                print(f"  Found {len(failed_indices)} segments with invalid format, retrying (once each)...")
            
            for idx in failed_indices:
                item = group_2001_4000[idx]
                if SHOW_PROGRESS:
                    print(f"  [Retry] Original index {item['index']}...")
                
                retry_response = call_model(item["prompt"], MAX_GENERATE_TOKENS)
                result, ok = parse_outline_result(retry_response)
                
                if ok:
                    group_parsed_list[idx] = result
                    outlines[item["index"]] = result["outline"]
                    if SHOW_PROGRESS:
                        print(f"  [Success] Original index {item['index']} retry succeeded")
                else:
                    # Still failed after retry, use default (empty text)
                    outlines[item["index"]] = ""
                    if SHOW_PROGRESS:
                        print(f"  [Skipped] Original index {item['index']} still failed after retry, using default")
    
    # Process 4001-6000 char group
    if group_4001_6000:
        if SHOW_PROGRESS:
            print(f"\nProcessing 4001-6000 char group ({len(group_4001_6000)} segments, BATCH_SIZE=4)...")
        raw_results = generate_outline_batch(group_4001_6000, batch_size=4)
        
        # First parse round: collect success and failure indices
        group_parsed_list = [None] * len(group_4001_6000)
        failed_indices = []
        
        for idx, (item, raw_result) in enumerate(zip(group_4001_6000, raw_results)):
            result, ok = parse_outline_result(raw_result)
            if ok:
                group_parsed_list[idx] = result
                outlines[item["index"]] = result["outline"]
            else:
                failed_indices.append(idx)
                if SHOW_PROGRESS:
                    print(f"[Warning] Original index {item['index']} format invalid, will retry once")
        
        # Retry format-invalid ones (only once)
        if failed_indices:
            if SHOW_PROGRESS:
                print(f"  Found {len(failed_indices)} segments with invalid format, retrying (once each)...")
            
            for idx in failed_indices:
                item = group_4001_6000[idx]
                if SHOW_PROGRESS:
                    print(f"  [Retry] Original index {item['index']}...")
                
                retry_response = call_model(item["prompt"], MAX_GENERATE_TOKENS)
                result, ok = parse_outline_result(retry_response)
                
                if ok:
                    group_parsed_list[idx] = result
                    outlines[item["index"]] = result["outline"]
                    if SHOW_PROGRESS:
                        print(f"  [Success] Original index {item['index']} retry succeeded")
                else:
                    # Still failed after retry, use default (empty text)
                    outlines[item["index"]] = ""
                    if SHOW_PROGRESS:
                        print(f"  [Skipped] Original index {item['index']} still failed after retry, using default")
    
    # Process 6000+ char group
    if group_6000_plus:
        if SHOW_PROGRESS:
            print(f"\nProcessing 6000+ char group ({len(group_6000_plus)} segments, BATCH_SIZE=1)...")
        for item in group_6000_plus:
            if SHOW_PROGRESS:
                print(f"  Processing index {item['index']} (length: {item['length']} chars)...")
            outline = generate_outline_for_long_text(item["text"])
            outlines[item["index"]] = outline
    
    # Assemble final results
    result_chunks = []
    for i, chunk in enumerate(chunks):
        new_chunk = dict(chunk)
        new_chunk["length"] = chunk["length"]
        new_chunk["h2_outline"] = outlines[i]
        new_chunk["text"] = chunk["text"]
        result_chunks.append(new_chunk)
    
    return result_chunks


def main():
    if SHOW_PROGRESS:
        print("=" * SEPARATOR_LENGTH)
        print("Outline Generation System")
        print("=" * SEPARATOR_LENGTH)
    
    script_dir = Path(__file__).parent.absolute()
    input_file_path = Path(INPUT_FILE)
    if not input_file_path.is_absolute():
        input_file_path = script_dir / INPUT_FILE
    
    if not input_file_path.exists():
        print(f"[Error] File not found: {input_file_path}")
        sys.exit(1)
    
    # Build output file path
    output_file_path = input_file_path.parent / (input_file_path.stem + OUTPUT_SUFFIX)
    
    if SHOW_PROGRESS:
        print(f"Input file: {input_file_path}")
        print(f"Output file: {output_file_path}")
        print("=" * SEPARATOR_LENGTH)
    
    # Initialize backend
    try:
        _init_sglang_backend()
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("\nPlease start the SGLang server first, e.g.:")
        print(f"python -m sglang.launch_server --model-path {MODEL_PATH} --port 30000")
        sys.exit(1)
    
    # Read input file
    if SHOW_PROGRESS:
        print("Reading input file...")
    
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = data.get("chunks", [])
    if not chunks:
        print("[Error] No chunks data in input file.")
        sys.exit(1)
    
    if SHOW_PROGRESS:
        print(f"Read {len(chunks)} text segments")
        print("=" * SEPARATOR_LENGTH)
    
    # Start processing
    start_time = time.time()
    result_chunks = process_chunks(chunks)
    duration = time.time() - start_time
    
    if SHOW_PROGRESS:
        print("\n" + "=" * SEPARATOR_LENGTH)
        print(f"Processing complete! Total time: {duration:.2f} seconds")
        print("=" * SEPARATOR_LENGTH)
    
    # Save results
    output_data = {
        "max_length": data.get("max_length", 0),
        "min_length": data.get("min_length", 0),
        "chunks": result_chunks,
        "anchor_file": data.get("anchor_file", ""),
        "source_text_file": data.get("source_text_file", ""),
    }
    
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    if SHOW_PROGRESS:
        print(f"Results saved to: {output_file_path}")
        print("=" * SEPARATOR_LENGTH)


if __name__ == "__main__":
    main()
