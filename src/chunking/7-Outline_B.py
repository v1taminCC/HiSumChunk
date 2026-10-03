#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate outlines for h2_regions.
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
INPUT_FILE: str = str(EXAMPLE_BOOK_DIR / "Far From the Madding Crowd_MergeRule_Title_NoHeaderFooter_Level_L2Outline_MergeH2.json")
OUTPUT_SUFFIX: str = "_L1Outline.json"

# ------------------------------ Word Limit Config -----------------------------------
# Dynamically set word_limit based on merged content character count
WORD_LIMIT_0_50: int = 50
WORD_LIMIT_50_200: int = 100
WORD_LIMIT_201_500: int = 300
WORD_LIMIT_501_2000: int = 1000
WORD_LIMIT_2001_4000: int = 2000
WORD_LIMIT_4001_12000: int = 3000

# -------------------------------- Batch Size Config ---------------------------------
BATCH_SIZE_0_50: int = 16
BATCH_SIZE_50_200: int = 16
BATCH_SIZE_201_500: int = 16
BATCH_SIZE_501_2000: int = 16
BATCH_SIZE_2001_4000: int = 8
BATCH_SIZE_4001_12000: int = 3

# ------------------------------- Prompt Template ------------------------------------
PROMPT_Outline_B: str = """
Based on the following text, summarize an outline of {word_limit} words or fewer. The outline must accurately capture the information in the text, strictly based on the text, without adding any subjective content.
Given text:
"{content}"
Return the following JSON format:
{{
    "reasoning": "Do you think the summarized outline has hallucinations? Explain in 300 words or fewer",
    "outline": "outline content in {word_limit} words or fewer"
}}
"""

# ---------------------------------- Other Config ------------------------------------
SEPARATOR_LENGTH: int = 60
SHOW_PROGRESS: bool = True
OUTLINE_SEPARATOR: str = ".\n"

# ================================================================================================
# ========================================== Code =================================================
# ================================================================================================

import json
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


def determine_word_limit_and_batch_size(content_length: int) -> Tuple[int, int]:
    """
    Determine word_limit and batch_size based on merged content character count
    Returns: (word_limit, batch_size)
    """
    if 0 <= content_length < 50:
        return WORD_LIMIT_0_50, BATCH_SIZE_0_50
    elif 50 <= content_length <= 200:
        return WORD_LIMIT_50_200, BATCH_SIZE_50_200
    elif 201 <= content_length <= 500:
        return WORD_LIMIT_201_500, BATCH_SIZE_201_500
    elif 501 <= content_length <= 2000:
        return WORD_LIMIT_501_2000, BATCH_SIZE_501_2000
    elif 2001 <= content_length <= 4000:
        return WORD_LIMIT_2001_4000, BATCH_SIZE_2001_4000
    elif 4001 <= content_length <= 12000:
        return WORD_LIMIT_4001_12000, BATCH_SIZE_4001_12000
    else:
        # Beyond 12000 chars, use max config
        return WORD_LIMIT_4001_12000, BATCH_SIZE_4001_12000


def merge_chunks_outlines(chunks: List[Dict[str, Any]]) -> str:
    """
    Merge all non-empty outlines from chunks into a single string
    Uses OUTLINE_SEPARATOR as delimiter
    """
    outlines = []
    for chunk in chunks:
        outline = chunk.get("h2_outline", "")
        if outline and isinstance(outline, str) and outline.strip():
            outlines.append(outline.strip())
    
    return OUTLINE_SEPARATOR.join(outlines)


def generate_outline_batch(items: List[Dict[str, Any]], batch_size: int) -> List[str]:
    """
    Batch generate outlines
    items: list of dicts containing {"index", "content", "length", "word_limit", "prompt"}
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


def process_h2_regions(h2_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all h2_regions, generate h1_outline for each group of chunks
    Returns h2_regions list with h1_outline field
    """
    # Step 1: Merge each group of chunks' outlines and prepare processing items
    processing_items = []
    
    for h2_idx, h2_region in enumerate(h2_regions):
        chunks = h2_region.get("chunks", [])
        if not chunks:
            continue
        
        # Merge outlines
        merged_content = merge_chunks_outlines(chunks)
        if not merged_content:
            # Skip if no valid outlines
            continue
        
        content_length = len(merged_content)
        word_limit, batch_size = determine_word_limit_and_batch_size(content_length)
        
        # Generate prompt
        prompt = apply_summary_language_instruction(PROMPT_Outline_B.format(
            word_limit=word_limit,
            content=merged_content
        ))
        
        processing_items.append({
            "index": h2_idx,
            "content": merged_content,
            "length": content_length,
            "word_limit": word_limit,
            "batch_size": batch_size,
            "prompt": prompt
        })
    
    if not processing_items:
        if SHOW_PROGRESS:
            print("[Warning] No h2_regions found to process")
        return h2_regions
    
    # Step 2: Sort by character count
    processing_items.sort(key=lambda x: x["length"])
    
    if SHOW_PROGRESS:
        print(f"\nPreparing to process {len(processing_items)} h2_regions")
        print("Grouped by character count:")
        groups = {
            "0-50": 0,
            "50-200": 0,
            "201-500": 0,
            "501-2000": 0,
            "2001-4000": 0,
            "4001-12000": 0
        }
        for item in processing_items:
            length = item["length"]
            if 0 <= length < 50:
                groups["0-50"] += 1
            elif 50 <= length <= 200:
                groups["50-200"] += 1
            elif 201 <= length <= 500:
                groups["201-500"] += 1
            elif 501 <= length <= 2000:
                groups["501-2000"] += 1
            elif 2001 <= length <= 4000:
                groups["2001-4000"] += 1
            elif 4001 <= length <= 12000:
                groups["4001-12000"] += 1
        for group_name, count in groups.items():
            if count > 0:
                print(f"  {group_name} chars: {count}")
    
    # Step 3: Process by batch_size grouping
    # Initialize results array
    h2_outlines = [""] * len(h2_regions)
    
    # Group by batch_size
    groups_by_batch = {}
    for item in processing_items:
        batch_size = item["batch_size"]
        if batch_size not in groups_by_batch:
            groups_by_batch[batch_size] = []
        groups_by_batch[batch_size].append(item)
    
    # Process from smallest to largest batch_size (16 -> 8 -> 3)
    batch_sizes_order = [16, 8, 3]
    for batch_size in batch_sizes_order:
        if batch_size not in groups_by_batch:
            continue
        
        group_items = groups_by_batch[batch_size]
        if not group_items:
            continue
        
        # Determine the character range for display
        lengths = [item["length"] for item in group_items]
        min_len = min(lengths)
        max_len = max(lengths)
        
        if SHOW_PROGRESS:
            print(f"\nProcessing {min_len}-{max_len} char group ({len(group_items)} h2_regions, BATCH_SIZE={batch_size})...")
        
        raw_results = generate_outline_batch(group_items, batch_size=batch_size)
        
        # First parse round: collect success and failure indices
        group_parsed_list = [None] * len(group_items)
        failed_indices = []
        
        for idx, (item, raw_result) in enumerate(zip(group_items, raw_results)):
            result, ok = parse_outline_result(raw_result)
            if ok:
                group_parsed_list[idx] = result
                h2_outlines[item["index"]] = result["outline"]
            else:
                failed_indices.append(idx)
                if SHOW_PROGRESS:
                    print(f"[Warning] h2_region index {item['index']} format invalid, will retry once")
        
        # Retry format-invalid ones (only once)
        if failed_indices:
            if SHOW_PROGRESS:
                print(f"  Found {len(failed_indices)} h2_regions with invalid format, retrying (once each)...")
            
            for idx in failed_indices:
                item = group_items[idx]
                if SHOW_PROGRESS:
                    print(f"  [Retry] h2_region index {item['index']}...")
                
                retry_response = call_model(item["prompt"], MAX_GENERATE_TOKENS)
                result, ok = parse_outline_result(retry_response)
                
                if ok:
                    group_parsed_list[idx] = result
                    h2_outlines[item["index"]] = result["outline"]
                    if SHOW_PROGRESS:
                        print(f"  [Success] h2_region index {item['index']} retry succeeded")
                else:
                    # Still failed after retry, use default (empty text)
                    h2_outlines[item["index"]] = ""
                    if SHOW_PROGRESS:
                        print(f"  [Skipped] h2_region index {item['index']} still failed after retry, using default")
    
    # Step 4: Assemble final results
    result_h2_regions = []
    for i, h2_region in enumerate(h2_regions):
        # Create new dict, ensuring h1_outline comes before chunks
        new_h2_region = {}
        # Add h1_outline first
        new_h2_region["h1_outline"] = h2_outlines[i]
        # Then add all other fields (including chunks)
        for key, value in h2_region.items():
            new_h2_region[key] = value
        result_h2_regions.append(new_h2_region)
    
    return result_h2_regions


def main():
    if SHOW_PROGRESS:
        print("=" * SEPARATOR_LENGTH)
        print("h2_regions Outline Generation System")
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

    # Handle input file format:
    # - Compatible with old array format: [{"chunks": [...]}, ...]
    # - Compatible with new object format: {"max_length": ..., "min_length": ..., "h2_regions": [...]}
    max_length = 0
    min_length = 0
    if isinstance(data, list):
        h2_regions = data
    elif isinstance(data, dict) and "h2_regions" in data:
        max_length = data.get("max_length", 0)
        min_length = data.get("min_length", 0)
        h2_regions = data.get("h2_regions", [])
    elif isinstance(data, dict) and "chunks" in data:
        # If only chunks field exists, the input file format is incorrect
        print("[Error] Input file only has chunks field, missing h2_regions data.")
        print("[Hint] Please run 6-Merge_H2.py first to generate the file with h2_regions, then run this script.")
        print(f"[Hint] Or change the input file to: {INPUT_FILE.replace('_L2Outline.json', '_L2Outline_MergeH2.json')}")
        sys.exit(1)
    else:
        h2_regions = []
    
    if not h2_regions:
        print("[Error] No h2_regions data in input file.")
        sys.exit(1)
    
    if SHOW_PROGRESS:
        print(f"Read {len(h2_regions)} h2_regions")
        print("=" * SEPARATOR_LENGTH)
    
    # Start processing
    start_time = time.time()
    result_h2_regions = process_h2_regions(h2_regions)
    duration = time.time() - start_time
    
    if SHOW_PROGRESS:
        print("\n" + "=" * SEPARATOR_LENGTH)
        print(f"Processing complete! Total time: {duration:.2f} seconds")
        print("=" * SEPARATOR_LENGTH)
    
    # Save results
    with open(output_file_path, "w", encoding="utf-8") as f:
        output_data = {
            "max_length": max_length,
            "min_length": min_length,
            "h2_regions": result_h2_regions,
            "anchor_file": data.get("anchor_file", ""),
            "source_text_file": data.get("source_text_file", ""),
        }
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    if SHOW_PROGRESS:
        print(f"Results saved to: {output_file_path}")
        print("=" * SEPARATOR_LENGTH)


if __name__ == "__main__":
    main()
