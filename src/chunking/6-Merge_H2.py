#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge H2 title regions.
Based on chunk outlines, logically merge using SGLang model calls.
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

# ----------------------------------- File Config ------------------------------------
INPUT_FILE: str = str(EXAMPLE_BOOK_DIR / "Far From the Madding Crowd_MergeRule_Title_NoHeaderFooter_Level_L2Outline.json")
OUTPUT_SUFFIX: str = "_MergeH2.json"

# ------------------------------- Batch Config ---------------------------------------
MAX_CHUNKS_PER_BATCH: int = 20

# ------------------------------- Prompt Template ------------------------------------
PROMPT_TEMPLATE: str = """
Below are the summaries of {} chunks. Assign them an ID (starting from 1), and logically divide them into one or more H2 title regions (it can also be a single region).
Note: The chunks are sequential, i.e., the summary order must not be shuffled. Do not omit any given chunks.

Summaries:
{}

Return strictly in the following JSON format (do not include Markdown code block markers, return raw JSON):
{{
    "reasoning": "What is the thematic flow between chunks? Explain in 500 words or fewer",
    "merge_range": 
    {{
        "region_1_id_range": [start_id, end_id],
        "region_2_id_range": [start_id, end_id]
        If there are more regions, continue until none remain.
    }}
}}
"""

# ---------------------------------- Other Config ------------------------------------
SEPARATOR_LENGTH: int = 60
SHOW_PROGRESS: bool = True

# ================================================================================================
# ========================================== Code =================================================
# ================================================================================================

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
EXAMPLE_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))
from typing import List, Tuple, Dict, Any, Optional

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


def parse_merge_result(model_response: str) -> Tuple[Dict[str, Any], bool]:
    """
    Strictly parse the merge result returned by the model
    Returns: (result dict, is_format_valid)
    """
    result = {
        "reasoning": "",
        "merge_range": {}
    }
    ok = False
    
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
        if "merge_range" not in data:
            return result, ok
        
        # Extract values
        result["reasoning"] = str(data.get("reasoning", "") or "").strip()
        merge_range = data.get("merge_range", {})
        
        if not isinstance(merge_range, dict):
            return result, ok
        
        # Parse merge ranges
        result["merge_range"] = {}
        for key, value in merge_range.items():
            if isinstance(value, list) and len(value) == 2:
                start_id = value[0]
                end_id = value[1]
                if isinstance(start_id, int) and isinstance(end_id, int):
                    result["merge_range"][key] = [start_id, end_id]
        
        # If at least one region exists, consider format valid
        if result["merge_range"]:
            ok = True
        
    except Exception:
        # Parse failed, return False directly
        pass
    
    return result, ok


def format_summaries_section(chunk_items: List[Dict[str, Any]], start_id: int) -> str:
    """
    Format the summaries section (the 4-10 line portion in prompt template)
    """
    lines = []
    current_id = start_id
    
    for item in chunk_items:
        chunk_index = item["chunk_index"]  # original index (starting from 1)
        outline = item.get("outline", "")
        lines.append("--------------------------------")
        lines.append(f'ID: "{current_id}"')
        lines.append(f"Summary: {outline}")
        current_id += 1
    
    lines.append("--------------------------------")
    return "\n".join(lines)


def build_prompt(chunk_items: List[Dict[str, Any]], start_id: int) -> str:
    """
    Build prompt
    """
    num_chunks = len(chunk_items)
    summaries_section = format_summaries_section(chunk_items, start_id)
    
    prompt = PROMPT_TEMPLATE.format(num_chunks, summaries_section)
    return prompt


def process_batch(
    chunk_items: List[Dict[str, Any]], 
    start_id: int,
    batch_num: int
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Process a batch of chunks
    """
    if not chunk_items:
        return None, False
    
    prompt = build_prompt(chunk_items, start_id)
    
    if SHOW_PROGRESS:
        print(f"\nProcessing batch {batch_num} (ID range: {start_id} - {start_id + len(chunk_items) - 1}, total {len(chunk_items)} chunks)")
    
    # First call
    response = call_model(prompt, MAX_GENERATE_TOKENS)
    result, ok = parse_merge_result(response)
    
    if ok:
        if SHOW_PROGRESS:
            print(f"  [Success] Batch {batch_num} parsed successfully")
        return result, True
    else:
        # Retry once
        if SHOW_PROGRESS:
            print(f"  [Warning] Batch {batch_num} format invalid, will retry once...")
        
        retry_response = call_model(prompt, MAX_GENERATE_TOKENS)
        result, ok = parse_merge_result(retry_response)
        
        if ok:
            if SHOW_PROGRESS:
                print(f"  [Success] Batch {batch_num} retry succeeded")
            return result, True
        else:
            # Still failed after retry
            if SHOW_PROGRESS:
                print(f"  [Skipped] Batch {batch_num} still failed after retry, skipping")
            return None, False


def extract_last_region_ids(merge_range: Dict[str, Any]) -> List[int]:
    """
    Extract all IDs from the last region in the merge range
    [Modified]: Sort by ID value, not key string order
    Returns: ID list (in order)
    """
    if not merge_range:
        return []
    
    # [Modified] Sort by value[0] (start ID), ignoring key name ordering issues
    sorted_items = sorted(merge_range.items(), key=lambda x: x[1][0])
    
    if sorted_items:
        # Take the last one
        last_range = sorted_items[-1][1]
        if isinstance(last_range, list) and len(last_range) == 2:
            start_id, end_id = last_range[0], last_range[1]
            return list(range(start_id, end_id + 1))
    
    return []


def merge_regions_across_batches(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge regions across batches.
    Since each batch's IDs are independent (starting from 1), convert to global IDs.
    """
    global_regions = []
    
    for i, batch_result in enumerate(all_results):
        if not batch_result or "merge_range" not in batch_result:
            continue
        
        merge_range = batch_result["merge_range"]
        batch_start_global_id = batch_result["batch_start_global_id"]
        
        # [Modified] Sort by start ID to ensure correct processing order
        sorted_regions = sorted(merge_range.items(), key=lambda x: x[1][0])
        
        # Check if next batch exists and succeeded
        has_next_batch = (i + 1 < len(all_results))
        
        if has_next_batch:
            # Next batch exists: save all regions except the last one
            # The last region will be carried to the next batch for processing
            regions_to_save = sorted_regions[:-1] if len(sorted_regions) > 1 else []
        else:
            # No next batch (last batch or next batch failed): save all regions (including last)
            regions_to_save = sorted_regions
        
        # Convert batch-relative IDs to global IDs and save
        for region_key, id_range in regions_to_save:
            if isinstance(id_range, list) and len(id_range) == 2:
                relative_start, relative_end = id_range[0], id_range[1]
                # Convert to global IDs
                global_start = batch_start_global_id + (relative_start - 1)
                global_end = batch_start_global_id + (relative_end - 1)
                
                global_regions.append({
                    "h2_index": len(global_regions) + 1,
                    "id_range": [global_start, global_end]
                })
    
    return global_regions


def main():
    if SHOW_PROGRESS:
        print("=" * SEPARATOR_LENGTH)
        print("H2 Region Merge System")
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

    # Inherit max/min character length info from upstream (consistent with 5-Outline_A.py output)
    max_length = data.get("max_length", 0)
    min_length = data.get("min_length", 0)
    
    chunks = data.get("chunks", [])
    if not chunks:
        print("[Error] No chunks data in input file.")
        sys.exit(1)
    
    if SHOW_PROGRESS:
        print(f"Read {len(chunks)} text segments")
        print("=" * SEPARATOR_LENGTH)
    
    # Start processing
    start_time = time.time()
    
    # Prepare chunk items
    chunk_items_all = []
    for i, chunk in enumerate(chunks):
        chunk_items_all.append({
            "chunk_index": i + 1,  # original index (starting from 1)
            "outline": chunk.get("h2_outline", ""),
            "length": chunk.get("length", 0),
            "text": chunk.get("text", "")
        })
    
    # Batch processing
    all_batch_results = []
    current_index = 0
    batch_num = 1
    carry_in_ids = []  # All IDs from previous batch's last region (global IDs)
    
    while current_index < len(chunk_items_all):
        # Build current batch's chunk items
        batch_chunk_items = []
        batch_start_id = 1  # Starting ID within current batch (from 1)
        batch_start_global_id = None  # Global starting ID for current batch
        
        # If there are carry_in_ids, add them first
        if carry_in_ids:
            batch_start_global_id = carry_in_ids[0]  # Batch's global start ID is carry_in_ids first element
            for global_id in carry_in_ids:
                if 1 <= global_id <= len(chunk_items_all):
                    item = chunk_items_all[global_id - 1].copy()
                    batch_chunk_items.append(item)
                    if SHOW_PROGRESS:
                        print(f"  [Carry-over] Carrying previous batch's last region ID {global_id} into this batch")
        
        # Add new chunk items until reaching MAX_CHUNKS_PER_BATCH
        while len(batch_chunk_items) < MAX_CHUNKS_PER_BATCH and current_index < len(chunk_items_all):
            batch_chunk_items.append(chunk_items_all[current_index])
            if batch_start_global_id is None:
                batch_start_global_id = current_index + 1  # Global ID starts from 1
            current_index += 1
        
        if not batch_chunk_items:
            break
        
        # Process current batch
        result, success = process_batch(batch_chunk_items, batch_start_id, batch_num)
        
        if success and result:
            # Extract all IDs from the last region (for next batch)
            last_region_relative_ids = extract_last_region_ids(result["merge_range"])
            
            # Record last region info (for later processing)
            last_region_info = None
            if last_region_relative_ids:
                # Convert to global IDs
                last_region_global_ids = [batch_start_global_id + (id - 1) for id in last_region_relative_ids]
                
                # [Modified] Sort by value[0] here too to find the correct last region info
                sorted_regions = sorted(result["merge_range"].items(), key=lambda x: x[1][0])
                
                if sorted_regions:
                    last_region_key, last_region_range = sorted_regions[-1]
                    last_region_info = {
                        "region_key": last_region_key,
                        "relative_range": last_region_range,
                        "global_ids": last_region_global_ids
                    }
                carry_in_ids = last_region_global_ids
                if SHOW_PROGRESS:
                    print(f"  [Record] Last region global IDs: {carry_in_ids}")
            else:
                carry_in_ids = []
            
            # Record batch info
            batch_info = {
                "batch_num": batch_num,
                "batch_start_global_id": batch_start_global_id,
                "batch_chunk_count": len(batch_chunk_items),
                "merge_range": result["merge_range"],
                "thought": result["reasoning"],
                "last_region_info": last_region_info  # Record last region info
            }
            all_batch_results.append(batch_info)
        else:
            # On failure, don't clear carry_in_ids, keep using them
            if SHOW_PROGRESS:
                print(f"  [Warning] Batch {batch_num} failed, carry_in_ids unchanged: {carry_in_ids}")
        
        batch_num += 1
    
    # If there are still carry_in_ids after loop ends, process one more time
    if carry_in_ids and current_index >= len(chunk_items_all):
        batch_chunk_items = []
        batch_start_global_id = carry_in_ids[0]
        
        for global_id in carry_in_ids:
            if 1 <= global_id <= len(chunk_items_all):
                item = chunk_items_all[global_id - 1].copy()
                batch_chunk_items.append(item)
                if SHOW_PROGRESS:
                    print(f"  [Carry-over] Carrying previous batch's last region ID {global_id} into this batch (final)")
        
        if batch_chunk_items:
            result, success = process_batch(batch_chunk_items, 1, batch_num)
            
            if success and result:
                # Final batch, no next batch, so last_region_info is None
                batch_info = {
                    "batch_num": batch_num,
                    "batch_start_global_id": batch_start_global_id,
                    "batch_chunk_count": len(batch_chunk_items),
                    "merge_range": result["merge_range"],
                    "thought": result["reasoning"],
                    "last_region_info": None  # Final batch, no next
                }
                all_batch_results.append(batch_info)
            else:
                pass  # Don't record on failure
    
    duration = time.time() - start_time
    
    if SHOW_PROGRESS:
        print("\n" + "=" * SEPARATOR_LENGTH)
        print(f"Processing complete! Total time: {duration:.2f} seconds")
        print(f"Successful batches: {len(all_batch_results)}")
        print("=" * SEPARATOR_LENGTH)
    
    # Merge regions across batches
    global_regions = merge_regions_across_batches(all_batch_results)
    
    # Build final results
    h2_regions = []
    for region in global_regions:
        start_id, end_id = region["id_range"]
        
        # Collect all chunks for this region
        region_chunks = []
        for chunk_id in range(start_id, end_id + 1):
            if 1 <= chunk_id <= len(chunk_items_all):
                item = chunk_items_all[chunk_id - 1]
                source_chunk = chunks[chunk_id - 1]
                new_chunk = dict(source_chunk)
                new_chunk["length"] = item["length"]
                new_chunk["h2_outline"] = item["outline"]
                new_chunk["text"] = item["text"]
                region_chunks.append(new_chunk)

        h2_regions.append({
            "chunks": region_chunks
        })
    
    # Save results
    with open(output_file_path, "w", encoding="utf-8") as f:
        output_data = {
            "max_length": max_length,
            "min_length": min_length,
            "h2_regions": h2_regions,
            "anchor_file": data.get("anchor_file", ""),
            "source_text_file": data.get("source_text_file", ""),
        }
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    if SHOW_PROGRESS:
        print(f"Results saved to: {output_file_path}")
        print(f"Generated {len(h2_regions)} H2 regions")
        print("=" * SEPARATOR_LENGTH)


if __name__ == "__main__":
    main()
