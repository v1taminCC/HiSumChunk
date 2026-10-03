#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Determine whether a document contains level-1 titles and level-2 titles.
Uses SGLang framework to call local-llm model.
Every 4 pages form one group for judgment, using parallel processing.
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

# ----------------------------- Parallel Processing Config ---------------------------
BATCH_SIZE: int = 4
PAGES_PER_BATCH: int = 4

# ---------------------------- Title Judgment Weight Config --------------------------
LEVEL1_WEIGHT: int = 2
LEVEL2_WEIGHT: int = 1
THRESHOLD: float = 0.2

# ----------------------------------- File Config ------------------------------------
INPUT_FILE: str = str(EXAMPLE_BOOK_DIR / "work_txt" / "Far From the Madding Crowd_MergeRule.txt")

# ----------------------------- Page Marker Config -----------------------------------
# Page markers like: >>>PAGE 1<<<
PAGE_MARKER_PREFIX: str = ">>>PAGE "
PAGE_MARKER_SUFFIX: str = "<<<"

# ------------------------------ Judgment Prompt Template ----------------------------
PROMPT_TEMPLATE: str = """
Below are {} pages of a document. Based on these {} pages, determine whether this document contains level-1 titles and level-2 titles.
Note: Sentence-initial enumerative guides such as "first, second, third...", "for one thing, for another..." etc., no matter whether numbered or not, are only logical sequencing phrases within the body text and should NOT be considered as titles.
Given text: 
Page 1{} 
Page 2{} 
Page 3{}
Page 4{}
Return the following JSON format:
{{
    "has_level1_titles": "yes or no",
    "has_level2_titles": "yes or no",
    "reasoning": "Why you think titles exist or not, 300 word explanation"
}}
"""

# ---------------------------------- Other Config ------------------------------------
SEPARATOR_LENGTH: int = 60
SHOW_PROGRESS: bool = True

# ================================================================================================
# ================================================================================================

import sys
import re
import time
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
EXAMPLE_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))
from typing import Optional, List, Tuple
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
    if not _backend_initialized:
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
    """
    Call model for Q&A (each call is independent, no memory)
    
    Args:
        prompt: User input question
        max_new_tokens: Max tokens to generate (optional, defaults to MAX_GENERATE_TOKENS)
    
    Returns:
        Model response
    """
    _init_sglang_backend()
    generate_max_tokens = max_new_tokens if max_new_tokens is not None else MAX_GENERATE_TOKENS
    try:
        res = _sglang_single_chat(prompt=prompt, max_tokens=generate_max_tokens)
        # Handle different return formats
        if isinstance(res, dict):
            answer = res.get("response", "")
        elif hasattr(res, "response"):
            answer = res.response
        elif hasattr(res, "__getitem__"):
            try:
                answer = res["response"]
            except (KeyError, TypeError):
                answer = str(res)
        elif isinstance(res, str):
            answer = res
        else:
            answer = str(res)
        
        # Clean answer
        answer = answer.strip() if answer else ""
        
        # If answer contains "USER:" or "ASSISTANT:", keep only the first response
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


def split_into_pages(file_path: str) -> List[str]:
    """
    Read file, split by page markers into page list (no page marker info included)
    Content between >>>PAGE 1<<< and >>>PAGE 2<<< is one page
    
    Args:
        file_path: File path
        
    Returns:
        Page content list, each element is one page's text (no page markers)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Use regex to match page markers like ">>>PAGE 1<<<"
        pattern = re.escape(PAGE_MARKER_PREFIX) + r"\d+" + re.escape(PAGE_MARKER_SUFFIX)
        
        # Find all page marker positions
        matches = list(re.finditer(pattern, content))
        
        if not matches:
            # If no page markers found, treat entire content as one page
            if content.strip():
                pages = [content.strip()]
            else:
                pages = []
            return pages
        
        pages = []
        # Starting from first page marker, extract each page's content
        for i, match in enumerate(matches):
            # After current page marker (content start after marker)
            start_pos = match.end()
            # Next page marker position, or end of file
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(content)
            
            # Extract page content (without page marker)
            page_content = content[start_pos:end_pos].strip()
            if page_content:  # Only keep non-empty pages
                pages.append(page_content)
        
        return pages
    except Exception as e:
        print(f"[Error] Failed to read file: {e}")
        return []


def format_prompt(pages: List[str]) -> str:
    """
    Format prompt based on page list
    
    Args:
        pages: Page content list (max 4 pages)
        
    Returns:
        Formatted prompt
    """
    num_pages = len(pages)
    
    # Ensure max 4 pages
    if num_pages > PAGES_PER_BATCH:
        pages = pages[:PAGES_PER_BATCH]
        num_pages = PAGES_PER_BATCH
    
    # Pad to 4 pages (fill empty strings if fewer)
    while len(pages) < PAGES_PER_BATCH:
        pages.append("")
    
    return PROMPT_TEMPLATE.format(
        num_pages,
        num_pages,
        pages[0],
        pages[1],
        pages[2],
        pages[3]
    )


def judge_pages_batch(page_groups: List[List[str]]) -> List[str]:
    """
    Batch parallel judgment for whether multiple page groups contain titles
    
    Args:
        page_groups: List of page groups, each element is a page list (max 4 pages)
        
    Returns:
        Judgment result list, matching input order
    """
    _init_sglang_backend()
    
    # Prepare all prompts
    prompts = []
    for pages in page_groups:
        prompt = format_prompt(pages)
        prompts.append(prompt)
    
    results = [None] * len(prompts)  # Pre-allocate result list, preserve order
    batch_indices = list(range(len(prompts)))
    
    # Use thread pool to send requests concurrently (all groups within current batch)
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(call_model, prompt=prompt, max_new_tokens=MAX_GENERATE_TOKENS): idx
            for idx, prompt in zip(batch_indices, prompts)
        }
        
        # Collect results, preserving order
        batch_results = {}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                ans = future.result()
                batch_results[idx] = ans
            except Exception as exc:
                if SHOW_PROGRESS:
                    print(f"    [Error] Call failed for index {idx}: {exc}", flush=True)
                batch_results[idx] = ""
        
        # Fill results by index order
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
    
    Args:
        model_response: Full model response
        
    Returns:
        (result dict, is_format_valid)
        result dict contains "has_level1_titles", "has_level2_titles", "reasoning"
    """
    result = {
        "has_level1_titles": "no",
        "has_level2_titles": "no",
        "reasoning": ""
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
        if "has_level1_titles" not in data or "has_level2_titles" not in data:
            return result, ok
        
        # Extract values
        level1_value = str(data.get("has_level1_titles", "no") or "no").strip()
        level2_value = str(data.get("has_level2_titles", "no") or "no").strip()
        
        if level1_value not in ["yes", "no"] or level2_value not in ["yes", "no"]:
            return result, ok
        
        result["has_level1_titles"] = level1_value
        result["has_level2_titles"] = level2_value
        result["reasoning"] = str(data.get("reasoning", "") or "").strip()
        ok = True
        
    except Exception:
        # Parse failed, return False directly
        pass
    
    return result, ok


def format_duration(seconds: float) -> str:
    """
    Format seconds into "HHh-MMm-SSs-mmmms"
    """
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h:02d}h-{m:02d}m-{s:02d}s-{ms:03d}ms"


def main():
    """Main function: read file, split by pages, judge every 4 pages for titles"""
    if SHOW_PROGRESS:
        print("=" * SEPARATOR_LENGTH)
        print("Document Title Detection System")
        print("=" * SEPARATOR_LENGTH)
    
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    input_file_path = Path(INPUT_FILE)
    if not input_file_path.is_absolute():
        input_file_path = script_dir / INPUT_FILE
    
    if not input_file_path.exists():
        print(f"[Error] File not found: {input_file_path}")
        sys.exit(1)
    
    if SHOW_PROGRESS:
        print(f"Input file: {input_file_path}")
        print("=" * SEPARATOR_LENGTH)
    
    # Initialize backend connection
    try:
        _init_sglang_backend()
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("\nPlease start the SGLang server first, e.g.:")
        print(f"python -m sglang.launch_server --model-path {MODEL_PATH} --port 30000")
        sys.exit(1)
    
    # Read file and split by pages
    if SHOW_PROGRESS:
        print("Reading file and splitting by pages...")
    pages = split_into_pages(str(input_file_path))
    
    if len(pages) == 0:
        print(f"[Error] No page content found in file.")
        sys.exit(1)
    
    if SHOW_PROGRESS:
        print(f"Found {len(pages)} pages")
        print(f"Judging every {PAGES_PER_BATCH} pages per group")
        print(f"Concurrent processing count: {BATCH_SIZE}")
        print("=" * SEPARATOR_LENGTH)
    
    # Group pages, every 4 pages per group
    page_groups = []
    for i in range(0, len(pages), PAGES_PER_BATCH):
        group = pages[i:i + PAGES_PER_BATCH]
        page_groups.append(group)
    
    total_groups = len(page_groups)
    total_score = total_groups * 2  # Total score = batch count * 2
    threshold_score = total_score * THRESHOLD  # Threshold score
    
    if SHOW_PROGRESS:
        print(f"Divided into {total_groups} groups")
        print(f"Total score: {total_score}")
        print(f"Threshold: {THRESHOLD * 100}% (reaching {threshold_score} points means has titles)")
        print(f"Level-1 title weight: {LEVEL1_WEIGHT} points/time")
        print(f"Level-2 title weight: {LEVEL2_WEIGHT} points/time")
        print("-" * SEPARATOR_LENGTH)
    
    # Judge batch by batch, stop early if threshold reached
    if SHOW_PROGRESS:
        print("\nStarting batch-by-batch judgment...")
        print("-" * SEPARATOR_LENGTH)
    
    try:
        start_time = time.time()
        parsed_results = []
        current_score = 0
        early_stop = False
        
        # Process batch by batch
        for group_idx in range(0, total_groups, BATCH_SIZE):
            batch_groups = page_groups[group_idx:group_idx + BATCH_SIZE]
            batch_num = group_idx // BATCH_SIZE + 1
            total_batches = (total_groups + BATCH_SIZE - 1) // BATCH_SIZE
            
            if SHOW_PROGRESS:
                print(f"\nBatch processing progress: {batch_num}/{total_batches} batches (current score: {current_score}/{total_score}, {current_score/total_score*100:.1f}%)", flush=True)
            
            # Call model to batch judge current batch
            batch_results = judge_pages_batch(batch_groups)
            
            # First parse round: collect success and failure indices
            batch_parsed_list = [None] * len(batch_results)
            failed_indices = []
            
            for i, result in enumerate(batch_results):
                parsed, ok = parse_result(result)
                if ok:
                    batch_parsed_list[i] = parsed
                else:
                    failed_indices.append(i)
                    if SHOW_PROGRESS:
                        actual_group_idx = group_idx + i
                        print(f"[Warning] Pages {actual_group_idx*PAGES_PER_BATCH+1}-{min((actual_group_idx+1)*PAGES_PER_BATCH, len(pages))} format invalid, will retry once")
            
            # Retry format-invalid ones (only once)
            if failed_indices:
                if SHOW_PROGRESS:
                    print(f"  Found {len(failed_indices)} groups with invalid format, retrying (once each)...")
                
                for i in failed_indices:
                    actual_group_idx = group_idx + i
                    if SHOW_PROGRESS:
                        print(f"  [Retry] Pages {actual_group_idx*PAGES_PER_BATCH+1}-{min((actual_group_idx+1)*PAGES_PER_BATCH, len(pages))}...")
                    
                    # Regenerate prompt and call model
                    retry_prompt = format_prompt(batch_groups[i])
                    retry_response = call_model(retry_prompt, MAX_GENERATE_TOKENS)
                    parsed, ok = parse_result(retry_response)
                    
                    if ok:
                        batch_parsed_list[i] = parsed
                        if SHOW_PROGRESS:
                            print(f"  [Success] Pages {actual_group_idx*PAGES_PER_BATCH+1}-{min((actual_group_idx+1)*PAGES_PER_BATCH, len(pages))} retry succeeded")
                    else:
                        # Still failed after retry, use default values
                        batch_parsed_list[i] = {
                            "has_level1_titles": "no",
                            "has_level2_titles": "no",
                            "reasoning": ""
                        }
                        if SHOW_PROGRESS:
                            print(f"  [Skipped] Pages {actual_group_idx*PAGES_PER_BATCH+1}-{min((actual_group_idx+1)*PAGES_PER_BATCH, len(pages))} still failed after retry, using defaults")
            
            # Parse results and calculate score
            for i, parsed in enumerate(batch_parsed_list):
                actual_group_idx = group_idx + i
                parsed["page_range"] = f"Pages {actual_group_idx*PAGES_PER_BATCH+1}-{min((actual_group_idx+1)*PAGES_PER_BATCH, len(pages))}"
                parsed_results.append(parsed)
                
                # Calculate score
                if parsed["has_level1_titles"] == "yes":
                    current_score += LEVEL1_WEIGHT
                if parsed["has_level2_titles"] == "yes":
                    current_score += LEVEL2_WEIGHT
                
                if SHOW_PROGRESS:
                    print(f"\n[{parsed['page_range']}]")
                    print(f"  Has level-1 titles: {parsed['has_level1_titles']}")
                    print(f"  Has level-2 titles: {parsed['has_level2_titles']}")
                    print(f"  Reasoning: {parsed.get('reasoning', '')}")
                    print(f"  Cumulative score: {current_score}/{total_score} ({current_score/total_score*100:.1f}%)")
                
                # Check if threshold reached
                if current_score >= threshold_score:
                    early_stop = True
                    if SHOW_PROGRESS:
                        print(f"\n>>> Score reached threshold ({current_score} >= {threshold_score}), early stopping <<<")
                    break
            
            if early_stop:
                break
        
        judge_duration = time.time() - start_time
        
        if SHOW_PROGRESS:
            print("\n" + "=" * SEPARATOR_LENGTH)
            print("Judgment complete!")
            print(f"Total judgment time: {format_duration(judge_duration)}")
            print(f"Groups actually processed: {len(parsed_results)}/{total_groups}")
            print(f"Final score: {current_score}/{total_score} ({current_score/total_score*100:.1f}%)")
            print("=" * SEPARATOR_LENGTH)
        
        # Determine if document has titles based on score
        has_title = current_score >= threshold_score
        
        if SHOW_PROGRESS:
            print("\n" + "=" * SEPARATOR_LENGTH)
            print("Overall judgment result:")
            print(f"  Document verdict: {'Has Titles' if has_title else 'No Titles'}")
            print(f"  Basis: Score {current_score}/{total_score} ({current_score/total_score*100:.1f}%) {'meets' if has_title else 'does not meet'} threshold {THRESHOLD*100}%")
            print("=" * SEPARATOR_LENGTH)
        
        # Rename file based on judgment result
        if has_title:
            suffix = "_Title"
        else:
            suffix = "_NoTitle"
        
        # Build new file path
        new_file_path = input_file_path.parent / (input_file_path.stem + suffix + input_file_path.suffix)
        
        # Rename original file
        input_file_path.rename(new_file_path)
        
        if SHOW_PROGRESS:
            print(f"\nFile renamed to: {new_file_path}")
            print("=" * SEPARATOR_LENGTH)
        
    except Exception as e:
        print(f"\n[Error] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if SHOW_PROGRESS:
        print("All processing complete!")
        print("=" * SEPARATOR_LENGTH)


if __name__ == "__main__":
    main()
