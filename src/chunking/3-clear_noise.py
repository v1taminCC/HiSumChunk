#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect and remove headers/footers (LLM-based identification + rule validation)

Workflow:
1) Read txt with page markers (page markers like: >>>PAGE 1<<<)
2) Split by page (keep page markers, but don't feed markers to model for identification)
3) Take first N pages, group by 4 pages each, let model identify header/footer candidate "lines" (return JSON array)
4) Global statistics: count header and footer occurrences separately, take top 3 each
5) Global deletion: delete matching top-3 header/footer content from all pages
6) Group validation deletion: validate with global frequency, only delete candidates "repeated across many pages"
7) Delete matching lines within each page's top/bottom window, output new file (keep page markers)
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
CATALOG_BATCH_SIZE: int = 16
PAGES_PER_BATCH: int = 4

# ----------------------------------- File Config ------------------------------------
INPUT_FILE: str = str(EXAMPLE_BOOK_DIR / "work_txt" / "Far From the Madding Crowd_MergeRule_Title.txt")
OUTPUT_SUFFIX: str = "_NoHeaderFooter"
OUTPUT_ENCODING: str = "utf-8"

# ----------------------------- Page Marker Config -----------------------------------
PAGE_MARKER_PREFIX: str = ">>>PAGE "
PAGE_MARKER_SUFFIX: str = "<<<"

# ----------------------------- Detection Range Config -------------------------------
# Whether to use "all pages" to judge headers/footers; if False, only use first 4 pages (cheaper but may miss)
DETECT_ALL_PAGES: bool = True

# ------------------------------ Judgment Prompt Template ----------------------------
PROMPT_TEMPLATE: str = """
Below are {} consecutive pages of a document. Based on these {} consecutive pages, determine whether the document contains headers, footers, or page numbers. 
Given text: 
Page 1 "{first}" 
Page 2 "{second}" 
Page 3 "{third}"
Page 4 "{fourth}"
Return the following JSON format:
{{
    "page_1_has_header": "yes or no",
    "page_1_header_content": "Specific header content. If a header exists, return its specific content; otherwise return empty.",
    "page_1_has_footer": "yes or no",
    "page_1_footer_content": "Specific footer content. If a footer exists, return its specific content; otherwise return empty.",
    "page_2_has_header": "yes or no",
    "page_2_header_content": "Specific header content. If a header exists, return its specific content; otherwise return empty.",
    "page_2_has_footer": "yes or no",
    "page_2_footer_content": "Specific footer content. If a footer exists, return its specific content; otherwise return empty.",
    "page_3_has_header": "yes or no",
    "page_3_header_content": "Specific header content. If a header exists, return its specific content; otherwise return empty.",
    "page_3_has_footer": "yes or no",
    "page_3_footer_content": "Specific footer content. If a footer exists, return its specific content; otherwise return empty.",
    "page_4_has_header": "yes or no",
    "page_4_header_content": "Specific header content. If a header exists, return its specific content; otherwise return empty.",
    "page_4_has_footer": "yes or no",
    "page_4_footer_content": "Specific footer content. If a footer exists, return its specific content; otherwise return empty.",
    "reasoning": "Your reasoning, 500 word explanation"
}}
""".strip()



# ----------------------------- Catalog Detection Prompt Template --------------------
CATALOG_PROMPT_TEMPLATE: str = """
Below is one page of a document. Based on this page, determine whether it contains a table of contents / page-number index (e.g., abundant "numbers", "leader dots", "dotted lines", "ellipses", "guide lines", etc.).
Note: standalone page numbers appearing in headers/footers do NOT count as a table of contents index.
Given text: 
"{page}" 
Return the following JSON format:
{{
    "answer": "yes or no",
    "reasoning": "Why you think this page does or does not contain a catalog/table-of-contents-like page-number index, 300 word explanation"
}}
""".strip()

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
from typing import Optional, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

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
    try:
        res = _sglang_single_chat(prompt=prompt, max_tokens=generate_max_tokens)
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
    except Exception as exc:
        # Fallback: on model call exception, return empty string and print error
        if SHOW_PROGRESS:
            print(f"[Error] Model call failed: {exc}")
        answer = ""

    # Only keep these two basic denoising lines (remove if not needed), all other processing removed
    answer = answer.strip() if answer else ""
    return answer


def _page_marker_pattern() -> str:
    return re.escape(PAGE_MARKER_PREFIX) + r"\d+" + re.escape(PAGE_MARKER_SUFFIX)


def split_pages_with_markers(file_path: str) -> List[Tuple[str, str]]:
    """
    Split by page markers, returns [(marker, page_text), ...]
    page_text does not include the marker.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[Error] Failed to read file: {e}")
        return []

    pattern = _page_marker_pattern()
    matches = list(re.finditer(pattern, content))
    if not matches:
        text = content.strip()
        return [("", text)] if text else []

    pages: List[Tuple[str, str]] = []
    for i, match in enumerate(matches):
        marker = match.group(0)
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        page_text = content[start_pos:end_pos].strip()
        pages.append((marker, page_text))
    return pages


def extract_page_number(marker: str) -> str:
    """Extract real page number from page marker, e.g., >>>PAGE 17<<< -> 17"""
    pattern = re.escape(PAGE_MARKER_PREFIX) + r"(\d+)" + re.escape(PAGE_MARKER_SUFFIX)
    match = re.search(pattern, marker)
    return match.group(1) if match else "unknown"


def _normalize_line(s: str) -> str:
    # For matching: strip leading/trailing whitespace, collapse consecutive whitespace into one space
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def format_prompt(pages_text: List[str]) -> str:
    """Format according to prompt template (max 4 pages, fill empty if fewer; support named placeholders)"""
    pages = pages_text[:PAGES_PER_BATCH]
    while len(pages) < PAGES_PER_BATCH:
        pages.append("")  # Fewer than 4 pages: remaining placeholders empty
    num_pages = min(len(pages_text), PAGES_PER_BATCH)
    # Compatible with template having both {} and {first}/{second}/{third}/{fourth}
    return PROMPT_TEMPLATE.format(
        num_pages,
        num_pages,
        first=pages[0],
        second=pages[1],
        third=pages[2],
        fourth=pages[3],
    )


def format_catalog_prompt(page_text: str) -> str:
    """Catalog detection prompt formatting (single page)"""
    return CATALOG_PROMPT_TEMPLATE.format(page=page_text)


def judge_batches(prompts: List[str], batch_size: int = None, print_raw: bool = True) -> List[str]:
    """
    Parallel model calls, preserving order
    
    Args:
        prompts: prompt list
        batch_size: concurrent request count, defaults to BATCH_SIZE
        print_raw: whether to print raw response in terminal, default True
    """
    _init_sglang_backend()
    if batch_size is None:
        batch_size = BATCH_SIZE
    results: List[Optional[str]] = [None] * len(prompts)
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in range(0, len(prompts), batch_size):
        batch_num = i // batch_size + 1
        batch_prompts = prompts[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(prompts))))
        if SHOW_PROGRESS:
            print(f"  Batch progress: {batch_num}/{total_batches} - processing concurrently...", flush=True)
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_index = {
                executor.submit(call_model, prompt=p, max_new_tokens=MAX_GENERATE_TOKENS): idx
                for idx, p in zip(batch_indices, batch_prompts)
            }
            batch_results: Dict[int, str] = {}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    batch_results[idx] = future.result()
                    if SHOW_PROGRESS and print_raw:
                        print("=" * SEPARATOR_LENGTH)
                        print(f"[Raw model response] Group {idx + 1}")
                        print(batch_results[idx])
                        print("=" * SEPARATOR_LENGTH)
                except Exception as exc:
                    if SHOW_PROGRESS:
                        print(f"    [Error] idx={idx} call failed: {exc}", flush=True)
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


def parse_header_footer(model_response: str) -> Tuple[dict, bool]:
    """
    Parse model response:
    Returns: (result dict, is_format_valid)
    Result dict format:
    {
      "pages": [
        {"has_header": "yes or no", "header_content": "...", "has_footer": "yes or no", "footer_content": "..."},
        ...
      ],
      "reasoning": "..."
    }
    """
    result = {
        "pages": [
            {"has_header": "no", "header_content": "", "has_footer": "no", "footer_content": ""} for _ in range(4)
        ],
        "reasoning": "",
    }
    ok = False
    
    try:
        # Extract complete JSON object
        json_str = _extract_json_from_text(model_response)
        if json_str:
            # Tolerance: replace Chinese punctuation
            json_str = json_str  # already uses English punctuation
            # Try parsing
            data = json.loads(json_str)
            if isinstance(data, dict):
                # Check if required page-level fields exist (at least one page with data)
                page_names = ["page_1", "page_2", "page_3", "page_4"]
                has_page_data = False
                
                for i, pn in enumerate(page_names):
                    has_h = str(data.get(f"{pn}_has_header", "no") or "no").strip()
                    h_txt = str(data.get(f"{pn}_header_content", "") or "").strip()
                    has_f = str(data.get(f"{pn}_has_footer", "no") or "no").strip()
                    f_txt = str(data.get(f"{pn}_footer_content", "") or "").strip()
                    
                    result["pages"][i] = {
                        "has_header": has_h,
                        "header_content": h_txt,
                        "has_footer": has_f,
                        "footer_content": f_txt,
                    }
                    
                    # If at least one page has data, consider format valid
                    if has_h != "no" or has_f != "no" or h_txt or f_txt:
                        has_page_data = True
                
                result["reasoning"] = str(data.get("reasoning", "") or "").strip()
                
                # If at least one page has data, consider format valid
                if has_page_data or result["reasoning"]:
                    ok = True
    except Exception as e:
        # Fallback: extract from text
        if SHOW_PROGRESS:
            print(f"[Warning] Failed to parse header/footer response: {e}")
        think_match = re.search(r"(?:reasoning)\s*[:]\s*(.+)", model_response, re.DOTALL)
        if think_match:
            result["reasoning"] = think_match.group(1).strip()
    
    return result, ok


def parse_catalog_response(model_response: str) -> Tuple[dict, bool]:
    """
    Strictly parse catalog detection response (single page format):
    Only do strict JSON parsing, no regex matching or tolerance
    
    Returns: (result dict, is_format_valid)
    Result dict format:
    {
      "answer": "yes or no",
      "reasoning": "..."
    }
    """
    result = {
        "answer": "no",
        "reasoning": ""
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
        key = "answer"
        if key not in data:
            return result, ok
        
        # Extract value
        value = data.get(key)
        if value not in ["yes", "no"]:
            return result, ok
        
        result[key] = str(value).strip()
        result["reasoning"] = str(data.get("reasoning", "") or "").strip()
        ok = True
        
    except Exception:
        # Parse failed, return False directly
        pass
    
    return result, ok


def _is_none_value(s: str) -> bool:
    s2 = (s or "").strip()
    return s2 in ("", "none", "None", "null")


def remove_header_footer_from_page(
    page_text: str,
    header_text: str,
    footer_text: str,
    page_num: str = "unknown"
) -> Tuple[str, int]:
    """
    Fixed version: Delete headers/footers by complete text blocks (supports multi-segment content)
    No normalization, strict match on raw text blocks, returns (new_text, deleted_line_count)
    """
    if not page_text:
        return "", 0

    raw_lines = page_text.splitlines()
    new_lines = raw_lines.copy()
    removed = 0
    removed_lines = []

    # Define all target text blocks to delete
    delete_targets = []
    if not _is_none_value(header_text):
        delete_targets.append(("header", header_text))
    if not _is_none_value(footer_text):
        delete_targets.append(("footer", footer_text))

    # Iterate over all text blocks to delete, delete one by one
    for target_type, target_text in delete_targets:
        if not target_text:
            continue
            
        target_lines = target_text.splitlines()
        target_line_count = len(target_lines)
        if target_line_count == 0:
            continue

        # Method 1: Match complete text block (priority)
        if target_text in page_text:
            # Directly delete the entire text block
            new_text = page_text.replace(target_text, "")
            # Count deleted lines
            removed += target_line_count
            if SHOW_PROGRESS:
                print(f"[Page {page_num}] Matched complete {target_type} text block, deleted {target_line_count} lines")
            raw_lines = new_text.splitlines()
            new_lines = raw_lines
            continue

        # Method 2: Line-by-line match deletion (fallback)
        for i, line in enumerate(new_lines):
            if line.strip() == target_text.strip() and not _is_none_value(line):
                removed_lines.append(i)
                removed += 1
                if SHOW_PROGRESS:
                    print(f"[Page {page_num}] Line-by-line match, deleted {target_type} line: {line[:50]}...")

    # Reverse delete (avoid index shifting)
    for idx in sorted(removed_lines, reverse=True):
        if 0 <= idx < len(new_lines):
            del new_lines[idx]

    new_text = "\n".join(new_lines).strip()
    return new_text, removed


def global_remove_top3_header_footer(
    pages: List[Tuple[str, str]],
    top3_header: List[str],
    top3_footer: List[str]
) -> List[Tuple[str, str]]:
    """
    Fixed version: Iterate over each top3 content, delete individually (not concatenated)
    """
    processed_pages = []
    total_global_removed = 0

    for marker, page_text in pages:
        page_num = extract_page_number(marker)
        current_text = page_text
        current_removed = 0

        # Delete top3 headers one by one
        for idx, header in enumerate(top3_header):
            if _is_none_value(header):
                continue
            new_text, removed = remove_header_footer_from_page(current_text, header, "", page_num)
            if removed > 0:
                if SHOW_PROGRESS:
                    print(f"[Page {page_num}] Globally deleted #{idx+1} header, removed {removed} lines")
                current_removed += removed
                current_text = new_text

        # Delete top3 footers one by one
        for idx, footer in enumerate(top3_footer):
            if _is_none_value(footer):
                continue
            new_text, removed = remove_header_footer_from_page(current_text, "", footer, page_num)
            if removed > 0:
                if SHOW_PROGRESS:
                    print(f"[Page {page_num}] Globally deleted #{idx+1} footer, removed {removed} lines")
                current_removed += removed
                current_text = new_text

        processed_pages.append((marker, current_text))
        total_global_removed += current_removed

    if SHOW_PROGRESS:
        print(f"\nGlobal deletion complete, total {total_global_removed} lines removed")
    return processed_pages


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
    if SHOW_PROGRESS:
        print("=" * SEPARATOR_LENGTH)
        print("Header/Footer Detection and Cleanup (Global-Vote Priority Deletion)")
        print("=" * SEPARATOR_LENGTH)

    script_dir = Path(__file__).parent.absolute()
    input_path = Path(INPUT_FILE)
    if not input_path.is_absolute():
        input_path = script_dir / INPUT_FILE
    if not input_path.exists():
        print(f"[Error] File not found: {input_path}")
        sys.exit(1)

    pages = split_pages_with_markers(str(input_path))
    if not pages:
        print("[Error] Could not read any page content")
        sys.exit(1)

    if SHOW_PROGRESS:
        print(f"Input file: {input_path}")
        print(f"Found {len(pages)} pages")
        print(f"Header/footer detection concurrency: {BATCH_SIZE}")
        print(f"Catalog detection concurrency: {CATALOG_BATCH_SIZE}")
        print("=" * SEPARATOR_LENGTH)

    # -------------------------- Catalog Detection --------------------------
    detect_pages_for_catalog = pages if DETECT_ALL_PAGES else pages[:PAGES_PER_BATCH]

    if SHOW_PROGRESS:
        print("-" * SEPARATOR_LENGTH)
        scope = "all pages" if DETECT_ALL_PAGES else "first 4 pages"
        print(f"Starting catalog page detection (using {scope}, one page at a time, concurrency={CATALOG_BATCH_SIZE})...")

    catalog_detect_start = time.time()
    
    # Generate prompts for each page, recording page info
    catalog_prompts = []
    catalog_page_info = []  # Store (marker, page_num) info for each page
    for marker, page_text in detect_pages_for_catalog:
        catalog_prompts.append(format_catalog_prompt(page_text))
        page_num = extract_page_number(marker)
        catalog_page_info.append((marker, page_num))
    
    # Batch call model (don't print raw response)
    catalog_responses = judge_batches(catalog_prompts, batch_size=CATALOG_BATCH_SIZE, print_raw=False) if catalog_prompts else []
    
    # First parse round
    # Initialize result list, keeping same index as catalog_responses
    catalog_parsed_list = [None] * len(catalog_responses)
    failed_indices = []  # Record format-invalid indices
    error_responses = []  # Collect all error responses [(page_num, response), ...]
    success_responses = []  # Collect all success responses [(page_num, response), ...]
    
    for idx, response in enumerate(catalog_responses):
        parsed, ok = parse_catalog_response(response)
        if ok:
            catalog_parsed_list[idx] = parsed
            page_num = catalog_page_info[idx][1]
            success_responses.append((page_num, response))
        else:
            failed_indices.append(idx)
            page_num = catalog_page_info[idx][1]
            error_responses.append((page_num, response))
            if SHOW_PROGRESS:
                print(f"[Warning] Catalog detection page {page_num} format invalid, will retry once")
    
    # Retry format-invalid ones (only once)
    retry_success_indices = []
    skipped_indices = []  # Indices still failed after retry (skip this page)
    
    if failed_indices:
        if SHOW_PROGRESS:
            print(f"\nFound {len(failed_indices)} pages with invalid format, retrying (once each)...")
        
        for idx in failed_indices:
            page_num = catalog_page_info[idx][1]
            if SHOW_PROGRESS:
                print(f"  [Retry] Catalog detection page {page_num}...")
            
            retry_response = call_model(catalog_prompts[idx], MAX_GENERATE_TOKENS)
            parsed, ok = parse_catalog_response(retry_response)
            
            if ok:
                catalog_parsed_list[idx] = parsed
                retry_success_indices.append(idx)
                success_responses.append((page_num, retry_response))
                if SHOW_PROGRESS:
                    print(f"  [Success] Catalog detection page {page_num} retry succeeded")
            else:
                skipped_indices.append(idx)
                error_responses.append((page_num, retry_response))
                # Create default result for skipped pages (don't process, don't delete)
                catalog_parsed_list[idx] = {
                    "answer": "no",
                    "reasoning": ""
                }
                if SHOW_PROGRESS:
                    print(f"  [Skipped] Catalog detection page {page_num} still failed after retry, skipping")

    catalog_detect_duration = time.time() - catalog_detect_start
    
    # Extract catalog page indices (skip those that were skipped)
    catalog_page_indices = set()
    key = "answer"
    for idx, parsed in enumerate(catalog_parsed_list):
        # Skip pages that failed parsing (don't process, don't delete)
        if idx in skipped_indices:
            continue
        if parsed and str(parsed.get(key, "no") or "no").strip() == "yes":
            catalog_page_indices.add(idx)

    removed_catalog_pages = []
    # Need to delete from original pages, so find corresponding indices
    for idx in sorted(catalog_page_indices, reverse=True):
        if 0 <= idx < len(detect_pages_for_catalog):
            # Find position in original pages
            marker, _ = detect_pages_for_catalog[idx]
            page_num = extract_page_number(marker)
            # In original pages, find and delete
            for i, (m, _) in enumerate(pages):
                if m == marker:
                    pages.pop(i)
                    removed_catalog_pages.append(page_num)
                    break

    if SHOW_PROGRESS:
        print("-" * SEPARATOR_LENGTH)
        if removed_catalog_pages:
            print(f"Catalog pages detected, removed page numbers: {removed_catalog_pages}")
        else:
            print("No catalog pages detected for removal")
        print(f"After catalog removal: {len(pages)} pages remaining")
        print("-" * SEPARATOR_LENGTH)
    
    # Output parse result statistics
    total_pages = len(catalog_responses)
    first_success = total_pages - len(failed_indices)
    if SHOW_PROGRESS:
        print("-" * SEPARATOR_LENGTH)
        print("Catalog detection - Parse result statistics:")
        print(f"  First parse success: {first_success}/{total_pages}")
        if failed_indices:
            print(f"  First parse failure: {len(failed_indices)}/{total_pages}")
            print(f"  Retry succeeded: {len(retry_success_indices)}/{len(failed_indices)}")
            print(f"  Skipped pages: {len(skipped_indices)}/{len(failed_indices)}")
        print("-" * SEPARATOR_LENGTH)
        print(f"Catalog detection time: {format_duration(catalog_detect_duration)}")
    

    # -------------------------- Header/Footer Detection --------------------------
    header_footer_start_time = time.time()
    
    detect_pages = pages if DETECT_ALL_PAGES else pages[:PAGES_PER_BATCH]
    page_groups: List[List[str]] = []
    for i in range(0, len(detect_pages), PAGES_PER_BATCH):
        group = [t for _, t in detect_pages[i:i + PAGES_PER_BATCH]]
        page_groups.append(group)

    if SHOW_PROGRESS:
        print("-" * SEPARATOR_LENGTH)
        scope = "all pages" if DETECT_ALL_PAGES else "first 4 pages"
        print(f"Starting header/footer detection (using {scope}, {PAGES_PER_BATCH} pages per group, concurrency={BATCH_SIZE})...")

    prompts = [format_prompt(g) for g in page_groups]
    responses = judge_batches(prompts) if prompts else []
    parsed_list = [parse_header_footer(r)[0] for r in responses]  # Only take result dict, ignore ok flag

    # -------------------------- Global Statistics: Headers/Footers, Take Top 3 --------------------------
    header_counter = Counter()
    footer_counter = Counter()

    # Iterate over all detected pages, collect valid header/footer content
    for group_idx, parsed in enumerate(parsed_list):
        group_start = group_idx * PAGES_PER_BATCH
        group_end = min(group_start + PAGES_PER_BATCH, len(detect_pages))
        for page_in_group_idx in range(group_end - group_start):
            page_info = parsed["pages"][page_in_group_idx]
            header_content = page_info["header_content"].strip()
            footer_content = page_info["footer_content"].strip()

            # Only add valid content to statistics (no normalization)
            if not _is_none_value(header_content):
                header_counter[header_content] += 1
            if not _is_none_value(footer_content):
                footer_counter[footer_content] += 1

    # Get top 3 most frequent headers/footers
    top3_header = [content for content, _ in header_counter.most_common(3)]
    top3_footer = [content for content, _ in footer_counter.most_common(3)]

    if SHOW_PROGRESS:
        print("-" * SEPARATOR_LENGTH)
        print(f"Global statistics (no normalization):")
        print(f"Top 3 headers: {[h[:50]+'...' if len(h)>50 else h for h in top3_header] if top3_header else 'none'}")
        print(f"Top 3 footers: {[f[:50]+'...' if len(f)>50 else f for f in top3_footer] if top3_footer else 'none'}")
        print("-" * SEPARATOR_LENGTH)

    # -------------------------- Execute Global Deletion --------------------------
    pages_after_global_remove = global_remove_top3_header_footer(pages, top3_header, top3_footer)
    if SHOW_PROGRESS:
        print("Global top-3 header/footer deletion completed")
        print("=" * SEPARATOR_LENGTH)

    # Print identification results
    if SHOW_PROGRESS:
        print("-" * SEPARATOR_LENGTH)
        print(f"Detection groups: {len(page_groups)}")
        for idx, parsed in enumerate(parsed_list, 1):
            group_start = (idx - 1) * PAGES_PER_BATCH
            pages_info = parsed.get("pages", []) or []
            for j in range(min(4, len(pages_info))):
                if group_start + j >= len(detect_pages):
                    break
                # Extract real page number
                marker = detect_pages[group_start + j][0]
                real_page_num = extract_page_number(marker)
                p = pages_info[j] or {}
                print(f"[Page {real_page_num}] Has header: {p.get('has_header', '')}")
                print(f"[Page {real_page_num}] Header content: {p.get('header_content', '')[:100]}..." if len(p.get('header_content', ''))>100 else f"[Page {real_page_num}] Header content: {p.get('header_content', '')}")
                print(f"[Page {real_page_num}] Has footer: {p.get('has_footer', '')}")
                print(f"[Page {real_page_num}] Footer content: {p.get('footer_content', '')[:100]}..." if len(p.get('footer_content', ''))>100 else f"[Page {real_page_num}] Footer content: {p.get('footer_content', '')}")
            print(f"[Group {idx}] Reasoning: {parsed.get('reasoning', '')[:200]}..." if len(parsed.get('reasoning', ''))>200 else f"[Group {idx}] Reasoning: {parsed.get('reasoning', '')}")
            print("-" * SEPARATOR_LENGTH)
        print("=" * SEPARATOR_LENGTH)

    # Execute group-based deletion (on globally-deleted pages)
    cleaned_pages: List[Tuple[str, str]] = []
    total_removed_lines = 0
    target_pages = pages_after_global_remove

    for page_idx, (marker, text) in enumerate(target_pages):
        page_num = extract_page_number(marker)
        header_text = ""
        footer_text = ""

        if DETECT_ALL_PAGES:
            group_idx = min(page_idx // PAGES_PER_BATCH, len(parsed_list) - 1 if parsed_list else 0)
            in_group_idx = page_idx % PAGES_PER_BATCH
        else:
            # Only detected first 4 pages, others are not deleted
            group_idx = 0
            if page_idx >= len(parsed_list) * PAGES_PER_BATCH:
                header_text = ""
                footer_text = ""
            else:
                in_group_idx = page_idx % PAGES_PER_BATCH

        # Get header/footer config for current page
        if DETECT_ALL_PAGES and parsed_list and group_idx < len(parsed_list):
            g = parsed_list[group_idx]
            pages_info = g.get("pages", []) if isinstance(g, dict) else []
            if in_group_idx < len(pages_info):
                p = pages_info[in_group_idx]
                header_text = p.get("header_content", "") if p.get("has_header", "no") == "yes" else ""
                footer_text = p.get("footer_content", "") if p.get("has_footer", "no") == "yes" else ""

        # Execute group deletion
        new_text, removed = remove_header_footer_from_page(text, header_text, footer_text, page_num)
        cleaned_pages.append((marker, new_text))
        total_removed_lines += removed

    header_footer_duration = time.time() - header_footer_start_time

    # Write output file
    out_path = input_path.with_name(input_path.stem + OUTPUT_SUFFIX + input_path.suffix)
    with open(out_path, "w", encoding=OUTPUT_ENCODING) as f:
        for marker, text in cleaned_pages:
            if text:
                f.write(text)
                f.write("\n")

    if SHOW_PROGRESS:
        print(f"Output file: {out_path}")
        print(f"Group deletion total lines removed: {total_removed_lines}")
        print("=" * SEPARATOR_LENGTH)
        print(f"Catalog detection time: {format_duration(catalog_detect_duration)}")
        print(f"Header/footer removal time: {format_duration(header_footer_duration)}")
        print("=" * SEPARATOR_LENGTH)
        print("Processing complete!")


if __name__ == "__main__":
    main()
