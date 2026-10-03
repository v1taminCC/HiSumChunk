#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LBDC local version — local-llm + batch pairwise paragraph judgment.
All paragraph pairs within a chapter processed in one batch, one forward pass for all judgments.

Usage:
  python3 llm_semantic_local.py --batch_size 4
"""

from __future__ import annotations

import os
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

# ============================================================
# ==== Parameters (edit here) ====
# ============================================================
MODEL_PATH         = os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507")
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
WORK_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))
INPUT_DIR          = str(WORK_BOOK_DIR)
OUTPUT_DIR         = str(WORK_BOOK_DIR)

BATCH_SIZE         = 5         # Paragraph pairs per forward pass (GPU memory constraint)
MAX_SEQ_LENGTH     = 1536      # Max tokens per prompt
MAX_PARA_CHARS     = 800       # Truncate each paragraph to this many chars (prevent context overflow)

TARGET_WORDS       = 1000      # Target chunk size in words
MIN_CHUNK_WORDS    = 200       # Merge threshold for undersized chunks
MAX_CHUNK_WORDS    = 2200      # Split threshold for oversized chunks
MERGE_SMALL_MIN    = 80        # Auto-merge very short paragraphs

PROMPT_SYSTEM = (
    "Determine whether two consecutive paragraphs discuss the SAME topic or a DIFFERENT topic. "
    "A topic change means the subject, logical phase, or core argument shifts significantly. "
    "Respond with exactly ONE word: SAME or DIFFERENT. No punctuation, no explanation."
)
# ============================================================

ANCHOR_PATTERN = re.compile(r"^<<<CHAPTER_ID:(.*?)>>>\s*$", re.MULTILINE)
WORD_PATTERN   = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

SPLIT_METHOD  = "lbd_semantic_boundary"
OUTPUT_SUFFIX = "_LBDSemantic_split.json"


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def _elapsed(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# ==================== Model loading ====================

def load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_path}")
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def gpu_mem() -> str:
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    return f"{a:.1f}/{r:.1f}G"


# ==================== Text processing ====================

def split_paragraphs(text: str) -> list[str]:
    raw = re.split(r"\n\s*\n", text)
    result = []
    for p in raw:
        p = p.strip()
        if p and count_words(p) >= 10:
            result.append(p)
        elif p and result:
            result[-1] += " " + p
    merged = []
    for p in result:
        if merged and count_words(p) < MERGE_SMALL_MIN:
            merged[-1] += "\n" + p
        else:
            merged.append(p)
    return merged


def parse_book(raw_text: str) -> list[dict]:
    matches = list(ANCHOR_PATTERN.finditer(raw_text))
    if not matches:
        return [{"chapter_id": "unknown", "text": raw_text, "offset": 0}]
    chapters = []
    for idx, m in enumerate(matches):
        cid = m.group(1).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        text = raw_text[start:end].strip()
        if text:
            chapters.append({"chapter_id": cid, "text": text, "offset": start})
    return chapters


# ==================== Post-processing ====================

def _postprocess(chunks: list[str]) -> list[str]:
    """Merge undersized chunks, split oversized chunks"""
    if not chunks:
        return chunks
    # Merge < MIN_CHUNK_WORDS
    merged = []
    for c in chunks:
        wc = count_words(c)
        if merged and wc < MIN_CHUNK_WORDS:
            merged[-1] += "\n" + c
        else:
            merged.append(c)
    # Merge undersized first chunk
    if len(merged) >= 2 and count_words(merged[0]) < MIN_CHUNK_WORDS:
        merged[1] = merged[0] + "\n" + merged[1]
        merged = merged[1:]
    # Merge undersized last chunk
    if len(merged) >= 2 and count_words(merged[-1]) < MIN_CHUNK_WORDS:
        merged[-2] += "\n" + merged[-1]
        merged = merged[:-1]
    # Split > MAX_CHUNK_WORDS
    result = []
    for c in merged:
        wc = count_words(c)
        if wc <= MAX_CHUNK_WORDS:
            result.append(c)
        else:
            paras = c.split("\n")
            mid = len(paras) // 2
            a = "\n".join(paras[:mid]).strip()
            b = "\n".join(paras[mid:]).strip()
            if a: result.append(a)
            if b:
                if count_words(b) > MAX_CHUNK_WORDS:
                    result.extend(_postprocess([b]))
                else:
                    result.append(b)
    return result


# ==================== Core batch inference ====================

def build_pair_prompt(para_a: str, para_b: str) -> str:
    return (
        f"Paragraph A:\n{para_a[-MAX_PARA_CHARS:]}\n\n"
        f"Paragraph B:\n{para_b[:MAX_PARA_CHARS]}\n\n"
        f"Same topic? (SAME/DIFFERENT):"
    )


def batch_judge_pairs(pairs: list[tuple[str, str]], model, tokenizer, batch_size: int) -> list[bool]:
    """Batch process all paragraph pairs, return is_different list"""
    decisions = []
    t0 = time.time()
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        prompts = [build_pair_prompt(a, b) for a, b in batch]

        messages_list = [
            [{"role": "system", "content": PROMPT_SYSTEM}, {"role": "user", "content": p}]
            for p in prompts
        ]
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in messages_list]

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=MAX_SEQ_LENGTH).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=3, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for i in range(len(batch)):
            out_ids = outputs[i][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(out_ids, skip_special_tokens=True).strip().upper()
            decisions.append("DIFFERENT" in response or "YES" in response)

        torch.cuda.empty_cache()
        # Progress callback
        done = start + len(batch)
        pct = done / max(len(pairs), 1) * 100
        bar_fill = int(20 * done / max(len(pairs), 1))
        bar = "█" * bar_fill + "░" * (20 - bar_fill)
        elapsed = time.time() - t0
        eta = _elapsed(elapsed / done * (len(pairs) - done)) if 0 < done < len(pairs) else ""
        print(f"\r    {bar} {done}/{len(pairs)} ({pct:.0f}%)  {gpu_mem()}  {_elapsed(elapsed)}{' ~'+eta if eta else ''}",
              end="", flush=True)

    print(flush=True)
    return decisions


def chunk_paragraphs(paras: list[str], decisions: list[bool]) -> list[str]:
    """Merge paragraphs into chunks by boundary judgments, then post-process"""
    if not decisions:
        return _postprocess(["\n".join(paras)])

    chunks_raw = []
    current = [paras[0]]
    for i in range(len(decisions)):
        if decisions[i]:
            chunks_raw.append("\n".join(current))
            current = [paras[i + 1]]
        else:
            current.append(paras[i + 1])
    if current:
        chunks_raw.append("\n".join(current))

    return _postprocess(chunks_raw)


# ==================== Book processing ====================

def _make_item(chapter: dict, text: str, idx: int) -> dict:
    return {
        "region_id": "1", "is_region_summary": False,
        "chunk_info": f"LBDSemantic---Block {idx:>3}",
        "length": count_words(text), "text": text,
        "original_length": len(text), "word_count": count_words(text),
        "active_chapter_id": chapter["chapter_id"],
        "covered_chapter_ids": [chapter["chapter_id"]],
        "crosses_chapter": False,
        "source_start": chapter["offset"],
        "source_end": chapter["offset"] + len(chapter["text"]),
    }


def _progress_book(book_idx: int, nb: int, ch_idx: int, nc: int, pairs: int,
                   decisions: int, elapsed: float):
    b_bar = "█" * int(10 * book_idx / max(nb, 1)) + "░" * max(0, 10 - int(10 * book_idx / max(nb, 1)))
    ch_bar = "█" * int(14 * ch_idx / max(nc, 1)) + "░" * max(0, 14 - int(14 * ch_idx / max(nc, 1)))
    mem = gpu_mem()
    line = f"  {b_bar} b{book_idx}/{nb}  {ch_bar} ch{ch_idx}/{nc}  pairs={pairs}  {_elapsed(elapsed)}  GPU:{mem}"
    print(line[:150], end="", flush=True)
    sys.stdout.flush()


def process_book(book_txt_path: Path, model, tokenizer, batch_size: int,
                 book_idx: int, nb_books: int) -> list[dict]:
    raw = book_txt_path.read_text(encoding="utf-8")
    chapters_raw = parse_book(raw)
    all_items: list[dict] = []
    item_idx = 0
    t0 = time.time()

    for chi, ch in enumerate(chapters_raw, 1):
        paras = split_paragraphs(ch["text"])
        if len(paras) <= 1:
            for c in _postprocess([ch["text"].strip()]):
                item_idx += 1
                if c: all_items.append(_make_item(ch, c, item_idx))
            continue

        pairs = [(paras[i], paras[i + 1]) for i in range(len(paras) - 1)]
        print(f"  ch{chi}/{len(chapters_raw)} [{ch['chapter_id'][:40]}]  {len(paras)}paras→{len(pairs)}pairs",
              flush=True)

        decisions = batch_judge_pairs(pairs, model, tokenizer, batch_size)
        chunks = chunk_paragraphs(paras, decisions)
        for c in chunks:
            item_idx += 1
            if c.strip():
                all_items.append(_make_item(ch, c.strip(), item_idx))

        _progress_book(book_idx, nb_books, chi, len(chapters_raw), len(pairs),
                       len(decisions), time.time() - t0)

    print(flush=True)
    print(f"  -> {len(all_items)} chunks, {_elapsed(time.time()-t0)}", flush=True)
    return all_items


# ==================== Main entry ====================

def main():
    parser = argparse.ArgumentParser(description="LBDC local batch version (local-llm)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Parallel paragraph pairs (default {BATCH_SIZE})")
    args = parser.parse_args()

    print(f"BATCH_SIZE = {args.batch_size}", flush=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(MODEL_PATH)
    txt_files = sorted(p for p in Path(INPUT_DIR).glob("*.txt") if p.is_file())

    total_c = 0
    t0 = time.time()
    for fi, fp in enumerate(txt_files, 1):
        torch.cuda.empty_cache()
        out_path = Path(OUTPUT_DIR) / f"{fp.stem}{OUTPUT_SUFFIX}"
        if out_path.exists():
            data = json.loads(out_path.read_text(encoding="utf-8"))
            n = len(data.get("items", []))
            total_c += n
            print(f"[{fi}/{len(txt_files)}] {fp.stem}  SKIP ({n} chunks)", flush=True)
            continue
        print(f"\n[{fi}/{len(txt_files)}] {fp.stem}", flush=True)
        items = process_book(fp, model, tokenizer, args.batch_size, fi, len(txt_files))
        data = {
            "split_method": SPLIT_METHOD, "model": MODEL_PATH.split("/")[-1],
            "target_words": TARGET_WORDS,
            "max_length": max((i["length"] for i in items), default=0),
            "min_length": min((i["length"] for i in items), default=0),
            "items": items,
        }
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        total_c += len(items)
        lens = [i["length"] for i in items]
        if lens:
            print(f"  -> {len(items)} chunks, avg={sum(lens)/len(lens):.0f}w [{min(lens)}-{max(lens)}]", flush=True)
        else:
            print(f"  -> {len(items)} chunks", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}\nDone: {len(txt_files)} books, {total_c} chunks, {_elapsed(elapsed)}\n{'='*60}")


if __name__ == "__main__":
    main()
