#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LumberChunker local version — local-llm + multi-chapter parallel batch processing.
Each chapter's LLM judgment requests packed into a batch, one forward pass processes multiple chapters.

Usage:
  python3 lumberchunker_local.py --batch_size 2
"""

from __future__ import annotations

import argparse
import json
import os
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

WORD_THRESHOLD     = 550
MERGE_SMALL_MIN    = 80
MAX_SEQ_LENGTH     = 2048
MIN_CHUNK_WORDS    = 200       # Merge threshold for undersized chunks
MAX_CHUNK_WORDS    = 2200      # Split threshold for oversized chunks
# ============================================================

ANCHOR_PATTERN = re.compile(r"^<<<CHAPTER_ID:(.*?)>>>\s*$", re.MULTILINE)
WORD_PATTERN   = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
ID_PATTERN     = re.compile(r"ID\s*(\d+)", re.I)

PROMPT_SYSTEM = (
    "You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.\n\n"
    "Task: Find the first paragraph (not the first one) where the content clearly changes "
    "compared to the previous paragraphs.\n\n"
    "Output: Return the ID of the paragraph with the content shift as in the exemplified format: "
    "'Answer: ID XXXX'.\n\n"
    "Additional Considerations: Avoid very long groups of paragraphs. "
    "Aim for a good balance between identifying content shifts and keeping groups manageable."
)
SPLIT_METHOD  = "lumberchunker_local"
OUTPUT_SUFFIX = "_LumberChunker_split.json"


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


def _make_item(chapter: dict, text: str, idx: int) -> dict:
    return {
        "region_id": "1", "is_region_summary": False,
        "info": f"LumberChunker---block {idx:>3}",
        "length": count_words(text), "text": text,
        "original_length": len(text), "word_count": count_words(text),
        "active_chapter_id": chapter["chapter_id"],
        "covered_chapter_ids": [chapter["chapter_id"]],
        "crosses_chapter": False,
        "source_start": chapter["offset"],
        "source_end": chapter["offset"] + len(chapter["text"]),
    }


def _postprocess(chunks: list[dict]) -> list[dict]:
    """Post-process chunk sizes: merge undersized, split oversized"""
    if not chunks:
        return chunks
    # Merge chunks < MIN_CHUNK_WORDS
    merged = []
    for c in chunks:
        wc = count_words(c["text"])
        if merged and wc < MIN_CHUNK_WORDS:
            merged[-1]["text"] += "\n" + c["text"]
            merged[-1]["paras"] += c["paras"]
        else:
            merged.append(dict(c))
    # Re-check first chunk
    if len(merged) >= 2 and count_words(merged[0]["text"]) < MIN_CHUNK_WORDS:
        merged[1]["text"] = merged[0]["text"] + "\n" + merged[1]["text"]
        merged[1]["paras"] += merged[0]["paras"]
        merged = merged[1:]
    # Split chunks > MAX_CHUNK_WORDS
    result = []
    for c in merged:
        wc = count_words(c["text"])
        if wc <= MAX_CHUNK_WORDS or c["paras"] <= 1:
            result.append(c)
        else:
            # Bisect at paragraph boundary
            paras_list = c["text"].split("\n")
            mid = len(paras_list) // 2
            a = "\n".join(paras_list[:mid]); b = "\n".join(paras_list[mid:])
            if a.strip(): result.append({"text": a.strip(), "paras": mid})
            if b.strip(): result.append({"text": b.strip(), "paras": len(paras_list) - mid})
    return result


# ==================== Core algorithm ====================

class ChapterState:
    def __init__(self, ch: dict):
        self.chapter = ch
        self.paragraphs = split_paragraphs(ch["text"])
        self.id_chunks = [f"ID {i}: {p}" for i, p in enumerate(self.paragraphs)]
        self.n = len(self.id_chunks)
        self.chunk_number = 0
        self.boundary_ids: list[int] = []
        self.calls = 0
        self._trivial = self.n <= 2       # Too few paragraphs, no LLM iteration needed
        self._finalized = False            # Whether chunk has been produced

    def done(self) -> bool:
        return self._finalized or self._trivial or self.chunk_number >= self.n - 3

    def build_document(self) -> str | None:
        if self.done():
            return None
        wc, i = 0, 0
        while wc < WORD_THRESHOLD and i + self.chunk_number < self.n:
            i += 1
            wc = count_words("\n".join(self.id_chunks[self.chunk_number: self.chunk_number + i]))
        if i <= 1:
            self.chunk_number += 1
            return None
        doc = "\n".join(self.id_chunks[self.chunk_number: self.chunk_number + i - 1])
        return PROMPT_SYSTEM + "\n\nDocument:\n" + doc

    def apply_result(self, response: str) -> None:
        self.calls += 1
        m = ID_PATTERN.search(response.strip())
        if not m:
            self.chunk_number += 1
            return
        new_id = int(m.group(1))
        if new_id <= self.chunk_number or new_id >= self.n:
            self.chunk_number += 1
        else:
            self.boundary_ids.append(new_id)
            self.chunk_number = new_id

    def finalize(self) -> list[dict]:
        if self._finalized:
            return []
        self._finalized = True
        if self._trivial:
            return _postprocess([{"text": self.chapter["text"].strip(), "paras": self.n}])
        ids = list(self.boundary_ids) + [self.n]
        clean = [re.sub(r"^ID\s+\d+:\s*", "", p, flags=re.I) for p in self.id_chunks]
        result = []
        prev = 0
        for bid in ids:
            text = "\n".join(clean[prev:bid]).strip()
            if text:
                result.append({"text": text, "paras": bid - prev})
            prev = bid
        return _postprocess(result)


def _progress(book_idx: int, nb_books: int, states: list[ChapterState],
              elapsed: float, batch_size: int):
    done_cnt = sum(1 for s in states if s._finalized)
    total_cnt = len(states)
    total_calls = sum(s.calls for s in states)
    active = [s for s in states if not s._finalized and not s.done()]
    n_active = len(active)
    mem_gb = torch.cuda.memory_allocated() / 1024**3

    b_bar = "█" * int(10 * book_idx / max(nb_books, 1)) + "░" * max(0, 10 - int(10 * book_idx / max(nb_books, 1)))
    ch_bar = "█" * int(14 * done_cnt / max(total_cnt, 1)) + "░" * max(0, 14 - int(14 * done_cnt / max(total_cnt, 1)))

    active_info = "│".join(
        f"c{c.calls:>3}@{c.chunk_number:>3}/{c.n:<3}"
        for c in active[:3]
    )
    if len(active) > 3:
        active_info += f"│+{len(active)-3}"

    line = (
        f"\r{b_bar} b{book_idx}/{nb_books} {ch_bar} ch{done_cnt}/{total_cnt} "
        f"act={n_active} calls={total_calls} bs={batch_size} {mem_gb:.1f}G "
        f"[{active_info}] {_elapsed(elapsed)}"
    )
    print(line[:170], end="", flush=True)
    sys.stdout.flush()


def batch_generate(prompts: list[str], model, tokenizer) -> list[str]:
    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages_list]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=MAX_SEQ_LENGTH).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=15, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    results = []
    for i in range(len(prompts)):
        out_ids = outputs[i][inputs["input_ids"].shape[1]:]
        results.append(tokenizer.decode(out_ids, skip_special_tokens=True).strip())
    torch.cuda.empty_cache()
    return results


def process_book(book_txt_path: Path, model, tokenizer, batch_size: int,
                 book_idx: int, nb_books: int) -> list[dict]:
    raw = book_txt_path.read_text(encoding="utf-8")
    chapters_raw = parse_book(raw)
    states = [ChapterState(ch) for ch in chapters_raw]

    all_items: list[dict] = []
    item_idx = 0
    t0 = time.time()

    while True:
        batch_prompts: list[str] = []
        batch_indices: list[int] = []

        for idx, st in enumerate(states):
            if st.done():
                continue
            doc = st.build_document()
            if doc is not None:
                batch_prompts.append(doc)
                batch_indices.append(idx)
            if len(batch_prompts) >= batch_size:
                break

        if not batch_prompts:
            still_active = any(not s.done() for s in states)
            if not still_active:
                break
            continue

        responses = batch_generate(batch_prompts, model, tokenizer)
        for i, idx in enumerate(batch_indices):
            states[idx].apply_result(responses[i])

        # Produce chunk immediately on chapter completion
        for st in states:
            if st.done() and not st._finalized:
                for seg in st.finalize():
                    item_idx += 1
                    t = seg["text"].strip()
                    if t:
                        all_items.append(_make_item(st.chapter, t, item_idx))

        _progress(book_idx, nb_books, states, time.time() - t0, batch_size)

    # Fallback: process chapters not yet finalized
    for st in states:
        if not st._finalized:
            for seg in st.finalize():
                item_idx += 1
                t = seg["text"].strip()
                if t:
                    all_items.append(_make_item(st.chapter, t, item_idx))

    print(flush=True)
    print(f"  -> {len(all_items)} chunks, total_calls={sum(s.calls for s in states)}, "
          f"{_elapsed(time.time()-t0)}", flush=True)
    return all_items


def main():
    parser = argparse.ArgumentParser(description="LumberChunker local multi-chapter parallel version")
    parser.add_argument("--batch_size", type=int, default=5, help="Parallel chapters (1-8)")
    args = parser.parse_args()

    print(f"BATCH_SIZE = {args.batch_size}", flush=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(MODEL_PATH)
    txt_files = sorted(p for p in Path(INPUT_DIR).glob("*.txt") if p.is_file())

    total_c = 0
    t0 = time.time()
    for fi, fp in enumerate(txt_files, 1):
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
            "word_threshold": WORD_THRESHOLD,
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
