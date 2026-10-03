#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
WORK_BOOK_DIR = Path(os.environ.get(
    "HISUMCHUNK_BOOK_DIR",
    REPO_ROOT / "example_data" / "Far From the Madding Crowd",
))
EXAMPLE_BOOK_DIR = WORK_BOOK_DIR
"""
RAPTOR Tree Construction
=========================
Build a recursive summary tree from fixed-size base chunks using embedding, KMeans clustering, and LLM summarization.

Input:  example_data/Far From the Madding Crowd/*_Fixed1024W_NoOverlap_split.json
output:  example_data/Far From the Madding Crowd/{BookTitle}_RAPTOR_tree.json
       example_data/Far From the Madding Crowd/{BookTitle}_RAPTOR_nodes.json
"""

# ================================================================================================
# ==================================== Parameters ====================================================
# ================================================================================================

LOCAL_LLM_PATH: str = os.getenv("CHUNKING_MODEL_PATH", "<MODEL_DIR>/Qwen3-4B-Instruct-2507")
SGLANG_ENDPOINT: str = "http://127.0.0.1:30000"
EMBEDDING_MODEL_PATH: str = os.getenv("EMBEDDING_MODEL_PATH", "<MODEL_DIR>/Qwen3-Embedding-0.6B")

TEMPERATURE: float = 0.1
TOP_P: float = 0.5
TOP_K = 15
MAX_GENERATE_TOKENS: int = 1024
MAX_INPUT_LENGTH: int = 50000
SYSTEM_PROMPT: str = ""

TARGET_CLUSTER_SIZE: int = 6
MIN_CLUSTER_SIZE: int = 3
MAX_CLUSTER_SIZE: int = 10
MAX_LEVELS: int = 5
STOP_NODE_COUNT: int = 5

SUMMARY_MAX_INPUT_WORDS: int = 6000
SUMMARY_OUTPUT_WORDS: int = 200
EMBED_BATCH_SIZE: int = 16
SUMMARY_BATCH_SIZE: int = 1

INPUT_DIR: str = str(EXAMPLE_BOOK_DIR)
OUTPUT_DIR: str = str(EXAMPLE_BOOK_DIR)
CHECKPOINT_DIR: str = str(EXAMPLE_BOOK_DIR / "raptor_checkpoints")
FIXED_SUFFIX: str = "_Fixed1024W_NoOverlap_split.json"

FORCE_REBUILD: bool = False
BOOK_FILTER: str = None

SEPARATOR_LENGTH: int = 60
SHOW_PROGRESS: bool = True

SUMMARY_SYSTEM_PROMPT: str = (
    "You are building a hierarchical retrieval tree for a literary text. "
    "Given child chunks or child summaries from the same book, write a concise "
    "abstractive summary preserving key events, character names, causal relations, "
    "locations, and evidence-relevant details. Do not invent facts not supported "
    f"by the input. The summary should be around {SUMMARY_OUTPUT_WORDS} English words."
)
SUMMARY_USER_TEMPLATE: str = (
    "The following are child chunks or child summaries from the same book:\n\n"
    "{child_texts}\n\n"
    f"Write a concise abstractive summary (~{SUMMARY_OUTPUT_WORDS} words) that captures "
    "the key events, character names, locations, causal relations, and evidence-relevant "
    "details. Do not invent facts not in the input."
)

# ================================================================================================
# ========================================== Implementation ============================================
# ================================================================================================

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sglang import RuntimeEndpoint, assistant, function, gen, set_default_backend, system, user

_backend_initialized = False
_model_cache: Dict[Tuple[str, str], SentenceTransformer] = {}


# ---- atomic write ----

def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp_path), str(path))


# ---- sglang ----

def _init_sglang_backend():
    global _backend_initialized
    if _backend_initialized:
        return
    if SHOW_PROGRESS:
        print(f"Connecting to SGLang: {SGLANG_ENDPOINT}")
    try:
        set_default_backend(RuntimeEndpoint(SGLANG_ENDPOINT))
    except Exception as exc:
        print(f"SGLang connection failed: {exc}")
        print(f"Please start: python -m sglang.launch_server --model-path {LOCAL_LLM_PATH} --port 30000")
        raise
    _backend_initialized = True


@function
def _sglang_single_chat(s, prompt: str, max_tokens: int):
    truncated = prompt[:MAX_INPUT_LENGTH]
    s += system(SYSTEM_PROMPT)
    s += user(truncated)
    gen_kwargs = {"max_tokens": max_tokens}
    if TEMPERATURE is not None:
        gen_kwargs["temperature"] = TEMPERATURE
    if TOP_P is not None:
        gen_kwargs["top_p"] = TOP_P
    if TOP_K is not None:
        gen_kwargs["top_k"] = TOP_K
    s += assistant(gen("response", **gen_kwargs))


def call_model(prompt: str, max_new_tokens: int) -> str:
    _init_sglang_backend()
    try:
        res = _sglang_single_chat(prompt=prompt, max_tokens=max_new_tokens)
        if isinstance(res, dict):
            raw = res.get("response", "")
        elif hasattr(res, "response"):
            raw = res.response
        else:
            raw = str(res)
        return raw.strip() if isinstance(raw, str) else str(raw).strip()
    except Exception as exc:
        print(f"[ERROR] model call failed: {exc}")
        return ""


# ---- embedding ----

def load_embedding_model(model_path: str, device: str = "cuda") -> SentenceTransformer:
    key = (model_path, device)
    if key not in _model_cache:
        actual = device
        if device == "cuda" and not torch.cuda.is_available():
            actual = "cpu"
        _model_cache[key] = SentenceTransformer(model_path, device=actual)
    return _model_cache[key]


def encode_texts(texts: List[str], model_path: str, batch_size: int = 16) -> np.ndarray:
    truncated = [t[:4096] for t in texts]
    model = load_embedding_model(model_path)
    vectors = model.encode(truncated, batch_size=batch_size, show_progress_bar=False,
                           normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vectors, dtype=np.float32)


# ---- utils ----

WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def truncate_to_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def cluster_summary_key(level: int, child_ids: List[str]) -> str:
    """Generate a stable cluster cache key from level and sorted child IDs"""
    h = hashlib.md5(str((level, sorted(child_ids))).encode()).hexdigest()[:12]
    return f"L{level}_{h}"


# ---- load fixed splits ----

def load_fixed_split(split_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    book_title = split_path.name.replace(FIXED_SUFFIX, "")
    if items:
        first_ch = str(items[0].get("active_chapter_id", "") or "").strip()
        if first_ch:
            parts = first_ch.rsplit(".chapter", 1)
            if len(parts) == 2:
                book_title = parts[0].strip()
    return book_title, items


# ---- node constructors ----

def make_leaf_node(node_id: str, book_title: str, item: Dict[str, Any]) -> Dict[str, Any]:
    text = str(item.get("text", "") or "").strip()
    return {
        "node_id": node_id, "book_title": book_title, "level": 0,
        "node_type": "leaf", "text": text, "length": count_words(text),
        "children": [], "parent": None, "source_chunk_ids": [],
        "active_chapter_id": str(item.get("active_chapter_id", "") or "").strip(),
        "covered_chapter_ids": list(item.get("covered_chapter_ids", []) or []),
        "crosses_chapter": bool(item.get("crosses_chapter", False)),
        "source_start": item.get("source_start"), "source_end": item.get("source_end"),
    }


def make_summary_node(node_id: str, book_title: str, level: int,
                      summary_text: str, children_ids: List[str],
                      source_chunk_ids: List[str],
                      covered_chapter_ids: List[str]) -> Dict[str, Any]:
    return {
        "node_id": node_id, "book_title": book_title, "level": level,
        "node_type": "summary", "text": summary_text,
        "length": count_words(summary_text), "children": children_ids,
        "parent": None, "source_chunk_ids": source_chunk_ids,
        "covered_chapter_ids": covered_chapter_ids,
        "child_node_ids": children_ids, "summary_prompt_version": "v1",
        "crosses_chapter": len(set(covered_chapter_ids)) > 1,
        "active_chapter_id": covered_chapter_ids[0] if covered_chapter_ids else "",
        "source_start": None, "source_end": None,
    }


# ---- clustering ----

def cluster_nodes_kmeans(embeddings: np.ndarray, target_size: int,
                         min_size: int, max_size: int) -> List[List[int]]:
    from sklearn.cluster import KMeans
    n = len(embeddings)
    if n <= max_size:
        return [list(range(n))]
    k = max(2, n // target_size)
    k = min(k, n // min_size)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)
    result = []
    small_buf: List[int] = []
    for indices in clusters.values():
        if len(indices) < min_size:
            small_buf.extend(indices)
        else:
            result.append(indices)
    if small_buf:
        if result:
            smallest = min(range(len(result)), key=lambda i: len(result[i]))
            result[smallest].extend(small_buf)
        else:
            result.append(small_buf)
    final = []
    for indices in result:
        if len(indices) > max_size:
            for i in range(0, len(indices), max_size):
                final.append(indices[i:i + max_size])
        else:
            final.append(indices)
    return final


# ---- checkpoint (summary cache checkpoint) ----

def checkpoint_path(book_title: str) -> Path:
    return Path(CHECKPOINT_DIR) / f"{book_title}_RAPTOR_checkpoint.json"


def load_checkpoint(book_title: str) -> Optional[Dict[str, Any]]:
    p = checkpoint_path(book_title)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_checkpoint(book_title: str, status: str, current_level: int,
                    levels: Dict[int, List[str]], all_nodes_count: int,
                    summary_cache: Dict[str, Dict[str, Any]], params: Dict[str, Any],
                    checkpoint_count: int = 0) -> None:
    """Save a summary-cache checkpoint. This checkpoint stores only summary_cache and level structure, not the full node list. Recovery reuses cached summaries and rebuilds node structure from level 0."""
    ckpt = {
        "book_title": book_title,
        "status": status,
        "current_level": current_level,
        "levels": {str(k): v for k, v in levels.items()},
        "all_nodes_count": all_nodes_count,
        "summary_cache": summary_cache,
        "updated_at": datetime.now().isoformat(),
        "checkpoint_count": checkpoint_count,
        "params": params,
    }
    atomic_write_json(checkpoint_path(book_title), ckpt)


def load_summary_cache(book_title: str) -> Dict[str, Dict[str, Any]]:
    ckpt = load_checkpoint(book_title)
    if ckpt and "summary_cache" in ckpt:
        return ckpt["summary_cache"]
    return {}


# ---- summary generation ----

def generate_summary_for_cluster(cluster_nodes: List[Dict[str, Any]],
                                 max_input_words: int) -> str:
    child_texts = []
    for node in cluster_nodes:
        prefix = f"[Level {node['level']} {node['node_type']}]" if node['node_type'] == "summary" else "[Leaf]"
        child_texts.append(f"{prefix}\n{node['text']}")
    concatenated = "\n\n---\n\n".join(child_texts)
    concatenated = truncate_to_words(concatenated, max_input_words)
    user_prompt = SUMMARY_USER_TEMPLATE.format(child_texts=concatenated)
    full_prompt = f"{SUMMARY_SYSTEM_PROMPT}\n\n{user_prompt}"
    return call_model(full_prompt, MAX_GENERATE_TOKENS)


# ---- build tree ----

def build_raptor_tree(book_title: str, leaf_items: List[Dict[str, Any]],
                      summary_cache: Dict[str, Dict[str, Any]],
                      params: Dict[str, Any]
                      ) -> Tuple[List[Dict[str, Any]], Dict[int, List[str]],
                                 Dict[str, Dict[str, Any]], Dict[str, Any]]:
    all_nodes: List[Dict[str, Any]] = []
    levels: Dict[int, List[str]] = {}
    stats = {"cache_hit_success": 0, "cache_hit_fallback": 0,
             "newly_generated_success": 0, "newly_generated_fallback": 0,
             "level_times": {}, "embed_times": {}, "checkpoint_saves": 0}
    checkpoint_count = 0

    # Level 0: leaves
    current_level_nodes: List[Dict[str, Any]] = []
    for idx, item in enumerate(leaf_items):
        node_id = f"raptor_L0_{idx:04d}"
        node = make_leaf_node(node_id, book_title, item)
        current_level_nodes.append(node)
        all_nodes.append(node)
    levels[0] = [n["node_id"] for n in current_level_nodes]
    print(f"  Level 0: {len(current_level_nodes)} leaf nodes")

    # Save initial checkpoint (level 0 complete)
    save_checkpoint(book_title, "building", 0, levels, len(all_nodes),
                    summary_cache, params, checkpoint_count)
    stats["checkpoint_saves"] += 1

    current_level = 0
    while len(current_level_nodes) > STOP_NODE_COUNT and current_level < MAX_LEVELS - 1:
        current_level += 1
        level_start = time.time()

        # Embedding
        t0 = time.time()
        texts = [n["text"] for n in current_level_nodes]
        embeddings = encode_texts(texts, EMBEDDING_MODEL_PATH, batch_size=EMBED_BATCH_SIZE)
        embed_time = time.time() - t0
        stats["embed_times"][str(current_level)] = embed_time
        print(f"  Level {current_level}: embedding {len(texts)} nodes ({embed_time:.1f}s)")

        # Clustering
        clusters = cluster_nodes_kmeans(embeddings, TARGET_CLUSTER_SIZE,
                                        MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE)
        sizes = [len(c) for c in clusters]
        print(f"  Level {current_level}: {len(clusters)} clusters, sizes={sizes}")

        # Generate summaries (with cache, save checkpoint after each)
        summary_gen_start = time.time()
        parent_nodes: List[Dict[str, Any]] = []
        for cluster_idx, cluster_indices in enumerate(clusters):
            child_ids = [current_level_nodes[i]["node_id"] for i in cluster_indices]
            cache_key = cluster_summary_key(current_level, child_ids)

            # Check cache: any cached text is reusable (success or fallback)
            if cache_key in summary_cache and summary_cache[cache_key].get("text"):
                summary_text = summary_cache[cache_key]["text"]
                if summary_cache[cache_key].get("success"):
                    stats["cache_hit_success"] += 1
                else:
                    stats["cache_hit_fallback"] += 1
            else:
                cluster_nodes = [current_level_nodes[i] for i in cluster_indices]
                summary_text = generate_summary_for_cluster(cluster_nodes, SUMMARY_MAX_INPUT_WORDS)

                if not summary_text:
                    summary_text = " ".join(n["text"][:200] for n in cluster_nodes)
                    summary_text = truncate_to_words(summary_text, SUMMARY_OUTPUT_WORDS)
                    stats["newly_generated_fallback"] += 1
                    summary_cache[cache_key] = {"text": summary_text, "success": False, "fallback": True}
                else:
                    stats["newly_generated_success"] += 1
                    summary_cache[cache_key] = {"text": summary_text, "success": True, "fallback": False}

                # Save checkpoint after each cluster summary generation
                checkpoint_count += 1
                save_checkpoint(book_title, "building", current_level, levels,
                                len(all_nodes), summary_cache, params, checkpoint_count)
                stats["checkpoint_saves"] += 1

            node_id = f"raptor_L{current_level}_{cluster_idx:04d}"
            source_cids: List[str] = []
            covered_cids: List[str] = []
            for i in cluster_indices:
                source_cids.extend(current_level_nodes[i].get("source_chunk_ids", []))
                covered_cids.extend(current_level_nodes[i].get("covered_chapter_ids", []))
            source_cids = list(dict.fromkeys(source_cids))
            covered_cids = list(dict.fromkeys(covered_cids))

            parent = make_summary_node(node_id, book_title, current_level,
                                       summary_text, child_ids, source_cids, covered_cids)
            parent_nodes.append(parent)
            all_nodes.append(parent)

            for i in cluster_indices:
                current_level_nodes[i]["parent"] = node_id

        summary_gen_time = time.time() - summary_gen_start
        stats["level_times"][str(current_level)] = time.time() - level_start
        levels[current_level] = [n["node_id"] for n in parent_nodes]
        current_level_nodes = parent_nodes

        # Save checkpoint after each level
        save_checkpoint(book_title, "building", current_level, levels,
                        len(all_nodes), summary_cache, params, checkpoint_count)
        stats["checkpoint_saves"] += 1

        total_cache = stats["cache_hit_success"] + stats["cache_hit_fallback"]
        total_new = stats["newly_generated_success"] + stats["newly_generated_fallback"]
        print(f"  Level {current_level}: {len(parent_nodes)} summaries "
              f"(cache_hit={total_cache}, new={total_new}, "
              f"time={summary_gen_time:.0f}s)")

    # Mark roots
    for node in current_level_nodes:
        node["is_root"] = True

    return all_nodes, levels, summary_cache, stats


# ---- save outputs ----

def save_outputs(book_title: str, all_nodes: List[Dict[str, Any]],
                 levels: Dict[int, List[str]]) -> None:
    tree_path = Path(OUTPUT_DIR) / f"{book_title}_RAPTOR_tree.json"
    nodes_path = Path(OUTPUT_DIR) / f"{book_title}_RAPTOR_nodes.json"
    max_level = max((int(k) for k in levels.keys()), default=0)

    # Build tree structure
    node_map = {n["node_id"]: n for n in all_nodes}

    def build_tree_node(nid: str) -> Dict[str, Any]:
        n = node_map[nid]
        t = {"node_id": n["node_id"], "node_type": n["node_type"],
             "level": n["level"], "text": n["text"], "length": n["length"]}
        if n.get("children"):
            t["children"] = [build_tree_node(cid) for cid in n["children"]]
        return t

    root_ids = [n["node_id"] for n in all_nodes if n.get("is_root")]
    tree_children = [build_tree_node(rid) for rid in root_ids]

    tree_data = {
        "book_title": book_title, "method": "raptor",
        "leaf_source": "fixed_1024_no_overlap",
        "embedding_model": EMBEDDING_MODEL_PATH, "summary_model": LOCAL_LLM_PATH,
        "max_level": max_level,
        "levels": {str(k): v for k, v in levels.items()},
        "tree": tree_children, "nodes": all_nodes,
    }
    atomic_write_json(tree_path, tree_data)

    nodes_data = {
        "book_title": book_title, "method": "raptor",
        "leaf_source": "fixed_1024_no_overlap",
        "embedding_model": EMBEDDING_MODEL_PATH, "summary_model": LOCAL_LLM_PATH,
        "max_level": max_level, "total_nodes": len(all_nodes),
        "leaf_count": sum(1 for n in all_nodes if n["node_type"] == "leaf"),
        "summary_count": sum(1 for n in all_nodes if n["node_type"] == "summary"),
        "levels": {str(k): v for k, v in levels.items()},
        "nodes": all_nodes,
    }
    atomic_write_json(nodes_path, nodes_data)


# ---- process one book ----

def check_tree_complete(nodes_path: Path) -> bool:
    if not nodes_path.exists():
        return False
    try:
        data = json.loads(nodes_path.read_text(encoding="utf-8"))
        nodes = data.get("nodes", [])
        return any(n.get("is_root") for n in nodes)
    except Exception:
        return False


def process_one_book(split_path: Path) -> Optional[Dict[str, Any]]:
    book_title, items = load_fixed_split(split_path)
    tree_path = Path(OUTPUT_DIR) / f"{book_title}_RAPTOR_tree.json"
    nodes_path = Path(OUTPUT_DIR) / f"{book_title}_RAPTOR_nodes.json"

    # Check if already complete
    if not FORCE_REBUILD and check_tree_complete(nodes_path):
        try:
            data = json.loads(nodes_path.read_text(encoding="utf-8"))
            n = data.get("total_nodes", 0)
            l = data.get("leaf_count", 0)
            s = data.get("summary_count", 0)
            ml = data.get("max_level", 0)
            print(f"  [skip] {book_title}: already complete ({n} nodes, {l} leaf, {s} summary, L{ml})")
            return {"book_title": book_title, "total_nodes": n, "leaf_count": l,
                    "summary_count": s, "max_level": ml, "skipped": True}
        except Exception:
            pass

    t0 = time.time()
    print(f"\n{'=' * SEPARATOR_LENGTH}")
    print(f"Building: {book_title} ({len(items)} leaf chunks)")
    print(f"{'=' * SEPARATOR_LENGTH}")

    # Load summary cache from checkpoint (summary cache checkpoint, not full tree state)
    summary_cache = load_summary_cache(book_title)
    if summary_cache:
        n_success = sum(1 for v in summary_cache.values() if v.get("success"))
        n_fallback = sum(1 for v in summary_cache.values() if not v.get("success") and v.get("text"))
        print(f"  Recovered from summary-cache checkpoint: {len(summary_cache)}  entries "
              f"(success={n_success}, fallback={n_fallback})")
    else:
        print(f"  No checkpoint found; building from scratch")

    params = {
        "target_cluster_size": TARGET_CLUSTER_SIZE, "max_levels": MAX_LEVELS,
        "stop_node_count": STOP_NODE_COUNT, "summary_max_input_words": SUMMARY_MAX_INPUT_WORDS,
    }

    all_nodes, levels, summary_cache, stats = build_raptor_tree(book_title, items, summary_cache, params)

    # Save final outputs
    save_outputs(book_title, all_nodes, levels)

    # Save completed checkpoint
    max_lv = max((int(k) for k in levels.keys()), default=0)
    save_checkpoint(book_title, "completed", max_lv, levels,
                    len(all_nodes), summary_cache, params, stats["checkpoint_saves"])

    elapsed = time.time() - t0
    leaf_c = sum(1 for n in all_nodes if n["node_type"] == "leaf")
    summary_c = sum(1 for n in all_nodes if n["node_type"] == "summary")

    print(f"\n  Completed: {book_title}")
    print(f"    nodes: {len(all_nodes)} (leaf={leaf_c}, summary={summary_c})")
    print(f"    max_level: {max_lv}")
    print(f"    cache_hit: success={stats['cache_hit_success']}, fallback={stats['cache_hit_fallback']}")
    print(f"    new_gen:   success={stats['newly_generated_success']}, fallback={stats['newly_generated_fallback']}")
    print(f"    checkpoint_saves: {stats['checkpoint_saves']}")
    print(f"    elapsed: {elapsed:.0f}s")
    print(f"    output: {tree_path}")

    return {
        "book_title": book_title, "total_nodes": len(all_nodes),
        "leaf_count": leaf_c, "summary_count": summary_c, "max_level": max_lv,
        "elapsed_seconds": elapsed, "skipped": False, **stats,
    }


# ---- main ----

def list_fixed_split_files() -> List[Path]:
    return sorted(Path(INPUT_DIR).glob(f"*{FIXED_SUFFIX}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="RAPTOR tree construction")
    parser.add_argument("--book", type=str, default=None, help="process only the specified book title")
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    if args.book:
        split_files = [Path(INPUT_DIR) / f"{args.book}{FIXED_SUFFIX}"]
        if not split_files[0].exists():
            print(f"file does not exist: {split_files[0]}")
            sys.exit(1)
    elif BOOK_FILTER:
        split_files = [Path(INPUT_DIR) / f"{BOOK_FILTER}{FIXED_SUFFIX}"]
    else:
        split_files = list_fixed_split_files()

    if not split_files:
        print(f"not found: {INPUT_DIR}/*{FIXED_SUFFIX}")
        sys.exit(1)

    print(f"RAPTOR tree Building: {len(split_files)} books")

    reports = []
    total_start = time.time()

    for idx, sp in enumerate(split_files, start=1):
        print(f"\n[{idx}/{len(split_files)}] {sp.name}")
        report = process_one_book(sp)
        if report:
            reports.append(report)

    elapsed = time.time() - total_start
    skipped = sum(1 for r in reports if r.get("skipped"))
    processed = len(reports) - skipped

    print(f"\n{'=' * SEPARATOR_LENGTH}")
    print("RAPTOR build summary")
    print(f"  books: {len(reports)}, processed: {processed}, skip: {skipped}")
    if reports:
        total_n = sum(r["total_nodes"] for r in reports)
        total_l = sum(r["leaf_count"] for r in reports)
        total_s = sum(r["summary_count"] for r in reports)
        print(f"  nodes: {total_n} (leaf={total_l}, summary={total_s})")
    print(f"  elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
