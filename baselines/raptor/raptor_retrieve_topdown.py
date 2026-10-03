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
RAPTOR Top-Down Retrieval
==========================
Start from root nodes, keep the top BRANCH_K relevant nodes at each level, expand children,
continue down to leaf level, then re-rank candidates and output top-k results.

Input:  - Query embeddings: embeddings/by_book/{Book}.queries.npy + .meta.json
       - RAPTOR node embeddings: embeddings/by_book/{Book}.raptor.chunks.npy + .meta.json
       - RAPTOR tree: example_data/Far From the Madding Crowd/{Book}_RAPTOR_nodes.json
output:  runs/in_book/{Book}.raptor.top10.json
"""

# ================================================================================================
# ==================================== Parameters ====================================================
# ================================================================================================

SCHEME: str = "raptor"

BRANCH_K: int = 3
FINAL_TOP_K: int = 10
INCLUDE_SUMMARY_NODES: bool = True
INCLUDE_LEAF_NODES: bool = True
MIN_FINAL_LEAF_NODES: int = 2
DEDUPLICATE_BY_NODE_ID: bool = True
FORCE_RETRIEVE: bool = False

RAPTOR_NODES_DIR: str = str(EXAMPLE_BOOK_DIR)
RAPTOR_NODES_SUFFIX: str = "_RAPTOR_nodes.json"

QUERY_EMBEDDINGS_DIR: str = str(REPO_ROOT / "src" / "retrieval" / "embeddings" / "by_book")
CHUNK_EMBEDDINGS_DIR: str = str(REPO_ROOT / "src" / "retrieval" / "embeddings" / "by_book")

RUNS_DIR: str = str(REPO_ROOT / "src" / "retrieval" / "runs" / "in_book")

BOOK_FILTER: str = None

# ================================================================================================
# ========================================== Implementation ============================================
# ================================================================================================

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ---- atomic write ----

def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp_path), str(path))


# ---- IO helpers ----

def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_progress(current: int, total: int, prefix: str = "progress") -> None:
    total = max(total, 1)
    bar_len = 30
    filled = int(bar_len * current // total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix}: [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


# ---- cosine similarity (embeddings already L2-normalized) ----

def cosine_similarity_vec(q: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.dot(D, q)


# ---- RAPTOR tree loading ----

def load_raptor_nodes(nodes_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[int, List[str]]]:
    data = read_json(nodes_path)
    all_nodes = data.get("nodes", [])
    levels_raw = data.get("levels", {})
    node_map: Dict[str, Dict[str, Any]] = {}
    for node in all_nodes:
        node_map[node["node_id"]] = node
    levels: Dict[int, List[str]] = {}
    for k, v in levels_raw.items():
        levels[int(k)] = v
    return all_nodes, node_map, levels


def get_node_children(node_map: Dict[str, Dict[str, Any]], node_id: str) -> List[str]:
    node = node_map.get(node_id)
    if not node:
        return []
    return node.get("children", []) or node.get("child_node_ids", []) or []


# ---- Top-Down Retrieval ----

def topdown_retrieve(
    query_vec: np.ndarray,
    node_map: Dict[str, Dict[str, Any]],
    levels: Dict[int, List[str]],
    node_id_to_emb_idx: Dict[str, int],
    embeddings: np.ndarray,
    branch_k: int,
    final_top_k: int,
    include_summary: bool,
    include_leaf: bool,
) -> List[Tuple[str, float, str]]:
    """Return [(chunk_id, score, node_type), ...] sorted by descending score"""
    if not levels:
        return []

    max_level = max(levels.keys())
    root_ids = levels.get(max_level, [])
    if not root_ids:
        return []

    current_candidates = list(root_ids)
    collected: List[Tuple[str, float, str]] = []
    current_level = max_level

    while current_level >= 0 and current_candidates:
        scored: List[Tuple[str, float]] = []
        for node_id in current_candidates:
            emb_idx = node_id_to_emb_idx.get(node_id)
            if emb_idx is not None and 0 <= emb_idx < len(embeddings):
                sim = float(cosine_similarity_vec(query_vec, embeddings[emb_idx]))
                scored.append((node_id, sim))
            else:
                scored.append((node_id, -1.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_k_nodes = scored[:branch_k]

        for node_id, score in top_k_nodes:
            node = node_map.get(node_id, {})
            node_type = node.get("node_type", "unknown")
            collected.append((node_id, score, node_type))

        next_candidates: List[str] = []
        for node_id, _ in top_k_nodes:
            next_candidates.extend(get_node_children(node_map, node_id))

        current_candidates = next_candidates
        current_level -= 1

    # Filter by type and deduplicate
    seen: Set[str] = set()
    results: List[Tuple[str, float, str]] = []
    for node_id, score, node_type in collected:
        if node_id in seen:
            continue
        node = node_map.get(node_id, {})
        actual_type = node.get("node_type", node_type)
        book_title = node.get("book_title", "")
        chunk_id = f"{book_title}::raptor_node::{node_id}"
        if actual_type == "leaf" and include_leaf:
            results.append((chunk_id, score, actual_type))
            seen.add(node_id)
        elif actual_type == "summary" and include_summary:
            results.append((chunk_id, score, actual_type))
            seen.add(node_id)

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:final_top_k]


def fallback_flat_retrieve(
    query_vec: np.ndarray,
    all_nodes: List[Dict[str, Any]],
    node_id_to_emb_idx: Dict[str, int],
    embeddings: np.ndarray,
    exclude_ids: Set[str],
    top_k: int,
) -> List[Tuple[str, float, str]]:
    """Fill missing results from the flat node list by cosine similarity"""
    # Build reverse mapping from embedding row to node ID
    emb_idx_to_node_id: Dict[int, str] = {}
    for nid, eidx in node_id_to_emb_idx.items():
        emb_idx_to_node_id[eidx] = nid

    node_id_to_node: Dict[str, Dict[str, Any]] = {}
    for node in all_nodes:
        node_id_to_node[node["node_id"]] = node

    scores = cosine_similarity_vec(query_vec, embeddings)  # shape (N,)
    n = len(scores)
    k = min(top_k + len(exclude_ids), n)
    if k <= 0:
        return []
    top_idx = np.argpartition(-scores, kth=k - 1)[:k]

    results: List[Tuple[str, float, str]] = []
    for idx in top_idx:
        idx_int = int(idx)
        target_node_id = emb_idx_to_node_id.get(idx_int)
        if target_node_id is None or target_node_id in exclude_ids:
            continue
        node = node_id_to_node.get(target_node_id)
        if not node:
            continue
        book_title = node.get("book_title", "")
        chunk_id = f"{book_title}::raptor_node::{target_node_id}"
        node_type = node.get("node_type", "unknown")
        results.append((chunk_id, float(scores[idx]), node_type))
        if len(results) >= top_k:
            break

    return results


# ---- Validation ----

def validate_run(output_path: Path, expected_query_count: int,
                 raptor_chunk_ids: Set[str], final_top_k: int) -> Tuple[bool, str]:
    """Validate whether an existing run file is complete and usable"""
    if not output_path.exists():
        return False, "file does not exist"
    try:
        data = read_json(output_path)
    except Exception as e:
        return False, f"failed to read JSON: {e}"
    if not isinstance(data, list):
        return False, "top-level JSON is not a list"
    if len(data) != expected_query_count:
        return False, f"query count {len(data)} != expected {expected_query_count}"

    bad_entries = 0
    bad_chunks = 0
    short_topk = 0
    for i, entry in enumerate(data):
        # Check required fields
        for key in ("query_id", "book_title", "chapter_id", "top_k"):
            if key not in entry:
                return False, f"entry[{i}] missing {key}"
        top_k = entry.get("top_k", [])
        if len(top_k) < 1:
            bad_entries += 1
            continue
        if len(top_k) < final_top_k:
            short_topk += 1
        # Check each top_k item
        for item in top_k:
            for key in ("rank", "chunk_id", "score"):
                if key not in item:
                    return False, f"entry[{i}] top_k item missing {key}"
            if item["chunk_id"] not in raptor_chunk_ids:
                bad_chunks += 1

    if bad_entries > 0:
        return False, f"{bad_entries} entries have empty top_k"
    if bad_chunks > 0:
        return False, f"{bad_chunks} chunk IDs are not in RAPTOR chunks"
    if short_topk > 0:
        return False, f"{short_topk} entries have top_k < {final_top_k}"

    return True, "OK"


# ---- Main ----

def list_book_titles() -> List[str]:
    nodes_dir = Path(RAPTOR_NODES_DIR)
    return sorted(p.name.replace(RAPTOR_NODES_SUFFIX, "")
                  for p in nodes_dir.glob(f"*{RAPTOR_NODES_SUFFIX}")
                  if p.is_file())


def retrieve_one_book(book_title: str) -> Optional[Dict[str, Any]]:
    query_meta_path = Path(QUERY_EMBEDDINGS_DIR) / f"{book_title}.queries.meta.json"
    query_npy_path = Path(QUERY_EMBEDDINGS_DIR) / f"{book_title}.queries.npy"
    raptor_nodes_path = Path(RAPTOR_NODES_DIR) / f"{book_title}{RAPTOR_NODES_SUFFIX}"
    chunk_emb_path = Path(CHUNK_EMBEDDINGS_DIR) / f"{book_title}.raptor.chunks.npy"
    chunk_meta_path = Path(CHUNK_EMBEDDINGS_DIR) / f"{book_title}.raptor.chunks.meta.json"
    output_path = Path(RUNS_DIR) / f"{book_title}.{SCHEME}.top{FINAL_TOP_K}.json"

    # Check files
    missing = []
    for p, desc in [
        (query_meta_path, "query meta"),
        (query_npy_path, "query embeddings"),
        (raptor_nodes_path, "RAPTOR nodes"),
        (chunk_emb_path, "chunk embeddings"),
        (chunk_meta_path, "chunk meta"),
    ]:
        if not p.exists():
            missing.append(f"  {desc}: {p}")
    if missing:
        print(f"\n  Missing files for {book_title}:")
        for m in missing:
            print(m)
        return None

    query_meta = read_json(query_meta_path)
    query_emb = np.load(str(query_npy_path))

    if len(query_meta) != len(query_emb):
        print(f"  {book_title}: query meta ({len(query_meta)}) != embeddings ({len(query_emb)})")
        return None

    # Validate existing run
    chunk_meta = read_json(chunk_meta_path)
    raptor_chunk_ids = {e["chunk_id"] for e in chunk_meta}
    valid, reason = validate_run(output_path, len(query_meta), raptor_chunk_ids, FINAL_TOP_K)
    if valid and not FORCE_RETRIEVE:
        print(f"  [skip] {book_title}: existing run is valid ({len(query_meta)} queries)")
        return {"book_title": book_title, "queries": len(query_meta), "skipped": True}
    elif output_path.exists():
        print(f"  [recompute] {book_title}: {reason}")

    # Load data
    all_nodes, node_map, levels = load_raptor_nodes(raptor_nodes_path)
    chunk_embeddings = np.load(str(chunk_emb_path))

    # node_id -> embedding row index
    node_id_to_emb_idx: Dict[str, int] = {}
    for entry in chunk_meta:
        parts = entry["chunk_id"].split("::raptor_node::")
        if len(parts) == 2:
            node_id_to_emb_idx[parts[1]] = entry["row"]

    # Retrieve
    fallback_count = 0
    insufficient_count = 0
    summary_node_counts = []
    leaf_node_counts = []

    results: List[Dict[str, Any]] = []
    for q_idx, q_meta in enumerate(query_meta):
        query_id = q_meta["query_id"]
        chapter_id = q_meta.get("chapter_id", "")
        q_vec = query_emb[q_idx]

        top_results = topdown_retrieve(
            query_vec=q_vec,
            node_map=node_map,
            levels=levels,
            node_id_to_emb_idx=node_id_to_emb_idx,
            embeddings=chunk_embeddings,
            branch_k=BRANCH_K,
            final_top_k=FINAL_TOP_K,
            include_summary=INCLUDE_SUMMARY_NODES,
            include_leaf=INCLUDE_LEAF_NODES,
        )

        # Fallback
        if len(top_results) < FINAL_TOP_K:
            fallback_count += 1
            exclude_ids: Set[str] = set()
            for chunk_id, _, _ in top_results:
                parts = chunk_id.split("::raptor_node::")
                if len(parts) == 2:
                    exclude_ids.add(parts[1])
            fb = fallback_flat_retrieve(
                query_vec=q_vec,
                all_nodes=all_nodes,
                node_id_to_emb_idx=node_id_to_emb_idx,
                embeddings=chunk_embeddings,
                exclude_ids=exclude_ids,
                top_k=FINAL_TOP_K - len(top_results),
            )
            top_results.extend(fb)

        # Stats
        s_count = sum(1 for _, _, t in top_results if t == "summary")
        l_count = sum(1 for _, _, t in top_results if t == "leaf")
        summary_node_counts.append(s_count)
        leaf_node_counts.append(l_count)
        if len(top_results) < FINAL_TOP_K:
            insufficient_count += 1

        top_k_list = [
            {"rank": r + 1, "chunk_id": cid, "score": sc}
            for r, (cid, sc, _) in enumerate(top_results[:FINAL_TOP_K])
        ]
        results.append({
            "query_id": query_id,
            "book_title": book_title,
            "chapter_id": chapter_id,
            "query_variant": "default",
            "top_k": top_k_list,
        })

    atomic_write_json(output_path, results)

    avg_s = sum(summary_node_counts) / max(len(summary_node_counts), 1)
    avg_l = sum(leaf_node_counts) / max(len(leaf_node_counts), 1)

    print(f"  {book_title}: {len(results)} queries, "
          f"fallback={fallback_count}, insufficient={insufficient_count}, "
          f"avg_summary={avg_s:.1f}, avg_leaf={avg_l:.1f}")
    print(f"    -> {output_path.name}")

    return {
        "book_title": book_title,
        "queries": len(results),
        "fallback_count": fallback_count,
        "insufficient_count": insufficient_count,
        "avg_summary_nodes": avg_s,
        "avg_leaf_nodes": avg_l,
        "skipped": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAPTOR top-down retrieval")
    parser.add_argument("--book", type=str, default=None, help="process only the specified book title")
    args = parser.parse_args()

    if args.book:
        book_titles = [args.book]
    elif BOOK_FILTER:
        book_titles = [BOOK_FILTER]
    else:
        book_titles = list_book_titles()

    if not book_titles:
        print(f"not found RAPTOR nodes: {RAPTOR_NODES_DIR}/*{RAPTOR_NODES_SUFFIX}")
        sys.exit(1)

    Path(RUNS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"RAPTOR top-down retrieval: {len(book_titles)} books, "
          f"BRANCH_K={BRANCH_K}, FINAL_TOP_K={FINAL_TOP_K}")

    skipped = 0
    computed = 0
    total_queries = 0
    for idx, bt in enumerate(book_titles, start=1):
        print(f"\n[{idx}/{len(book_titles)}] {bt}")
        result = retrieve_one_book(bt)
        if result:
            total_queries += result["queries"]
            if result.get("skipped"):
                skipped += 1
            else:
                computed += 1

    print(f"\nSummary: {len(book_titles)} books, skip={skipped}, recompute={computed}, "
          f"total queries={total_queries}")


if __name__ == "__main__":
    main()
