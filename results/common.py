#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common helpers for the reproduction scripts (tables 2-7, figures 3-5).

All numbers are recomputed from the released per-question score files in
``data/scores``. No API calls, no GPU: everything here is pure CPU/pandas-free
standard library code so that a reviewer can audit the paper at zero cost.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")
SCORES = os.path.join(DATA, "scores")

DIMS = ["grounding", "correctness", "hallucination_absence"]
DIM_LABEL = {"grounding": "Grounding", "correctness": "Correctness",
             "hallucination_absence": "Hallucination absence"}

# Paper-facing display order (best to worst by composite)
SCHEME_ORDER = ["ours", "raptor", "ctxretrieval", "lumberchunker",
                "lbdsemantic", "semanticembed", "fixed"]
SCHEME_LABEL = {
    "ours": "HiSumChunk (ours)",
    "raptor": "RAPTOR",
    "ctxretrieval": "Contextual Retrieval",
    "lumberchunker": "LumberChunker",
    "lbdsemantic": "LLMSemantic",
    "semanticembed": "SemanticEmbedding",
    "fixed": "Fixed-1024W",
}

SEED = 2026                 # the single seed used across the paper
BOOT_DEFAULT = 20000        # bootstrap resamples for significance


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_main_scores(path=None):
    """Per-question 7-scheme LLM-as-Judge scores, zero-score rows dropped.

    Returns ``{scheme: {query_id: {"book":..., "question_type":..., "difficulty":...,
                                   "grounding": int, ...}}}``.
    """
    path = path or os.path.join(SCORES, "scores_main_7schemes.csv")
    out = defaultdict(dict)
    for r in read_csv(path):
        vals = {}
        ok = True
        for d in DIMS:
            try:
                v = int(r[d])
            except (KeyError, ValueError):
                ok = False
                break
            if v == 0:
                ok = False
                break
            vals[d] = v
        if not ok:
            continue
        out[r["scheme"]][r["query_id"]] = {
            "book": r["book"], "question_type": r.get("question_type", ""),
            "difficulty": r.get("difficulty", ""), **vals,
        }
    return out


def composite(vals):
    """100-point composite = equal-weight mean of (x-1)/4*100 over the three dims."""
    return sum(vals[d] - 1 for d in DIMS) / (4 * len(DIMS)) * 100


def composite_by_question(scores):
    """``{scheme: {query_id: composite}}``."""
    return {s: {q: composite(v) for q, v in qmap.items()} for s, qmap in scores.items()}


def dim_means(scores):
    """``{scheme: {dim: mean, ..., 'n': n, 'composite': mean}}``."""
    out = {}
    for s, qmap in scores.items():
        vals = list(qmap.values())
        row = {d: st.mean(v[d] for v in vals) for d in DIMS}
        row["composite"] = st.mean(composite(v) for v in vals)
        row["n"] = len(vals)
        out[s] = row
    return out


def book_of(scores, scheme="ours"):
    return {q: v["book"] for q, v in scores[scheme].items()}


def clustered_bootstrap(deltas_by_book, B=BOOT_DEFAULT, seed=SEED, rng=None, p_mode="table7"):
    """Book-clustered paired bootstrap over per-question deltas.

    ``deltas_by_book``: ``{book: [delta, ...]}``. Books are resampled with
    replacement; every resample keeps the questions inside a book together.

    ``p_mode`` selects the p-value convention:
      * ``"table7"``  - ``(min(n>=0, n<=0) + 1) / (B + 1)``, the convention of the
        main-results analysis scripts (Table 7);
      * ``"twosided"`` - ``2 * min(n>=0, n<=0) / B``, the convention of the
        purity test (Section 6.2 / CR recomputation script).

    The 95% CI is always ``[dist[int(B*0.025)], dist[int(B*0.975)]]`` after
    sorting. Pass a shared ``rng`` (``random.Random(SEED)``) to reproduce the
    paper's exact draws, which are consumed sequentially across comparisons.

    Returns ``(obs_mean, lo95, hi95, p)``.
    """
    rng = rng or random.Random(seed)
    books = list(deltas_by_book)
    n = sum(len(v) for v in deltas_by_book.values())
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    obs = sum(sum(v) for v in deltas_by_book.values()) / n
    dist = []
    for _ in range(B):
        s, nn = 0.0, 0
        for _ in range(len(books)):
            b = rng.choice(books)
            s += sum(deltas_by_book[b])
            nn += len(deltas_by_book[b])
        dist.append(s / nn)
    dist.sort()
    lo, hi = dist[int(B * 0.025)], dist[int(B * 0.975)]
    if p_mode == "twosided":
        p = 2 * min(sum(1 for x in dist if x >= 0), sum(1 for x in dist if x <= 0)) / B
    else:
        p = (min(sum(1 for x in dist if x >= 0), sum(1 for x in dist if x <= 0)) + 1) / (B + 1)
    return obs, lo, hi, min(1.0, p)


def paired_composite_deltas(scores, baseline="ours"):
    """Delta of every other scheme relative to ``baseline``, grouped by book."""
    base = {q: composite(v) for q, v in scores[baseline].items()}
    book_map = book_of(scores, baseline)
    out = {}
    for s, qmap in scores.items():
        if s == baseline:
            continue
        d = defaultdict(list)
        for q, v in qmap.items():
            if q in base:
                d[book_map[q]].append(composite(v) - base[q])
        out[s] = d
    return out


def holm(pvals):
    """Holm-style correction, matching the paper's analysis scripts.

    Comparisons are ranked by raw p ascending and the k-th one is multiplied by
    ``(m - k)``; this is the convention used for Table 7 in the paper. The
    returned list is aligned with the *input* order.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    for k, i in enumerate(order):
        adj[i] = min(1.0, pvals[i] * (m - k))
    return adj


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def round2(x):
    """Two-stage display rounding used in the paper: 3 decimals first, then 2 decimals.

    Converts 4.644714 -> 4.645 -> 4.65 and 92.66493 -> 92.665 -> 92.67, matching
    the tables of the manuscript (plain one-step rounding would give 4.64/92.66).
    """
    from decimal import ROUND_HALF_UP, Decimal
    d = Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    d = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(d)


def markdown_table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    def fmt(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    out = [fmt(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [fmt(r) for r in rows]
    return "\n".join(out)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("saved", os.path.relpath(path, REPO))


def save_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("saved", os.path.relpath(path, REPO))


OUT = os.path.join(HERE, "output")
