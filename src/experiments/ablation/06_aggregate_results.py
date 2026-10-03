#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate judgment results from ablation experiment, generate comparison table.
Includes our full system results from main experiment as reference.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import JUDGMENTS_BY_BOOK_DIR, RESULTS_DIR, VARIANTS
from utils_io import read_json, write_json

DIMS = ["grounding", "correctness", "hallucination_absence"]
DIM_LABELS = {
    "grounding": "Grounding",
    "correctness": "Correctness",
    "hallucination_absence": "Hallucination Absence",
}

OURS_SCORES = {
    "grounding": 4.77,
    "correctness": 4.79,
    "hallucination_absence": 4.76,
}


def collect_scores(exclude_zero: bool = True) -> dict[str, dict[str, list[float]]]:
    by_variant: dict[str, dict[str, list[float]]] = defaultdict(lambda: {d: [] for d in DIMS})

    for variant in VARIANTS:
        for fp in sorted(JUDGMENTS_BY_BOOK_DIR.glob(f"*.{variant}.eval.json")):
            data = read_json(fp)
            for j in data:
                scores = [j.get(d, 0) for d in DIMS]
                if exclude_zero and any(s == 0 for s in scores):
                    continue
                for d, s in zip(DIMS, scores):
                    by_variant[variant][d].append(s)

    return by_variant


def build_report(exclude_zero: bool = True) -> str:
    by_variant = collect_scores(exclude_zero=exclude_zero)

    lines = []
    lines.append("=" * 72)
    suffix = " (zero-score records excluded)" if exclude_zero else ""
    lines.append(f"  Ablation Experiment Evaluation{suffix}")
    lines.append("=" * 72)
    lines.append("")

    all_variants = ["ours"] + VARIANTS
    col_w = 12
    label_w = 35

    hdr = f"  {'Dimension':<{label_w}}"
    for v in all_variants:
        hdr += f" {v:>{col_w}}"
    lines.append(hdr)
    lines.append("  " + "-" * (label_w + (col_w + 1) * len(all_variants)))

    for d in DIMS:
        label = DIM_LABELS[d]
        row = f"  {label:<{label_w}}"
        row += f" {OURS_SCORES[d]:>{col_w}.2f}"
        for v in VARIANTS:
            vals = by_variant[v][d]
            avg = round(sum(vals) / len(vals), 2) if vals else 0.0
            row += f" {avg:>{col_w}.2f}"
        lines.append(row)

    lines.append("")

    # Record count
    lines.append("  Records:")
    for v in VARIANTS:
        vals = by_variant[v][DIMS[0]]
        lines.append(f"    {v}: {len(vals)} valid records")

    lines.append("")

    # Delta analysis
    lines.append("  Delta vs. full system (ours):")
    lines.append(f"  {'Dimension':<{label_w}}" + "".join(f" {v:>{col_w}}" for v in VARIANTS))
    lines.append("  " + "-" * (label_w + (col_w + 1) * len(VARIANTS)))
    for d in DIMS:
        label = DIM_LABELS[d]
        row = f"  {label:<{label_w}}"
        for v in VARIANTS:
            vals = by_variant[v][d]
            avg = round(sum(vals) / len(vals), 2) if vals else 0.0
            delta = avg - OURS_SCORES[d]
            row += f" {delta:>{col_w}.2f}"
        lines.append(row)

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def build_json_summary(exclude_zero: bool = True) -> dict:
    by_variant = collect_scores(exclude_zero=exclude_zero)
    rows = []

    rows.append({
        "variant": "ours",
        "n_records": "N/A (from main experiment)",
        **{d: OURS_SCORES[d] for d in DIMS},
    })

    for v in VARIANTS:
        vals = by_variant[v]
        n = len(vals[DIMS[0]])
        row = {"variant": v, "n_records": n}
        for d in DIMS:
            row[d] = round(sum(vals[d]) / len(vals[d]), 2) if vals[d] else 0.0
            row[f"{d}_delta"] = round(row[d] - OURS_SCORES[d], 2)
        rows.append(row)

    return {"dimensions": DIMS, "rows": rows}


def main():
    print("=" * 60)
    print("Ablation Experiment - Step 6: Aggregate results")
    print("=" * 60)
    ...
    print(f"\nReport saved: {report_path}")
    ...
    print(f"JSON saved: {json_path}")


if __name__ == "__main__":
    main()
