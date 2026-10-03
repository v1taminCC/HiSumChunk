#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from utils_io import ensure_dir, read_json, write_json

JUDGMENTS_DIR = BASE_DIR / "results" / "answer_judgments"
REPORTS_DIR = JUDGMENTS_DIR / "reports"

DEFAULT_TOP_K = 5


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_pct(num: float, denom: float) -> str:
    if denom == 0:
        return "N/A"
    return f"{num / denom * 100:.1f}%"


JUDGE_DIMS = ["grounding", "correctness", "hallucination_absence"]


def aggregate(judgments: list[dict]) -> dict:
    by_scheme: dict[str, dict] = {}
    by_qt: dict[str, dict] = {}
    by_diff: dict[str, dict] = {}
    by_book: dict[str, dict] = {}

    def _init_group():
        g = {"count": 0}
        for d in JUDGE_DIMS:
            g[d] = []
        return g

    def _update_group(g: dict, scores: dict):
        g["count"] += 1
        for d in JUDGE_DIMS:
            g[d].append(scores.get(d, 0))

    for j in judgments:
        scheme = j.get("scheme", "unknown")
        qt = j.get("question_type", "unknown")
        diff = j.get("difficulty", "unknown")
        bk = j.get("book_title", "unknown")

        by_scheme.setdefault(scheme, _init_group())
        _update_group(by_scheme[scheme], j)

        key = f"{scheme}__{qt}"
        by_qt.setdefault(key, _init_group())
        _update_group(by_qt[key], j)

        key = f"{scheme}__{diff}"
        by_diff.setdefault(key, _init_group())
        _update_group(by_diff[key], j)

        key = f"{scheme}__{bk}"
        by_book.setdefault(key, _init_group())
        _update_group(by_book[key], j)

    def _flatten(g: dict) -> dict:
        f = {"count": g["count"]}
        for d in JUDGE_DIMS:
            f[f"mean_{d}"] = safe_mean(g[d])
        return f

    schemes = sorted(by_scheme.keys())
    overall = {s: _flatten(by_scheme[s]) for s in schemes}

    qt_flat = {}
    for key in sorted(by_qt.keys()):
        qt_flat[key] = _flatten(by_qt[key])

    diff_flat = {}
    for key in sorted(by_diff.keys()):
        diff_flat[key] = _flatten(by_diff[key])

    book_flat = {}
    for key in sorted(by_book.keys()):
        book_flat[key] = _flatten(by_book[key])

    return {
        "total_judgments": len(judgments),
        "overall": overall,
        "by_question_type": qt_flat,
        "by_difficulty": diff_flat,
        "by_book": book_flat,
    }


def format_report(agg: dict, mode: str, top_k: int) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(f"  Answer Quality Report — Independent Scoring Mode")
    lines.append(f"  Retrieval mode: {mode}  |  Context chunks: top-{top_k}  |  Judgments: {agg['total_judgments']}")
    lines.append("=" * 80)
    lines.append("")

    overall = agg["overall"]
    schemes = sorted(overall.keys())

    lines.append("[Overall Comparison]")
    header = f"  {'Metric':<24}"
    for s in schemes:
        header += f" {s:>10}"
    header += f" {'Δ':>10}"
    lines.append(header)
    lines.append("  " + "-" * 54)

    metrics = [
        ("grounding", "Grounding"),
        ("correctness", "Correctness"),
        ("hallucination_absence", "Hallucination Absence"),
    ]
    for key, label in metrics:
        vals = []
        for s in schemes:
            o = overall[s]
            vals.append(round(o.get(f"mean_{key}", 0), 2))
        delta = vals[0] - vals[1] if len(vals) >= 2 else 0
        line = f"  {label:<30}"
        for v in vals:
            line += f" {v:>10.2f}"
        line += f" {delta:>+10.2f}"
        lines.append(line)
    lines.append("")

    lines.append("[By Question Type (grounding / correctness)]")
    for qt in sorted(set(k.split("__", 1)[-1] for k in agg["by_question_type"])):
        line = f"  {qt:<22}"
        for s in schemes:
            key = f"{s}__{qt}"
            o = agg["by_question_type"].get(key, {})
            line += f" {o.get('mean_grounding', 0):>5.1f}/{o.get('mean_correctness', 0):>5.1f}"
        lines.append(line)
    lines.append("")

    lines.append("[By Difficulty (grounding / hallucination_absence)]")
    for diff in sorted(set(k.split("__", 1)[-1] for k in agg["by_difficulty"])):
        line = f"  {diff:<10}"
        for s in schemes:
            key = f"{s}__{diff}"
            o = agg["by_difficulty"].get(key, {})
            line += f" {o.get('mean_grounding', 0):>5.1f}/{o.get('mean_hallucination_absence', 0):>5.1f}"
        lines.append(line)
    lines.append("")

    lines.append("[By Book (grounding)]")
    for bk in sorted(set(k.split("__", 1)[-1] for k in agg["by_book"])):
        line = f"  {bk[:45]:<46}"
        for s in schemes:
            key = f"{s}__{bk}"
            o = agg["by_book"].get(key, {})
            line += f" {s}:{o.get('mean_grounding',0):.1f}"
        lines.append(line)
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate independent scoring results")
    parser.add_argument("--mode", type=str, default="in_book", choices=["in_book", "global"])
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    ensure_dir(REPORTS_DIR)

    jf = JUDGMENTS_DIR / f"judgments_all.{args.mode}.top{args.top_k}_eval.json"
    if not jf.exists():
        print(f"Judgment file not found: {jf}")
        return

    judgments = read_json(jf)
    print(f"Loaded judgments: {len(judgments)} records")

    agg = aggregate(judgments)

    json_path = REPORTS_DIR / f"answer_scores.{args.mode}.top{args.top_k}_eval.json"
    write_json(json_path, agg)

    report = format_report(agg, args.mode, args.top_k)
    txt_path = REPORTS_DIR / f"answer_scores.{args.mode}.top{args.top_k}_eval.txt"
    txt_path.write_text(report, encoding="utf-8")

    print(f"Summary reports saved:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
