# -*- coding: utf-8 -*-
"""Merges the 7-scheme automatic-metric tables: newly computed CR results + existing 6-scheme summary (read-only reference; not copied)

Usage:
  python 06_merge_7schemes.py
  python 06_merge_7schemes.py \
      --cr-summary  ../results/reports/auto_metrics_summary.txt \
      --reference   src/metrics/results_summary/auto_metrics_summary.txt

Output:
  ../results/reports/auto_metrics_summary_7schemes.txt
  ../results/tables/auto_metrics_7schemes.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

SECTION_RE = re.compile(r"^\[([A-Za-z0-9_]+)\]\s*n=(\d+),\s*valid=(\d+)")
METRIC_RE = re.compile(r"^\s*-\s*([A-Za-z0-9_]+):\s*([0-9.]+)")
LOWER_IS_BETTER = {"hallucination_unsupported_entity_rate"}


def parse_summary(path: Path) -> dict[str, dict]:
    schemes: dict[str, dict] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = SECTION_RE.match(line.strip())
        if m:
            current = m.group(1)
            schemes[current] = {"n": int(m.group(2)), "valid": int(m.group(3)), "metrics": {}}
            continue
        if current:
            m2 = METRIC_RE.match(line)
            if m2:
                schemes[current]["metrics"][m2.group(1)] = float(m2.group(2))
    return schemes


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--cr-summary", default=str(here.parent / "results" / "reports" / "auto_metrics_summary.txt"))
    ap.add_argument("--reference", default=str(Path(__file__).resolve().parent.parent / "results_summary" / "auto_metrics_summary.txt"))
    args = ap.parse_args()

    cr_path = Path(args.cr_summary)
    ref_path = Path(args.reference)
    if not cr_path.exists():
        print(f"[ERROR] CR summary not found: {cr_path} (run 01->05 first)")
        return 1

    merged: dict[str, dict] = {}
    order: list[str] = []
    if ref_path.exists():
        for s, v in parse_summary(ref_path).items():
            merged[s] = v
            order.append(s)
        print(f"[ok] 6-scheme summary: {order}")
    else:
        print(f"[WARN] 6-scheme summary not found ({ref_path}); outputting the CR table only")

    cr = parse_summary(cr_path)
    for s, v in cr.items():
        merged[s] = v
        if s not in order:
            order.append(s)
    print(f"[ok] CR summary: {list(cr)}")

    metrics: list[str] = []
    for s in order:
        for k in merged[s]["metrics"]:
            if k not in metrics:
                metrics.append(k)

    best = {}
    for k in metrics:
        vals = [(s, merged[s]["metrics"][k]) for s in order if k in merged[s]["metrics"]]
        if vals:
            best[k] = min(vals, key=lambda kv: kv[1])[0] if k in LOWER_IS_BETTER else max(vals, key=lambda kv: kv[1])[0]

    out_dir = here.parent / "results"
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    lines = ["=== Automatic-metric 7-scheme merged table ===", f"Input: CR={cr_path.name} | 6 schemes={ref_path.name}", ""]
    lines.append(f"{'metric':>40} | " + " | ".join(f"{s:>14}" for s in order) + " | best")
    lines.append("-" * len(lines[-1]))
    for k in metrics:
        row = f"{k:>40} | " + " | ".join(f"{merged[s]['metrics'].get(k, float('nan')):14.6f}" for s in order) + f" | {best.get(k,'')}"
        lines.append(row)
    lines.append("")
    lines.append(f"{'n':>40} | " + " | ".join(f"{merged[s]['n']:14d}" for s in order) + " |")
    lines.append(f"{'valid':>40} | " + " | ".join(f"{merged[s]['valid']:14d}" for s in order) + " |")
    report = "\n".join(lines)
    (out_dir / "reports" / "auto_metrics_summary_7schemes.txt").write_text(report, encoding="utf-8")

    with open(out_dir / "tables" / "auto_metrics_7schemes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", *order])
        for k in metrics:
            w.writerow([k, *[merged[s]["metrics"].get(k, "") for s in order]])
        w.writerow(["n", *[merged[s]["n"] for s in order]])
        w.writerow(["valid", *[merged[s]["valid"] for s in order]])

    print(report)
    print("\nWrote:", out_dir / "reports" / "auto_metrics_summary_7schemes.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
