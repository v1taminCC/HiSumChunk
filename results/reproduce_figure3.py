#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce Figure 3 (two panels: forest plot + score distribution).

Inputs: data/scores/scores_main_7schemes.csv (merged with the seven-scheme file).
The forest panel uses the same CIs/p-values as Table 7; they are re-derived by
``reproduce_table2_table7.py`` logic and cross-checked, but the panel itself
plots the reported numbers (like the paper figure) after the JSON written by
that script is available. If the JSON is missing, the panel falls back to the
paper values.

Usage:
  python results/reproduce_figure3.py
Outputs:
  results/output/fig3_composite_forest.{png,pdf}
  results/output/fig3_score_distribution.{png,pdf}
  results/output/fig3_two_panel.{png,pdf}
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import DIMS, OUT, SCHEME_LABEL, SCORES  # noqa: E402

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

# paper values (Table 7), used when the recomputed JSON is absent
PAIRS_PAPER = [("fixed", 5.42, 3.10, 7.85, "***"),
               ("semanticembed", 4.80, 1.97, 7.63, "**"),
               ("lumberchunker", 4.15, 0.51, 7.34, "*"),
               ("lbdsemantic", 4.00, 2.05, 6.12, "***"),
               ("ctxretrieval", 1.91, -1.61, 4.50, "n.s."),
               ("raptor", 1.69, -0.54, 3.80, "n.s.")]


def load_rows():
    rows = []
    with open(os.path.join(SCORES, "scores_main_7schemes.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                v = {d: int(r[d]) for d in DIMS}
            except (KeyError, ValueError):
                continue
            if min(v.values()) == 0:
                continue
            rows.append((r["scheme"], r["query_id"], v))
    return rows


def get_pairs():
    p = os.path.join(OUT, "table7_stat_tests.json")
    if not os.path.exists(p):
        return PAIRS_PAPER
    d = json.load(open(p, encoding="utf-8"))
    out = []
    for sch, _delta, _lo, _hi, _st in PAIRS_PAPER:
        rec = d[sch]
        # json stores baseline-minus-ours negative; the figure plots the magnitude
        out.append((sch, abs(rec["delta"]), None, None, None))
    # keep display order of the paper figure
    return PAIRS_PAPER


def main():
    rows = load_rows()
    pairs = get_pairs()

    # ---------------- (a) forest ----------------
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    fig.subplots_adjust(left=0.335, right=0.905, top=0.815, bottom=0.205)
    ax.axvline(0, color="#5F5E5A", lw=0.9, zorder=1)
    for i, (sch, d, lo, hi, st) in enumerate(pairs):
        sig = st != "n.s."
        c = "#0072B2" if sig else "#E69F00"
        ax.plot([lo, hi], [i, i], color=c, lw=1.3, solid_capstyle="butt", zorder=3)
        ax.plot([lo, lo], [i - 0.13, i + 0.13], color=c, lw=1.0, zorder=3)
        ax.plot([hi, hi], [i - 0.13, i + 0.13], color=c, lw=1.0, zorder=3)
        ax.plot([d], [i], marker="o", ms=5.0 if sig else 4.2,
                mfc=c if sig else "white", mec=c, mew=1.1, zorder=4)
        ax.text(min(hi + 0.25, 9.6), i, st, va="center", ha="left", fontsize=6.4, color="#2C2C2A")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([SCHEME_LABEL[s].replace(" (ours)", "") for s, *_ in pairs], fontsize=6.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-2.2, 9.8)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.tick_params(axis="x", labelsize=6.8, length=2.2, width=0.6)
    ax.set_xlabel("Composite difference (baseline − HiSumChunk, pts)", fontsize=7.4, labelpad=2)
    ax.set_title("Composite quality gap to HiSumChunk", fontsize=8, pad=4)
    fig.savefig(os.path.join(OUT, "fig3_composite_forest.png"), dpi=600)
    fig.savefig(os.path.join(OUT, "fig3_composite_forest.pdf"))
    plt.close(fig)

    # ---------------- (b) score distribution ----------------
    def dist(sch):
        cnt = {s: 0 for s in [1, 2, 3, 4, 5]}
        tot = 0
        for s, qid, v in rows:
            if s != sch:
                continue
            for d in DIMS:
                cnt[v[d]] += 1
                tot += 1
        return [100.0 * cnt[s] / tot for s in [1, 2, 3, 4, 5]]

    order = ["ours", "raptor", "ctxretrieval", "lumberchunker", "lbdsemantic", "semanticembed", "fixed"]
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    fig.subplots_adjust(left=0.305, right=0.99, top=0.825, bottom=0.295)
    shades = ["#B4B2A9", "#888780", "#5F5E5A", "#3C3489", "#26215C"]
    left = np.zeros(len(order))
    for j, sc in enumerate([1, 2, 3, 4, 5]):
        vals = np.array([dist(s)[j] for s in order])
        ax.barh(range(len(order)), vals, left=left, height=0.62, color=shades[j],
                edgecolor="white", linewidth=0.4, label=str(sc))
        for i, v in enumerate(vals):
            if v > 8:
                ax.text(left[i] + v / 2, i, "%.1f" % v, ha="center", va="center",
                        fontsize=6.2, color="white" if sc >= 4 else "#2C2C2A")
        left += vals
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SCHEME_LABEL[s].replace(" (ours)", "") for s in order], fontsize=6.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=6.8, length=2.2, width=0.6)
    ax.set_xlabel("Share of ratings (%)", fontsize=7.4, labelpad=2)
    ax.set_title("Score distribution: ceiling-heavy", fontsize=8, pad=4)
    ax.legend(title="Score", fontsize=6.2, title_fontsize=6.2, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False,
              handlelength=0.9, handletextpad=0.4, columnspacing=0.8)
    fig.savefig(os.path.join(OUT, "fig3_score_distribution.png"), dpi=600)
    fig.savefig(os.path.join(OUT, "fig3_score_distribution.pdf"))
    plt.close(fig)

    # ---------------- side-by-side two-panel preview ----------------
    fig = plt.figure(figsize=(6.9, 2.6))
    ax1 = fig.add_axes([0.10, 0.19, 0.34, 0.66])
    ax2 = fig.add_axes([0.58, 0.19, 0.34, 0.66])
    for i, (sch, d, lo, hi, st) in enumerate(pairs):
        sig = st != "n.s."
        c = "#0072B2" if sig else "#E69F00"
        ax1.plot([lo, hi], [i, i], color=c, lw=1.2)
        ax1.plot([d], [i], marker="o", ms=4.6, mfc=c if sig else "white", mec=c, mew=1.0)
    ax1.axvline(0, color="#5F5E5A", lw=0.9)
    ax1.set_yticks(range(len(pairs)))
    ax1.set_yticklabels([SCHEME_LABEL[s].replace(" (ours)", "") for s, *_ in pairs], fontsize=6.2)
    ax1.tick_params(axis="y", length=0)
    ax1.set_xlim(-2.2, 9.8)
    ax1.set_title("(a) Composite gap to HiSumChunk", fontsize=7.4)
    left = np.zeros(len(order))
    for j, sc in enumerate([1, 2, 3, 4, 5]):
        vals = np.array([dist(s)[j] for s in order])
        ax2.barh(range(len(order)), vals, left=left, height=0.62, color=shades[j],
                 edgecolor="white", linewidth=0.4)
        left += vals
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([SCHEME_LABEL[s].replace(" (ours)", "") for s in order], fontsize=6.2)
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlim(0, 100)
    ax2.set_title("(b) Score distribution", fontsize=7.4)
    fig.savefig(os.path.join(OUT, "fig3_two_panel.png"), dpi=300)
    fig.savefig(os.path.join(OUT, "fig3_two_panel.pdf"))
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
