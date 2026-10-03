#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce Table 3 (context main-thread purity) and Figure 5 (context budget).

Input: data/scores/purity_per_question_7schemes.csv, which holds per-question
purity, top-5 word count, largest block and chapter count for all seven schemes
(the Contextual Retrieval row is recomputed from the released CR judgments
against the shared split boundaries).

Usage:
  python results/reproduce_table3_figure5.py
Outputs:
  results/output/table3_purity.md / .json
  results/output/fig5_context_budget.{png,pdf}
"""
from __future__ import annotations

import os
import random
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (BOOT_DEFAULT, OUT, SCORES, SCHEME_LABEL, SEED,
                    clustered_bootstrap, markdown_table, read_csv,
                    round2, save_json, save_text)  # noqa: E402

ORDER = ["ours", "ctxretrieval", "lumberchunker", "raptor",
         "semanticembed", "lbdsemantic", "fixed"]

PAPER_TABLE3 = {
    "ours": (58.4, 56.0, 3306, 770, 3.02),
    "ctxretrieval": (52.5, 46.0, 2912, 673, 3.31),
    "lumberchunker": (38.0, 35.0, 8263, 2776, 4.42),
    "raptor": (38.0, 29.3, 3542, 1024, 4.29),
    "semanticembed": (36.9, 34.1, 5684, 1621, 4.37),
    "lbdsemantic": (36.2, 33.6, 6725, 1926, 4.40),
    "fixed": (34.2, 39.8, 5048, 1021, 4.22),
}
PAPER_BOOTSTRAP = {  # vs ours, from the paper Section 6.2
    "ctxretrieval": (5.9, -4.7, 14.3, 0.27),
    "lumberchunker": (20.4, 15.8, 25.2, 0.0),
    "raptor": (20.5, 12.2, 28.5, 0.0),
    "semanticembed": (21.5, 15.8, 27.0, 0.0),
    "lbdsemantic": (22.3, 17.7, 26.8, 0.0),
    "fixed": (24.3, 19.3, 28.9, 0.0),
}


def load():
    by = defaultdict(dict)     # scheme -> qid -> row
    books = {}
    for r in read_csv(os.path.join(SCORES, "purity_per_question_7schemes.csv")):
        try:
            pur = float(r["purity"])
            words = int(float(r["total_words"]))
            mx = int(float(r["max_chunk"]))
            nch = int(float(r["n_chapters"]))
        except (KeyError, ValueError):
            continue
        by[r["scheme"]][r["query_id"]] = {"purity": pur, "words": words, "max": mx, "nch": nch}
        books[r["query_id"]] = r["book"]
    return by, books


def main():
    by, books = load()

    rows = []
    for s in ORDER:
        vals = list(by[s].values())
        m = st.mean(v["purity"] for v in vals) * 100
        med = st.median(v["purity"] for v in vals) * 100
        w = st.mean(v["words"] for v in vals)
        mx = st.mean(v["max"] for v in vals)
        nch = st.mean(v["nch"] for v in vals)
        rows.append([SCHEME_LABEL[s], f"{m:.1f}%", f"{med:.1f}%", f"{round(w):,}",
                     f"{round(mx):,}", f"{nch:.2f}"])
    text = ["# Table 3. Context main-thread purity (577 questions)",
            "",
            "Purity = share of top-5 words falling in the largest chapter/scene-level unit; "
            "the largest-block column is the mean word count of that unit.",
            "",
            markdown_table(["Method", "Mean purity", "Median purity", "Mean top-5 words",
                            "Mean largest block", "Chapters in top-5"], rows),
            "",
            "Self-check vs paper:"]
    for s, r in zip(ORDER, rows):
        p = PAPER_TABLE3[s]
        ok = (abs(float(r[1].rstrip("%")) - p[0]) < 0.05 and abs(round(float(r[3].replace(",", "")) - p[2]) < 1.5))
        text.append(f"- {SCHEME_LABEL[s]}: paper mean {p[0]}% / words {p[2]:,} vs reproduced {r[1]} / {r[3]} -> {'OK' if ok else 'DIFFERS'}")
    save_text(os.path.join(OUT, "table3_purity.md"), "\n".join(text))
    save_json(os.path.join(OUT, "table3_purity.json"),
              {s: {
                  "mean_purity": round(st.mean(v["purity"] for v in by[s].values()) * 100, 4),
                  "median_purity": round(st.median(v["purity"] for v in by[s].values()) * 100, 4),
                  "mean_top5_words": round(st.mean(v["words"] for v in by[s].values()), 1),
                  "mean_largest_block": round(st.mean(v["max"] for v in by[s].values()), 1),
                  "mean_chapters": round(st.mean(v["nch"] for v in by[s].values()), 2)}
               for s in ORDER})

    # ---- purity bootstrap: ours vs each baseline, clustered by book ----
    rng = random.Random(SEED)
    res = []
    for s in ["fixed", "lumberchunker", "raptor", "semanticembed", "lbdsemantic", "ctxretrieval"]:
        db = defaultdict(list)
        for q, rec in by["ours"].items():
            if q in by[s]:
                db[books[q]].append(rec["purity"] - by[s][q]["purity"])
        obs, lo, hi, p = clustered_bootstrap(db, B=BOOT_DEFAULT, rng=rng, p_mode="twosided")
        res.append([SCHEME_LABEL[s], 100 * obs, 100 * lo, 100 * hi, p])
    rows2 = [[r[0], f"{r[1]:+.1f}", f"[{r[2]:+.1f}, {r[3]:+.1f}]", f"{r[4]:.3f}"] for r in res]
    text2 = ["", "## Purity difference vs HiSumChunk (percentage points)",
             "", f"Book-clustered paired bootstrap, {BOOT_DEFAULT:,} resamples, seed 2026.",
             "", markdown_table(["Comparison", "Delta (pp)", "95% CI", "p"], rows2), "",
             "Self-check vs paper Section 6.2:"]
    for r in res:
        s = [k for k, v in SCHEME_LABEL.items() if v == r[0]][0]
        p = PAPER_BOOTSTRAP[s]
        ok = abs(r[1] - p[0]) < 1.0
        text2.append(f"- vs {r[0]}: paper {p[0]:+.1f} [{p[1]:+.1f}, {p[2]:+.1f}] p={p[3]:.2f} "
                     f"vs reproduced {r[1]:+.1f} [{r[2]:+.1f}, {r[3]:+.1f}] p={r[4]:.2f} -> {'OK' if ok else 'DIFFERS'}")
    save_text(os.path.join(OUT, "table3_purity.md"), "\n".join(text + text2))

    # ---- Figure 5: context budget mean vs median ----
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    fig.subplots_adjust(left=0.385, right=0.985, top=0.815, bottom=0.195)
    budget = []
    for s in ["ctxretrieval", "ours", "raptor", "fixed", "semanticembed", "lbdsemantic", "lumberchunker"]:
        vals = list(by[s].values())
        budget.append((SCHEME_LABEL[s].replace(" (ours)", ""),
                       round(st.mean(v["words"] for v in vals)),
                       round(st.median(v["words"] for v in vals))))
    for i, (name, mean, med) in enumerate(budget):
        ax.plot([mean, med], [i, i], color="#B4B2A9", lw=1.0, zorder=2)
        ax.plot([mean], [i], marker="o", ms=5.0, mfc="#0072B2", mec="#0072B2", zorder=4)
        ax.plot([med], [i], marker="s", ms=4.4, mfc="white", mec="#D85A30", mew=1.1, zorder=4)
    ax.set_yticks(range(len(budget)))
    ax.set_yticklabels([b[0] for b in budget], fontsize=6.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(2600, 8900)
    ax.set_xticks([3000, 4500, 6000, 7500])
    ax.set_xticklabels(["3.0k", "4.5k", "6.0k", "7.5k"], fontsize=6.8)
    ax.tick_params(axis="x", length=2.2, width=0.6)
    ax.set_xlabel("top-5 context size (words)", fontsize=7.4, labelpad=2)
    ax.set_title("Context budget: mean vs median", fontsize=8, pad=4)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", mfc="#0072B2", mec="#0072B2", ms=5.0, label="Mean"),
               Line2D([], [], marker="s", ls="", mfc="white", mec="#D85A30", mew=1.1, ms=4.4, label="Median")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.4, frameon=False,
              handletextpad=0.4, borderpad=0.2, labelspacing=0.3)
    fig.savefig(os.path.join(OUT, "fig5_context_budget.png"), dpi=600)
    fig.savefig(os.path.join(OUT, "fig5_context_budget.pdf"))
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
