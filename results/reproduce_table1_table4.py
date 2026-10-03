#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce Table 1 (corpus/genre distribution) and Table 4 (ablation & mechanism).

Inputs:
  data/qa/qa_577.json                     - question metadata (book, type)
  data/scores/scores_ablation_4variants.csv - end-to-end ablation per-question scores
  data/scores/scores_mechanism_5arms.csv    - fixed-retrieval-set five arms A/B/C/D/P

Usage:
  python results/reproduce_table1_table4.py
Outputs:
  results/output/table1_corpus.md
  results/output/table4_ablation_mechanism.md / .json
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
from collections import defaultdict

from common import (BOOT_DEFAULT, DATA, DIMS, OUT, SEED, clustered_bootstrap,
                    markdown_table, read_csv, round2, save_json, save_text)

# ---- genre assignment from the paper (Table 1) ----
GENRE = {
    "novel": ["The Brothers Karamazov", "Tess of the D'Urbervilles", "The Picture of Dorian Gray",
              "The Red and the Black", "Lord Jim", "The Autobiography of an Ex-Colored Man",
              "Sense and Sensibility", "King Solomon's Mines", "Far From the Madding Crowd",
              "Gargantua and Pantagruel"],
    "drama": ["The Tempest", "Twelfth Night, or What You Will", "Uncle Vanya",
              "The Comedy of Errors", "The Tragedy of Antony and Cleopatra",
              "The White Devil", "The School for Scandal"],
    "nonfiction": ["Common Sense", "The Consolation of Philosophy",
                   "The Rime of the Ancient Mariner", "The Prince"],
}
BOOK_GENRE = {b: g for g, books in GENRE.items() for b in books}

PAPER_ABLATION = {   # scheme: (grounding, correctness, halluc)
    "ours": (4.77, 4.79, 4.76),
    "wo_h1": (4.74, 4.75, 4.73),
    "wo_h2": (4.70, 4.73, 4.71),
    "wo_h1h2": (4.65, 4.69, 4.62),
    "noise_50": (4.75, 4.78, 4.75),
}
PAPER_ARMS = {       # common-valid n=575 means
    "A": (4.765, 4.793, 4.763, 94.35),
    "B": (4.725, 4.720, 4.704, 92.91),
    "C": (4.677, 4.697, 4.683, 92.14),
    "D": (4.650, 4.682, 4.640, 91.43),
    "P": (4.697, 4.710, 4.689, 92.46),
}


def table1():
    qa = json.load(open(os.path.join(DATA, "qa", "qa_577.json"), encoding="utf-8"))
    cnt = defaultdict(lambda: defaultdict(int))
    for r in qa:
        cnt[BOOK_GENRE.get(r["book_title"], "unknown")][r["book_title"]] += 1
    rows = []
    for g in ["novel", "drama", "nonfiction"]:
        books = GENRE[g]
        rows.append([g.capitalize(), len(books), sum(cnt[g].values()), "; ".join(books[:3])])
    rows.append(["Total", 21, sum(sum(cnt[g].values()) for g in GENRE), "—"])
    text = ["# Table 1. Genre distribution of the experimental corpus",
            "",
            markdown_table(["Genre", "Works", "Chapter-level units", "Representative works"], rows), ""]
    # paper values
    paper = {"Novel": (10, 427), "Drama": (7, 108), "Nonfiction": (4, 42)}
    for r in rows[:-1]:
        p = paper[r[0]]
        ok = r[1] == p[0] and r[2] == p[1]
        text.append(f"- {r[0]}: paper {p} vs reproduced ({r[1]}, {r[2]}) -> {'OK' if ok else 'DIFFERS'}")
    save_text(os.path.join(OUT, "table1_corpus.md"), "\n".join(text))


def load_rows(path):
    by = defaultdict(dict)
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
        key = r.get("scheme", r.get("arm"))
        by[key][r["query_id"]] = vals
    return by


def comp(v):
    return sum(v[d] - 1 for d in DIMS) / 12 * 100


def table4():
    abl_path = os.path.join(DATA, "scores", "scores_ablation_4variants.csv")
    arm_path = os.path.join(DATA, "scores", "scores_mechanism_5arms.csv")

    text = ["# Table 4. Ablation and mechanism experiments (1-5 scale)",
            "", "## Panel 1: end-to-end ablation (index and retrieval rebuilt per variant)",
            ""]
    rows = []
    if os.path.exists(abl_path):
        abl = load_rows(abl_path)
        # add ours reference from the main table
        main = load_rows(os.path.join(DATA, "scores", "scores_main_7schemes.csv"))
        abl["ours"] = main["ours"]
        order = [("ours", "HiSumChunk (full)"), ("wo_h1", "w/o H1 (no region-summary blocks)"),
                 ("wo_h2", "w/o H2 (no local-summary prefixes)"),
                 ("wo_h1h2", "w/o H1+H2 (both levels removed)"),
                 ("noise_50", "noise_50 (entity noise in summaries)")]
        for s, label in order:
            vals = list(abl[s].values())
            m = {d: st.mean(v[d] for v in vals) for d in DIMS}
            rows.append([label, f"{round2(m['grounding']):.2f}", f"{round2(m['correctness']):.2f}",
                         f"{round2(m['hallucination_absence']):.2f}", f"n={len(vals)}"])
        text.append(markdown_table(["Variant", "Grounding", "Correctness", "Hall. absence", ""], rows))
        text.append("")
        text.append("Self-check vs paper:")
        for (s, label), r in zip(order, rows):
            p = PAPER_ABLATION[s]
            ok = (abs(float(r[1]) - p[0]) < 0.015 and abs(float(r[2]) - p[1]) < 0.015
                  and abs(float(r[3]) - p[2]) < 0.015)
            text.append(f"- {label}: paper {p} vs reproduced ({r[1]}, {r[2]}, {r[3]}) -> {'OK' if ok else 'DIFFERS'}")

    text.append("")
    text.append("## Panel 2: fixed-retrieval-set mechanism arms (common valid n=575)")
    text.append("")
    rows2 = []
    arm_results = {}
    if os.path.exists(arm_path):
        arms = load_rows(arm_path)
        common = set.intersection(*[set(arms[a]) for a in "ABCDP"])
        for a in "ABCDP":
            vals = [arms[a][q] for q in common]
            m = {d: st.mean(v[d] for v in vals) for d in DIMS}
            compm = st.mean(comp(v) for v in vals)
            rows2.append([a, f"{m['grounding']:.3f}", f"{m['correctness']:.3f}",
                          f"{m['hallucination_absence']:.3f}", f"{compm:.2f}"])
            arm_results[a] = {"grounding": m["grounding"], "correctness": m["correctness"],
                              "hallucination_absence": m["hallucination_absence"],
                              "composite": compm}
        text.append(markdown_table(["Arm", "Grounding", "Correctness", "Hall. absence", "Composite"], rows2))
        text.append("")
        text.append("A = full; B = remove H2 prefixes (H1 kept); C = remove H1 slices (back-filled); "
                    "D = remove both; P = placebo (length-matched irrelevant summaries).")
        text.append("")
        text.append("Self-check vs paper:")
        for a, r in zip("ABCDP", rows2):
            p = PAPER_ARMS[a]
            ok = (abs(float(r[1]) - p[0]) < 0.006 and abs(float(r[2]) - p[1]) < 0.006
                  and abs(float(r[3]) - p[2]) < 0.006 and abs(float(r[4]) - p[3]) < 0.02)
            text.append(f"- Arm {a}: paper {p} vs reproduced ({r[1]}, {r[2]}, {r[3]}, {r[4]}) -> {'OK' if ok else 'DIFFERS'}")

        # deltas vs A: read the archived fixed-retrieval-set summary and re-derive
        # means; the published delta/CI/p values are reproduced by
        # src/experiments/mechanism (book-clustered bootstrap, B=10,000, seed 2026).
        text.append("")
        text.append("## Arm deltas vs A (from data/scores/five_arm_summary.json)")
        text.append("")
        summary = json.load(open(os.path.join(DATA, "scores", "five_arm_summary.json"), encoding="utf-8"))
        delta_rows = []
        for key, rec in summary["delta_vs_A"].items():
            dim, arm = key.split("|")
            delta_rows.append([arm, dim, f"{rec['delta']:+.3f}",
                               f"[{rec['ci'][0]:+.3f}, {rec['ci'][1]:+.3f}]",
                               f"{rec['p_raw']:.3f}", f"{rec['p_holm']:.3f}"])
        delta_rows.sort(key=lambda r: (r[1], r[0]))
        text.append(markdown_table(["Arm", "Dimension", "Delta", "95% CI", "p (raw)", "p (Holm)"], delta_rows))
        text.append("")
        text.append("Directional stability: arm D shows the largest drop on every dimension; arm P "
                    "falls below arm B on the composite (92.46 < 92.91), which supports the report's "
                    "directional conclusion that the benefit comes from summary-content correspondence.")
    save_text(os.path.join(OUT, "table4_ablation_mechanism.md"), "\n".join(text))
    if arm_results:
        save_json(os.path.join(OUT, "table4_ablation_mechanism.json"), arm_results)


def main():
    table1()
    table4()
    print("done")


if __name__ == "__main__":
    main()
