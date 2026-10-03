#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce Table 2 (main results) and Table 7 / Appendix A.2 (statistical tests).

Zero-cost: reads ``data/scores/scores_main_7schemes.csv`` only.

Usage:
  python results/reproduce_table2_table7.py
Outputs:
  results/output/table2_main_results.md / .json
  results/output/table7_stat_tests.md / .json
"""
from __future__ import annotations

import os

from common import (BOOT_DEFAULT, DATA, DIM_LABEL, DIMS, OUT, SCHEME_LABEL,
                    SCHEME_ORDER, SCORES, clustered_bootstrap, dim_means,
                    holm, load_main_scores, markdown_table, paired_composite_deltas,
                    read_csv, round2, save_json, save_text, stars)

PAPER_TABLE2 = {
    # scheme: (grounding, correctness, hallucination, composite, rouge, bertscore, eprec, erecall, unsupported)
    "fixed": (4.58, 4.59, 4.51, 88.95, 0.2144, 0.8609, 0.5302, 0.6909, 0.2018),
    "lumberchunker": (4.65, 4.65, 4.53, 90.22, 0.2068, 0.8594, 0.5027, 0.6898, 0.2545),
    "lbdsemantic": (4.63, 4.63, 4.58, 90.37, 0.2116, 0.8601, 0.5159, 0.6891, 0.2420),
    "semanticembed": (4.60, 4.60, 4.55, 89.57, 0.2120, 0.8603, 0.5142, 0.6771, 0.2298),
    "raptor": (4.69, 4.72, 4.71, 92.67, 0.2203, 0.8638, 0.4960, 0.6691, 0.1534),
    "ctxretrieval": (4.67, 4.73, 4.69, 92.46, 0.2438, 0.8776, 0.5388, 0.6744, 0.1678),
    "ours": (4.77, 4.79, 4.76, 94.37, 0.2516, 0.8790, 0.5462, 0.6743, 0.1809),
}


def main():
    scores = load_main_scores()
    means = dim_means(scores)

    # ---- automatic metrics ----
    auto = {r["metric"]: r for r in read_csv(os.path.join(SCORES, "auto_metrics_7schemes.csv"))}

    def a(metric, scheme):
        return float(auto[metric][scheme])

    rows = []
    for s in SCHEME_ORDER:
        m = means[s]
        rows.append([
            SCHEME_LABEL[s],
            f"{round2(m['grounding']):.2f}", f"{round2(m['correctness']):.2f}",
            f"{round2(m['hallucination_absence']):.2f}",
            f"{round2(m['composite']):.2f}",
            f"{a('grounding_context_rougeL', s):.4f}",
            f"{a('grounding_context_bertscore_f1', s):.4f}",
            f"{a('correctness_entity_precision', s):.4f}",
            f"{a('correctness_entity_recall', s):.4f}",
            f"{a('hallucination_unsupported_entity_rate', s):.4f}",
        ])
    headers = ["Method", "Grounding", "Correctness", "Hall. absence", "Composite",
               "Ctx ROUGE-L", "Ctx BERTScore", "Ent. prec.", "Ent. recall", "Unsup. ent. rate"]
    text = ["# Table 2. Main results of the seven methods",
            "",
            "Dimensions are 1-5 LLM-as-Judge means; composite is the 100-point equal-weight average of "
            "the three dimensions. 4,038 valid answer records (RAPTOR has 576).",
            "", markdown_table(headers, rows), ""]

    # self-check against the values printed in the paper
    checks = []
    for s in SCHEME_ORDER:
        m = means[s]
        g, c, h, comp = PAPER_TABLE2[s][:4]
        checks.append((s, abs(round2(m["grounding"]) - g) < 1e-9, abs(round2(m["correctness"]) - c) < 1e-9,
                       abs(round2(m["hallucination_absence"]) - h) < 1e-9, abs(round2(m["composite"]) - comp) < 1e-9))
    n_ok = sum(1 for ch in checks if all(ch[1:]))
    text.append(f"Self-check vs paper: {n_ok}/{len(checks)} methods match within 0.01 (two-stage rounding).")
    save_text(os.path.join(OUT, "table2_main_results.md"), "\n".join(text))
    save_json(os.path.join(OUT, "table2_main_results.json"),
              {s: {**{d: round(means[s][d], 6) for d in DIMS},
                   "composite": round(means[s]["composite"], 6), "n": means[s]["n"]}
               for s in SCHEME_ORDER})

    # ---- Table 7: composite difference vs ours, book-clustered bootstrap + Holm ----
    # The paper consumes the bootstrap RNG sequentially in this comparison order
    # (the order of the original analysis pipeline); this reproduces the printed
    # CIs and p-values exactly.
    import random
    from common import SEED
    deltas = paired_composite_deltas(scores, baseline="ours")
    order = ["fixed", "lumberchunker", "lbdsemantic", "semanticembed", "raptor", "ctxretrieval"]
    rng = random.Random(SEED)
    res = []
    for s in order:
        obs, lo, hi, p = clustered_bootstrap(deltas[s], B=BOOT_DEFAULT, rng=rng)
        res.append([SCHEME_LABEL[s], f"{obs:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", p])
    adj = holm([r[3] for r in res])
    # sort by raw p ascending, as in the paper's Table 7
    combo = sorted(zip(res, adj), key=lambda t: t[0][3])
    rows7 = []
    for r, pa in combo:
        rows7.append([r[0], r[1], r[2], f"{r[3]:.4f}", f"{pa:.4f}", stars(pa)])
    paper7 = {"fixed": (5.416, 3.096, 7.850, 0.0003),
              "semanticembed": (4.795, 1.971, 7.632, 0.0018),
              "lumberchunker": (4.145, 0.511, 7.344, 0.0394),
              "lbdsemantic": (4.001, 2.052, 6.115, 0.0005),
              "ctxretrieval": (1.906, -1.614, 4.502, 0.1246),
              "raptor": (1.693, -0.539, 3.797, 0.1219)}
    text7 = ["# Table 7 (= Appendix A.2). Composite-score difference vs HiSumChunk",
             "",
             f"Per-question paired differences, works resampled as clusters ({BOOT_DEFAULT:,} resamples, seed 2026), "
             "Holm correction over the six comparisons. Rows sorted by raw p.",
             "",
             markdown_table(["Comparison", "Delta (pts)", "95% CI", "p (raw)", "p (Holm)", "Verdict"], rows7),
             "",
             "Self-check of Holm p-values against the paper:",
             ""]
    for r, pa in zip(res, adj):
        s = [k for k, v in SCHEME_LABEL.items() if v == r[0]][0]
        ok = abs(pa - paper7[s][3]) < 0.01
        text7.append(f"- vs {r[0]}: paper {paper7[s][3]:.4f} vs reproduced {pa:.4f} -> {'OK' if ok else 'DIFFERS'}")
    save_text(os.path.join(OUT, "table7_stat_tests.md"), "\n".join(text7))
    save_json(os.path.join(OUT, "table7_stat_tests.json"),
              {[k for k, v in SCHEME_LABEL.items() if v == r[0]][0]:
               {"delta": float(r[1]), "ci": r[2], "p_raw": r[3], "p_holm": pa}
               for r, pa in zip(res, adj)})
    print("done")


if __name__ == "__main__":
    main()
