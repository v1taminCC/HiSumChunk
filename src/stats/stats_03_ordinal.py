# -*- coding: utf-8 -*-
"""Ordinal regression robustness (statsmodels OrderedModel, book-clustered robust SE, contrast=ours-baseline)

Usage:
  python scripts/stats_03_ordinal.py [--input ../data/scores_main.csv] [--out ./out] [--ref fixed]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

DIMS = ['grounding', 'correctness', 'hallucination_absence']
ORDER = ['ours', 'fixed', 'lumberchunker', 'lbdsemantic', 'semanticembed', 'raptor']


def cluster_robust_vcov(res, groups: np.ndarray) -> np.ndarray:
    """Cluster-robust sandwich covariance: bread = inv(-H), meat = sum_g s_g s_g'"""
    params = res.params
    H = res.model.hessian(params)
    bread = np.linalg.inv(-H)
    try:
        scores = np.asarray(res.model.score_obs(params))
    except Exception:  # noqa: BLE001
        scores = np.asarray(res.model.score_obs_(params))
    scores = scores.reshape(len(groups), -1)
    k = scores.shape[1]
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        sg = scores[groups == g].sum(axis=0)
        meat += np.outer(sg, sg)
    return bread @ meat @ bread


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(Path(__file__).parent.parent / 'data' / 'scores_main.csv'))
    ap.add_argument('--out', default=str(Path(__file__).parent / 'out'))
    ap.add_argument('--ref', default='ours', help='regression reference group; contrast outputs ref minus other schemes')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df[df['scheme'] != 'booksum_split'].copy()
    schemes = [s for s in ORDER if s in set(df['scheme'])]
    lines = ['# Ordinal regression robustness (OrderedModel + book-clustered robust SE)', '',
             f'Input: {args.input} | reference group: {args.ref}', '']
    summary = {}

    for dim in DIMS:
        sub = df[df['scheme'].isin(schemes)][['scheme', 'book', dim]].dropna()
        sub = sub[sub[dim] > 0]
        if sub.empty:
            continue
        y = sub[dim].astype(int).to_numpy()
        books = sub['book'].astype('category')
        groups = books.cat.codes.to_numpy()
        dummies = pd.get_dummies(sub['scheme'], prefix='m', drop_first=False).astype(float)
        ref_col = f'm_{args.ref}'
        if ref_col not in dummies.columns:
            continue
        cols = [c for c in dummies.columns if c != ref_col]
        X = dummies[cols].copy()  # OrderedModel takes no explicit constant; the intercept is carried by the threshold parameters

        mod = OrderedModel(y, X, distr='logit')
        res = mod.fit(method='bfgs', disp=False, maxiter=500)
        V = cluster_robust_vcov(res, groups)

        # statsmodels 0.15: parameter order = [regression coefficients..., thresholds...]
        k_beta = X.shape[1]
        beta = np.asarray(res.params)[:k_beta]
        names = list(X.columns)
        idx = {n: i for i, n in enumerate(names)}
        report_rows = []
        # contrast: ref - other = -beta_other (ref is the reference group; its dummy is dropped)
        pair = {}
        for col in cols:
            other = col[len('m_'):]
            i = idx[col]
            diff = -beta[i]
            var = V[i, i]
            se = float(np.sqrt(var)) if var > 0 else float('nan')
            z = diff / se if se and se > 0 else float('nan')
            p = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else float('nan')
            pair[other] = {'diff_logit': float(diff), 'or': float(np.exp(diff)), 'se': se, 'p': float(p)}
            report_rows.append((other, diff, se, z, p))
        summary[dim] = pair

        lines.append(f'## {dim}')
        lines.append('')
        lines.append(f'| Comparison ({args.ref} vs) | Δ(logit) | OR | SE(cluster) | z | p |')
        lines.append('|---|---:|---:|---:|---:|---:|')
        for other, diff, se, z, p in sorted(report_rows, key=lambda r: r[4]):
            lines.append(f'| {other} | {diff:+.3f} | {np.exp(diff):.3f} | {se:.3f} | {z:.2f} | {p:.4f} |')
        lines.append('')

    report = '\n'.join(lines) + '\n'
    (out / 'ordinal_report.md').write_text(report, encoding='utf-8')
    (out / 'ordinal_results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(report)
    print('Output ->', out / 'ordinal_report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
