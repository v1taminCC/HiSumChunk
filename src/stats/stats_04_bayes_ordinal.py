# -*- coding: utf-8 -*-
"""Bayesian hierarchical ordinal model (PyMC): question/book random intercepts + method fixed effects.

Output:
  - P(baseline > each scheme): posterior proportion of alpha_ours > alpha_other
  - Mean Δ(logit) and 95% HDI (arviz.hdi)
  - Convergence diagnostics: max R-hat, min ESS (bulk/tail)
  - Metadata: PyMC/NumPy/Python versions, random seed, md5 of the input CSV (reproducibility)

Supports per-dimension persistence and checkpoint resume (--force recomputes).

Usage:
  python scripts/stats_04_bayes_ordinal.py [--input ../data/scores_main.csv] [--out ./out]
                                           [--baseline ours] [--draws 1000] [--tune 1000]
                                           [--dims grounding,correctness,hallucination_absence] [--force]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

DIMS = ['grounding', 'correctness', 'hallucination_absence']
ORDER = ['ours', 'fixed', 'lumberchunker', 'lbdsemantic', 'semanticembed', 'raptor']


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path, drop_tied: bool = False) -> dict[str, list[tuple]]:
    by_dim = defaultdict(list)
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('scheme') == 'booksum_split':
                continue
            for d in DIMS:
                try:
                    y = int(row[d])
                except (KeyError, ValueError):
                    continue
                if y <= 0:
                    continue
                by_dim[d].append((row['scheme'], row['query_id'], row['book'], y))
    if drop_tied:
        for d in list(by_dim):
            per_q: dict[str, list[int]] = defaultdict(list)
            for _, qid, _, y in by_dim[d]:
                per_q[qid].append(y)
            keep = {q for q, ys in per_q.items() if len(set(ys)) > 1}
            by_dim[d] = [r for r in by_dim[d] if r[1] in keep]
    return by_dim


def fit_dim(rows: list[tuple], baseline: str, draws: int, tune: int, seed: int,
            target_accept: float = 0.9):
    schemes = [s for s in ORDER if s in {r[0] for r in rows}]
    items = sorted({r[1] for r in rows})
    books = sorted({r[2] for r in rows})
    s_idx = {s: i for i, s in enumerate(schemes)}
    i_idx = {q: i for i, q in enumerate(items)}
    b_idx = {b: i for i, b in enumerate(books)}

    m_idx = np.array([s_idx[r[0]] for r in rows])
    it_idx = np.array([i_idx[r[1]] for r in rows])
    bk_idx = np.array([b_idx[r[2]] for r in rows])
    # PyMC OrderedLogistic requires the dependent variable coded 0..K-1 (raw scores are 1..5, hence minus 1)
    y = np.array([r[3] for r in rows]) - 1

    with pm.Model() as model:
        alpha = pm.Normal('alpha', 0.0, 1.0, shape=len(schemes))
        z_item = pm.Normal('z_item', 0.0, 1.0, shape=len(items))
        sig_item = pm.HalfNormal('sig_item', 1.0)
        z_book = pm.Normal('z_book', 0.0, 1.0, shape=len(books))
        sig_book = pm.HalfNormal('sig_book', 1.0)
        eta = alpha[m_idx] + sig_item * z_item[it_idx] + sig_book * z_book[bk_idx]
        cut_raw = pm.Normal('cut_raw', 0.0, 1.5, shape=4)
        cut = pm.Deterministic('cut', pt.sort(cut_raw))
        pm.OrderedLogistic('y', eta=eta, cutpoints=cut, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=4, cores=4,
                          target_accept=target_accept, random_seed=seed, progressbar=False)

    diagnostics: dict[str, float] = {}
    try:
        rhat_all = az.rhat(idata)
        ess_all = az.ess(idata)
        diagnostics['rhat_max_all'] = float(max(float(rhat_all[v].values.max()) for v in rhat_all.data_vars))
        diagnostics['ess_min_all'] = float(min(float(ess_all[v].values.min()) for v in ess_all.data_vars))
    except Exception as exc:  # noqa: BLE001
        diagnostics['diag_error_all'] = str(exc)[:200]
    try:
        key_vars = ['alpha', 'cut', 'sig_item', 'sig_book']
        rhat_k = az.rhat(idata, var_names=key_vars)
        ess_k = az.ess(idata, var_names=key_vars)
        diagnostics['rhat_max_key'] = float(max(float(rhat_k[v].values.max()) for v in rhat_k.data_vars))
        diagnostics['ess_min_key'] = float(min(float(ess_k[v].values.min()) for v in ess_k.data_vars))
    except Exception as exc:  # noqa: BLE001
        diagnostics['diag_error_key'] = str(exc)[:200]
    try:
        diagnostics['divergences'] = int(idata.sample_stats['diverging'].sum())
    except Exception:  # noqa: BLE001
        diagnostics['divergences'] = -1

    post = idata.posterior['alpha'].values.reshape(-1, len(schemes))
    base_i = s_idx[baseline]
    out = {'schemes': schemes, 'n': len(rows), 'baseline': baseline,
           'pairwise': {}, 'target_accept': target_accept, **diagnostics}
    for s in schemes:
        if s == baseline:
            continue
        diff = post[:, base_i] - post[:, s_idx[s]]
        try:
            hdi = az.hdi(diff, prob=0.95)
        except TypeError:  # ArviZ < 1.0 compatibility
            hdi = az.hdi(diff, hdi_prob=0.95)
        out['pairwise'][s] = {
            'P_baseline_gt_other': float((diff > 0).mean()),
            'mean_diff_logit': float(diff.mean()),
            'hdi95': [float(hdi[0]), float(hdi[1])],
            'q025': float(np.percentile(diff, 2.5)),
            'q975': float(np.percentile(diff, 97.5)),
        }
    out['latent_means'] = {s: float(post.mean(axis=0)[s_idx[s]]) for s in schemes}
    return out


def build_report(results: dict, baseline: str, input_path: str, draws: int, tune: int,
                 input_md5: str = '', seed: int = 2026, drop_tied: bool = False) -> str:
    lines = ['# Bayesian hierarchical ordinal model (PyMC)', '']
    lines.append('Model: ordinal logistic; method fixed effects + question/book random intercepts (non-centered parameterization)')
    lines.append(f'Sampling: 4 chains x (tune={tune} + draws={draws}) | seed={seed} | target_accept=0.9 initial')
    lines.append(f'Input: {input_path} | md5={input_md5}')
    lines.append(f'Environment: Python {platform.python_version()} | PyMC {pm.__version__} | ArviZ {az.__version__} | NumPy {np.__version__}')
    lines.append(f'All-tied items: {"dropped (uninformative for method comparison)" if drop_tied else "kept"}')
    lines.append('')
    for dim in DIMS:
        if dim not in results:
            lines.append(f'## {dim} (incomplete)')
            lines.append('')
            continue
        res = results[dim]
        if 'error' in res:
            lines.append(f'## {dim} (failed)')
            lines.append('')
            lines.append(f'Error: {res["error"]}')
            lines.append('')
            continue
        lines.append(f'## {dim} (n={res["n"]})')
        lines.append('')
        lines.append(f'Convergence (key parameters alpha/cut/sigmas): max R-hat={res.get("rhat_max_key", "?")} | '
                     f'min ESS={res.get("ess_min_key", "?")}')
        lines.append(f'Convergence (all parameters incl. item intercepts): max R-hat={res.get("rhat_max_all", "?")} | '
                     f'min ESS={res.get("ess_min_all", "?")} | divergences={res.get("divergences", "?")} | '
                     f'target_accept={res.get("target_accept", "?")}')
        lines.append('')
        lines.append('| Comparison | P(baseline>scheme) | mean Δ(logit) | 95% HDI |')
        lines.append('|---|---:|---:|---|')
        for s, v in sorted(res['pairwise'].items(), key=lambda kv: -kv[1]['P_baseline_gt_other']):
            lines.append(f"| {baseline} vs {s} | {v['P_baseline_gt_other']:.3f} | "
                         f"{v['mean_diff_logit']:+.3f} | [{v['hdi95'][0]:+.3f}, {v['hdi95'][1]:+.3f}] |")
        lines.append('')
    lines.append('Note: intervals are 95% highest-density intervals (arviz.hdi); no multiple-comparison correction; a robustness complement to the frequentist conclusions;')
    lines.append('    The model assumes proportional odds with weakly informative priors (coefficients N(0,1), group-level SDs HalfNormal(1)).')
    return '\n'.join(lines) + '\n'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(Path(__file__).parent.parent / 'data' / 'scores_main.csv'))
    ap.add_argument('--out', default=str(Path(__file__).parent / 'out'))
    ap.add_argument('--baseline', default='ours')
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--tune', type=int, default=1000)
    ap.add_argument('--dims', default=','.join(DIMS))
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--target-accept', type=float, default=0.9)
    ap.add_argument('--drop-tied-items', action='store_true',
                    help='drop items where all six schemes share an identical score (uninformative for method comparison; improves item-intercept convergence)')
    ap.add_argument('--force', action='store_true', help='ignore completed dimensions and recompute')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    input_md5 = md5_of(input_path) if input_path.exists() else ''

    partial_json = out / 'bayes_results_partial.json'
    partial_md = out / 'bayes_report_partial.md'
    results: dict = {}
    if partial_json.exists() and not args.force:
        try:
            results = json.loads(partial_json.read_text(encoding='utf-8'))
            print(f'Detected completed dimensions: {sorted(k for k in results if not k.startswith("_"))} (resuming from checkpoint)', flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f'Failed to read the partial file (ignoring and starting fresh): {exc}', flush=True)
            results = {}

    by_dim = load(input_path, drop_tied=args.drop_tied_items)
    dims_to_run = [d for d in args.dims.split(',') if d in by_dim]
    accept_ladder = [args.target_accept, round(min(args.target_accept + 0.05, 0.99), 2), 0.99]
    for dim in dims_to_run:
        if dim in results and not args.force and 'error' not in results[dim]:
            print(f'[{dim}] already done, skipping', flush=True)
            continue
        res = None
        for attempt, ta in enumerate(dict.fromkeys(accept_ladder), start=1):
            print(f'[{dim}] sampling starts (attempt {attempt}, target_accept={ta})...', flush=True)
            try:
                res = fit_dim(by_dim[dim], args.baseline, args.draws, args.tune, args.seed, target_accept=ta)
            except Exception as exc:  # noqa: BLE001
                print(f'[{dim}] sampling attempt {attempt} failed: {exc}', flush=True)
                res = None
                continue
            div = res.get('divergences', -1)
            total_draws = args.draws * 4
            if div >= 0 and div > max(20, int(0.01 * total_draws)) and attempt < len(accept_ladder):
                print(f'[{dim}] divergences={div} too high; retrying with a larger target_accept', flush=True)
                res = None
                continue
            break

        if res is None:
            results[dim] = {'error': 'sampling failed after retries', 'schemes': [], 'pairwise': {}}
            print(f'[{dim}] sampling finally failed; recording the error and moving on', flush=True)
        else:
            results[dim] = res
            print(f'[{dim}] done (divergences={res.get("divergences")}, '
                  f'R-hat_max={res.get("rhat_max")}) and persisted', flush=True)
        partial_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        partial_md.write_text(build_report(results, args.baseline, args.input, args.draws, args.tune,
                                           input_md5, args.seed, args.drop_tied_items), encoding='utf-8')

    report = build_report(results, args.baseline, args.input, args.draws, args.tune, input_md5, args.seed, args.drop_tied_items)
    (out / 'bayes_report.md').write_text(report, encoding='utf-8')
    (out / 'bayes_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(report, flush=True)
    print('Output ->', out / 'bayes_report.md', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
