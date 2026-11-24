"""Summarise rank_results.json: expected MRR with random tie-breaking, Top-1, paired bootstrap CIs, and
metfrag + w * nn combinations with w selected on half A of 'eval' only.

Usage: python analyze.py WORK_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def exp_rr(scores, t=0):
    """Expected reciprocal rank of candidate t when ties are broken uniformly at random."""
    s = np.asarray(scores, float)
    s = np.where(np.isfinite(s), s, -1e30)
    better = int((s > s[t]).sum())
    ties = int((s == s[t]).sum())  # includes t itself
    return float(np.mean([1.0 / (better + j) for j in range(1, ties + 1)])), float(better == 0) / ties


def z(v):
    v = np.asarray(v, float)
    fin = np.isfinite(v)
    if not fin.any():
        return np.zeros_like(v)
    v = np.where(fin, v, v[fin].min() - 1)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)


def boot_ci(a, b, n=2000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def half(key):
    return 'A' if hashlib.sha256(key.encode()).digest()[0] % 2 == 0 else 'B'


def main(work):
    work = Path(work)
    R = json.load(open(work / 'rank_results.json'))
    methods = list(R[0]['scores'])
    nn = [m for m in methods if m.startswith('nn_')]
    W = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    rows = []
    for r in R:
        rec = dict(key=r['key'], split=r['split'], n=r['n'], half=half(r['key']) if r['split'] == 'eval' else 'np')
        for m in methods:
            rec[m], rec[m + '@1'] = exp_rr(r['scores'][m])
        mf = z(r['scores']['metfrag_lite'])
        for m in nn + ['rr8_losses']:
            zm = z(r['scores'][m])
            for w in W:
                rec[f'mf+{w}*{m}'], _ = exp_rr(mf + w * zm)
        # three-way: metfrag + best chem prior + morgan
        rows.append(rec)
    df = pd.DataFrame(rows)
    out = {}
    groups = {'eval_all': df.split == 'eval', 'eval_B': df.half == 'B', 'np_examples': df.split == 'np'}
    summary = []
    for g, msk in groups.items():
        d = df[msk]
        line = dict(set=g, panels=len(d), median_candidates=float(d.n.median()))
        for m in methods:
            line[m] = round(d[m].mean(), 4)
            line[m + '@1'] = round(d[m + '@1'].mean(), 4)
        summary.append(line)
    S = pd.DataFrame(summary).set_index('set').T
    print(S.to_string())
    out['summary'] = S.reset_index().to_dict('records')
    # choose w on half A only
    A, B, NP = df[df.half == 'A'], df[df.half == 'B'], df[df.split == 'np']
    combos = []
    for m in nn + ['rr8_losses']:
        best = max(W, key=lambda w: A[f'mf+{w}*{m}'].mean())
        col = f'mf+{best}*{m}'
        cB = boot_ci(B[col], B['metfrag_lite'])
        cN = boot_ci(NP[col], NP['metfrag_lite'])
        alone_B = boot_ci(B[m], B['metfrag_lite'])
        combos.append(dict(prior=m, w_selected_on_A=best, MRR_A=round(A[col].mean(), 4),
                           MRR_B=round(B[col].mean(), 4), MRR_B_metfrag=round(B['metfrag_lite'].mean(), 4),
                           dB=round(cB[0], 4), dB_ci=[round(cB[1], 4), round(cB[2], 4)],
                           MRR_np=round(NP[col].mean(), 4), MRR_np_metfrag=round(NP['metfrag_lite'].mean(), 4),
                           dnp=round(cN[0], 4), dnp_ci=[round(cN[1], 4), round(cN[2], 4)],
                           alone_vs_metfrag_B=[round(x, 4) for x in alone_B]))
    C = pd.DataFrame(combos)
    print(C.to_string())
    out['combos'] = combos
    # feature ablation: hr vs int (paired), chem vs morgan
    abl = []
    for a, b in (('nn_hr_chem', 'nn_int_chem'), ('nn_hr_morgan', 'nn_int_morgan'),
                 ('nn_hr_chem', 'nn_hr_morgan'), ('nn_int_chem', 'nn_int_morgan')):
        for g, msk in groups.items():
            d = df[msk]
            m, lo, hi = boot_ci(d[a], d[b])
            abl.append(dict(a=a, b=b, set=g, diff=round(m, 4), ci=[round(lo, 4), round(hi, 4)]))
    print(pd.DataFrame(abl).to_string())
    out['ablation'] = abl
    # dependence on panel size
    df['size_bin'] = pd.cut(df.n, [1, 3, 6, 12, 25, 40], labels=['2-3', '4-6', '7-12', '13-25', '26-40'])
    bysize = df[df.split == 'eval'].groupby('size_bin', observed=True)[['metfrag_lite', 'nn_hr_chem', 'nn_hr_morgan', 'rr8_losses']].mean().round(4)
    bysize['panels'] = df[df.split == 'eval'].groupby('size_bin', observed=True).size()
    print(bysize.to_string())
    out['by_size'] = bysize.reset_index().astype({'size_bin': str}).to_dict('records')
    json.dump(out, open(work / 'summary.json', 'w'), indent=1)
    df.to_csv(work / 'per_panel.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1])
