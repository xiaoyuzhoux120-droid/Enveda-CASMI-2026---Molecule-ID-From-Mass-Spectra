"""Pair-level evaluation: how well does a spectral score separate structurally related pairs?

Uses the Task B construction (held-out enveda-180 queries vs 40k-molecule library, same polarity,
|delta neutral mass| <= 200 Da). For every pair: score, matched peaks, Morgan-2 Tanimoto.
Reports ROC-AUC for Tanimoto >= 0.6 ("related", as in Bittremieux et al. 2022 style evaluations), and for
the neighbour-network rule (score >= tau and matches >= 6): number of edges per query and edge precision.
"""
import sys, json, argparse, re
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import smlib
import bench_common as bc

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--n-query', type=int, default=600)
a = ap.parse_args()
q, lib, qfp, lfp, pq_, pr_, tan = bc.build_task_b(a.data, a.n_query, 40000)
qrows = list(zip(q.ms2_mzs, q.ms2_normalized_intensities, q.precursor_mz))
rrows = list(zip(lib.ms2_mzs, lib.ms2_normalized_intensities, lib.precursor_mz))
VAR = {
    'network_modcos_0.0101_rmprec_top120': dict(metric=0, power=0.5, ent_weight=False, floor=0.0, topk=120, rm_prec=True, tol_da=0.0101, tol_ppm=0, shift=2, round2=True),
    'modcos_10ppm+0.002_rmprec_top120': dict(metric=0, power=0.5, ent_weight=False, floor=0.0, topk=120, rm_prec=True, tol_da=0.002, tol_ppm=10, shift=2, round2=False),
    'cos_0.0101_rmprec_top120': dict(metric=0, power=0.5, ent_weight=False, floor=0.0, topk=120, rm_prec=True, tol_da=0.0101, tol_ppm=0, shift=0, round2=True),
    'pipeline_entropy_maxshift': dict(metric=1, power=1.0, ent_weight=True, floor=0.002, topk=256, rm_prec=False, tol_da=0.01, tol_ppm=0, shift=1, round2=False),
    'mod_entropy_10ppm+0.002_rmprec': dict(metric=1, power=1.0, ent_weight=True, floor=0.002, topk=256, rm_prec=True, tol_da=0.002, tol_ppm=10, shift=2, round2=False),
    'mod_entropy_0.01_rmprec': dict(metric=1, power=1.0, ent_weight=True, floor=0.002, topk=256, rm_prec=True, tol_da=0.01, tol_ppm=0, shift=2, round2=False),
}
def auc(score, y):
    o = np.argsort(score, kind='stable'); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
rel = tan >= 0.6
res = dict(_meta=dict(pairs=int(len(tan)), related_pairs=int(rel.sum()), queries=int(len(q))))
for name, cfg in VAR.items():
    def rows_(rows):
        out = []
        for mz, it, p in rows:
            mz = np.asarray(mz); it = np.asarray(it)
            if cfg['round2']:  # the network's own loader: 0.01-Da rounding, max per bin
                b, inv = np.unique(np.round(mz, 2), return_inverse=True); mx = np.zeros(len(b)); np.maximum.at(mx, inv, it); mz, it = b, mx
            out.append((mz, it, p))
        return out
    cfg2 = {**cfg, 'centroid_da': 0.0, 'min_mz': 0.0, 'nl': 0.0}
    s, n = bc.run_pairs(pq_, pr_, rows_(qrows), rows_(rrows), cfg2)
    r = dict(auc_tan_ge_0p6=auc(s, rel), spearman=float(pd.Series(s).corr(pd.Series(tan), method='spearman')))
    for tau in (0.5, 0.6, 0.7, 0.8):
        e = (s >= tau) & (n >= 6)
        r[f'tau{tau}_edges_per_query'] = float(e.sum() / len(q)); r[f'tau{tau}_precision'] = float(rel[e].mean()) if e.any() else None
        r[f'tau{tau}_mean_tan'] = float(tan[e].mean()) if e.any() else None
    res[name] = r; print(name, json.dumps(r), flush=True)
json.dump(res, open(a.out, 'w'), indent=1)
