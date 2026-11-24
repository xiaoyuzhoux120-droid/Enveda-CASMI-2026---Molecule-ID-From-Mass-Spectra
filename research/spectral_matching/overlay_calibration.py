"""Rank-1 overlay calibration (Task A data): promote the best external-library hit when score >= tau.

For each cross-instrument molecule (timsTOF queries vs non-timsTOF library, same adduct, precursor within
10 ppm + 0.002 Da, >= 4 matched peaks as in the 0.495 overlay):
  known  : truth present in the library -> a promotion is correct only if the best hit is the truth
  novel  : truth's spectra removed (simulates a structure absent from the external library) -> any
           promotion is wrong. Hidden-test molecules are mostly novel, so this false-promotion rate matters.
For each metric reports, per tau: correct-promotion rate (known), false-promotion rate (known, wrong hit)
and false-promotion rate (novel).
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import bench_common as bc

ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
data = Path(a.data)
sp = pd.read_parquet(data / 'spectra.parquet'); roles = json.load(open(data / 'roles.json'))
cq = set(roles['cross_query'])
q = sp[(sp.ingest_lib == 'enveda-180') & sp.inchikey14.isin(cq)].reset_index(drop=True)
lib = sp[(sp.instrument_type != 'timsTOF')].reset_index(drop=True); lib = lib[lib.ms2_mzs.map(len) > 0].reset_index(drop=True)
idx = {}
for ad, g in lib.groupby('adduct'):
    o = np.argsort(g.precursor_mz.to_numpy()); idx[ad] = (g.precursor_mz.to_numpy()[o], g.index.to_numpy()[o])
pqs, prs = [], []
for i, r in q.iterrows():
    if r.adduct not in idx: continue
    pm, ix = idx[r.adduct]; tol = r.precursor_mz * 10e-6 + 0.002
    lo, hi = np.searchsorted(pm, r.precursor_mz - tol), np.searchsorted(pm, r.precursor_mz + tol, 'right')
    pqs += [i] * (hi - lo); prs += list(ix[lo:hi])
pq_ = np.array(pqs, np.int64); pr_ = np.array(prs, np.int64)
qik = q.inchikey14.to_numpy()[pq_]; rik = lib.inchikey14.to_numpy()[pr_]
qrows = list(zip(q.ms2_mzs, q.ms2_normalized_intensities, q.precursor_mz))
rrows = list(zip(lib.ms2_mzs, lib.ms2_normalized_intensities, lib.precursor_mz))
BASE = dict(metric=1, power=1.0, ent_weight=True, floor=0.002, topk=256, rm_prec=False, tol_da=0.01, tol_ppm=0.0,
            shift=0, centroid_da=0.0, min_mz=0.0, nl=0.0)
V = {'overlay_cos_sqrt_0.015 (current)': dict(metric=0, power=0.5, ent_weight=False, floor=0.0, topk=0, tol_da=0.015),
     'entropy_weighted_0.01': {}, 'entropy_weighted_10ppm+0.002': dict(tol_da=0.002, tol_ppm=10.0)}
mols = sorted(set(qik))
res = {}
for name, over in V.items():
    s, n = bc.run_pairs(pq_, pr_, qrows, rrows, {**BASE, **over})
    df = pd.DataFrame(dict(m=qik, c=rik, s=np.where(n >= 4, s, 0.0)))
    best = df.groupby(['m', 'c']).s.max().reset_index()
    known = best.sort_values('s', ascending=False).drop_duplicates('m').set_index('m')
    novel = best[best.m != best.c].sort_values('s', ascending=False).drop_duplicates('m').set_index('m')
    has_truth = set(best[best.m == best.c].m)
    kn = known.reindex(sorted(has_truth)); nv = novel.reindex(mols)
    nv_s = nv.s.fillna(0).to_numpy()
    curve = {}
    for tau in np.round(np.arange(0.5, 0.96, 0.05), 2):
        prom = kn.s >= tau
        curve[str(tau)] = dict(known_correct=float((prom & (kn.index == kn.c)).mean()),
                               known_wrong=float((prom & (kn.index != kn.c)).mean()),
                               novel_false=float((nv_s >= tau).mean()))
    res[name] = dict(n_known=len(kn), n_novel=len(nv), curve=curve)
    print(name, json.dumps(curve['0.8']), flush=True)
json.dump(res, open(a.out, 'w'), indent=1)
