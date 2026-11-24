"""Replay the analog layer on the public 250-molecule natural-product panel
(dmitriigluzdov/casmi26-natural-product-panel-benchmark, main_candidates.parquet: the public pipeline's ranked
main lists in three scenarios). Queries are the panel's enveda-np-examples spectra from train.parquet; every
spectrum of every panel structure is removed from the analog library. Candidates from pool_source 0 have no
structure in the mapped sources and get evidence NaN for truth and decoys alike (no truth bias).
"""
import sys, json, pickle, argparse, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
import analog_layer as al

ap = argparse.ArgumentParser()
ap.add_argument('--train', default='data/train.parquet'); ap.add_argument('--panel', required=True)
ap.add_argument('--ik2smi', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
m = pd.read_parquet(Path(a.panel) / 'main_candidates.parquet'); m['molecule'] = m.molecule.astype(str); m['scenario'] = m.scenario.astype(str)
mols = sorted(m.molecule.unique())
ik2smi = pickle.load(open(a.ik2smi, 'rb'))
t0 = time.time()
lib = al.build_library(a.train, exclude=frozenset(mols))
print('library built', round(time.time() - t0), 's', flush=True)
cols = ['ingest_lib', 'inchikey14', 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct']
q = pq.read_table(a.train, columns=cols, filters=[('ingest_lib', '=', 'enveda-np-examples')]).to_pandas()
q = q[q.inchikey14.isin(set(mols))]
AN = {}
t0 = time.time()
for mol, g in q.groupby('inchikey14'):
    AN[mol] = al.analogs(lib, g)
print('analogs', round(time.time() - t0), 's; molecules with analogs', sum(1 for v in AN.values() if v), flush=True)
res = {}
LAMS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
per = {}
for sc, g in m.groupby('scenario'):
    for mol in mols:
        gg = g[(g.molecule == mol) & (g['rank'] <= 25)].sort_values('rank')
        smi = [ik2smi.get(k) if s != 0 else None for k, s in zip(gg.candidate_key, gg.pool_source)]
        ev = al.evidence(lib, AN.get(mol, []), smi)
        truth = gg.is_truth.to_numpy()
        for lam in LAMS:
            order = al.rerank(list(range(len(gg))), ev, lam)
            t = np.flatnonzero(truth[order])
            per.setdefault((sc, lam), []).append(1.0 / (t[0] + 1) if len(t) else 0.0)
rng = np.random.default_rng(0); B = rng.integers(0, len(mols), (5000, len(mols)))
for sc in sorted(m.scenario.unique()):
    base = np.array(per[(sc, 0.0)]); res[sc] = {}
    for lam in LAMS:
        v = np.array(per[(sc, lam)]); d = v - base
        res[sc][str(lam)] = dict(mrr=float(v.mean()), top1=float((v == 1).mean()), d=float(d.mean()),
                                 ci95=[float(np.quantile(d[B].mean(1), .025)), float(np.quantile(d[B].mean(1), .975))],
                                 up=int((d > 0).sum()), down=int((d < 0).sum()))
        print(sc, lam, json.dumps(res[sc][str(lam)]), flush=True)
json.dump(res, open(a.out, 'w'), indent=1)
