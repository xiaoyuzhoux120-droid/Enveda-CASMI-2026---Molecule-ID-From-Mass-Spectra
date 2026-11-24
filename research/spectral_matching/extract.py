"""Extract the evaluation subsets used by the spectral-matching study.

Outputs (in --out): spectra.parquet (one row per kept spectrum with peaks) and roles.json.
Roles:
  analog_query   : enveda-180 molecules used as queries (their structure is held out of the analog library)
  analog_library : other enveda-180 molecules (the analog library)
  cross_query    : enveda-180 molecules that also have non-timsTOF spectra (cross-instrument identity task)
Library rows for the identity task are all non-timsTOF spectra whose adduct matches and whose precursor
lies within 10 ppm + 0.002 Da of a cross_query timsTOF precursor.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa

ap = argparse.ArgumentParser()
ap.add_argument('--train', default='data/train.parquet')
ap.add_argument('--meta', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--n-query', type=int, default=2000)
ap.add_argument('--n-lib', type=int, default=40000)
ap.add_argument('--seed', type=int, default=20261124)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
meta = pd.read_parquet(a.meta)
rng = np.random.default_rng(a.seed)
env = meta[meta.ingest_lib == 'enveda-180']
nontims = meta[meta.instrument_type != 'timsTOF']
cross = sorted(set(env.inchikey14) & set(nontims.inchikey14))
pool = np.array(sorted(set(env.inchikey14) - set(cross)))
rng.shuffle(pool)
aq = set(pool[:a.n_query]); al = set(pool[a.n_query:a.n_query + a.n_lib])
cq = set(cross)
# precursor windows for identity-task library rows (per adduct)
cqrows = env[env.inchikey14.isin(cq)]
win = {ad: np.sort(g.precursor_mz.to_numpy()) for ad, g in cqrows.groupby('adduct')}
cols = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'molecular_formula', 'instrument_type', 'adduct',
        'ionization_mode', 'precursor_mz', 'collision_energy_orig', 'ms2_mzs', 'ms2_normalized_intensities']
pf = pq.ParquetFile(a.train)
parts = []
for bi, b in enumerate(pf.iter_batches(batch_size=50000, columns=cols)):
    d = b.select(cols[:9]).to_pandas()
    lib = d.ingest_lib.to_numpy(); ik = d.inchikey14.to_numpy(); inst = d.instrument_type.to_numpy()
    is_env = lib == 'enveda-180'
    keep = is_env & (d.inchikey14.isin(aq | al | cq).to_numpy())
    near = np.zeros(len(d), bool)
    nt = np.flatnonzero(inst != 'timsTOF')
    for i in nt:
        w = win.get(d.adduct.iat[i])
        if w is None: continue
        p = d.precursor_mz.iat[i]
        if not np.isfinite(p): continue
        tol = p * 10e-6 + 0.002
        j = np.searchsorted(w, p - tol)
        if j < len(w) and w[j] <= p + tol: near[i] = True
    keep |= near
    idx = np.flatnonzero(keep)
    if len(idx):
        parts.append(b.take(pa.array(idx)))
    if bi % 10 == 0:
        print(bi, sum(len(p) for p in parts), flush=True)
tab = pa.concat_tables([pa.Table.from_batches([p]) for p in parts])
pq.write_table(tab, out / 'spectra.parquet')
json.dump(dict(analog_query=sorted(aq), analog_library=sorted(al), cross_query=sorted(cq)), open(out / 'roles.json', 'w'))
print('rows', tab.num_rows)
