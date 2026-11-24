"""Task C: does better analog retrieval rank the true structure higher among its isomers?

Same queries/library as Task B (query structures held out of the library). Candidates for a query are all
train.parquet structures with the query's exact molecular formula (truth included), capped at 200 decoys
(random, seeded) + truth; queries with < 2 candidates are skipped. A candidate's score is the analog
evidence used by the 0.495 pipeline's rank features: sum over the top-80 spectral analogs of
sim^3 * Tanimoto(candidate, analog) / sum sim^3. Reports MRR (ties: average rank) and top-1, with a
paired bootstrap CI95 against the pipeline setting.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
import bench_common as bc
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--train', default='data/train.parquet')
ap.add_argument('--out', required=True); ap.add_argument('--n-query', type=int, default=2000)
a = ap.parse_args()
BASE = dict(metric=1, power=1.0, ent_weight=True, floor=0.002, topk=256, rm_prec=False, tol_da=0.01, tol_ppm=0.0,
            shift=1, centroid_da=0.0, min_mz=0.0, nl=0.0)
V = {'pipeline_entropy': {}, 'ent_tol_10ppm+0.002': dict(tol_da=0.002, tol_ppm=10.0), 'ent_rm_prec': dict(rm_prec=True),
     'ent_modified': dict(shift=2), 'ent_10ppm_rmprec_modified': dict(tol_da=0.002, tol_ppm=10.0, rm_prec=True, shift=2),
     'cos_sqrt_modified_0.01_rmprec': dict(metric=0, power=0.5, ent_weight=False, floor=0.0, topk=0, shift=2, rm_prec=True),
     'cos_sqrt_0.01': dict(metric=0, power=0.5, ent_weight=False, floor=0.0, topk=0, shift=0)}
q, lib, qfp, lfp, pq_, pr_, tan = bc.build_task_b(a.data, a.n_query, 40000)
st = pq.read_table(a.train, columns=['inchikey14', 'normalized_smiles', 'molecular_formula']).to_pandas().drop_duplicates('inchikey14')
by_f = st.groupby('molecular_formula')
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
rng = np.random.default_rng(11)
cands = {}
for k, r in q.iterrows():
    if r.molecular_formula not in by_f.groups: continue
    g = st.loc[by_f.groups[r.molecular_formula]]
    dec = g[g.inchikey14 != r.inchikey14]
    if len(dec) == 0: continue
    if len(dec) > 200: dec = dec.iloc[rng.choice(len(dec), 200, replace=False)]
    fps = []
    for smi in [r.normalized_smiles] + dec.normalized_smiles.tolist():
        m = Chem.MolFromSmiles(smi)
        fps.append(np.zeros(2048, bool) if m is None else gen.GetFingerprintAsNumPy(m).astype(bool))
    cands[k] = np.stack(fps)  # row 0 = truth
print('queries with isomer candidates', len(cands), 'median candidates', np.median([len(c) for c in cands.values()]), flush=True)
order_q = np.argsort(pq_, kind='stable'); pq_s, pr_s = pq_[order_q], pr_[order_q]
bounds = np.r_[0, np.cumsum(np.bincount(pq_s, minlength=len(q)))]
qrows = list(zip(q.ms2_mzs, q.ms2_normalized_intensities, q.precursor_mz))
rrows = list(zip(lib.ms2_mzs, lib.ms2_normalized_intensities, lib.precursor_mz))
res, per = {}, {}
keys = sorted(cands)
for name, over in V.items():
    cfg = {**BASE, **over}
    s, _ = bc.run_pairs(pq_s, pr_s, qrows, rrows, cfg)
    rr = []
    for k in keys:
        a_, b_ = bounds[k], bounds[k + 1]
        ss = s[a_:b_]; o = np.lexsort((pr_s[a_:b_], -ss))[:80]
        w = np.maximum(ss[o], 0) ** 3
        A = lfp[pr_s[a_:b_][o]]; C = cands[k]
        inter = C.astype(np.float32) @ A.T.astype(np.float32)
        union = C.sum(1)[:, None] + A.sum(1)[None, :] - inter
        T = inter / np.maximum(union, 1)
        sc = T @ w / max(w.sum(), 1e-12)
        rank = 1 + (sc[1:] > sc[0]).sum() + 0.5 * (sc[1:] == sc[0]).sum()
        rr.append(1.0 / rank)
    rr = np.array(rr); per[name] = rr
    res[name] = dict(mrr=float(rr.mean()), top1=float((rr == 1).mean()), n=len(rr), cfg=cfg)
    print(name, res[name]['mrr'], res[name]['top1'], flush=True)
r0 = per['pipeline_entropy']; boot = np.random.default_rng(0).integers(0, len(r0), (2000, len(r0)))
for name in per:
    d = per[name] - r0; m = d[boot].mean(1)
    res[name]['d_mrr_vs_pipeline'] = float(d.mean()); res[name]['ci95'] = [float(np.quantile(m, .025)), float(np.quantile(m, .975))]
# random-score reference: expected reciprocal rank with uninformative scores
res['_reference'] = dict(random_mrr=float(np.mean([np.mean(1.0 / np.arange(1, len(cands[k]) + 1)) for k in keys])))
json.dump(res, open(a.out, 'w'), indent=1)
