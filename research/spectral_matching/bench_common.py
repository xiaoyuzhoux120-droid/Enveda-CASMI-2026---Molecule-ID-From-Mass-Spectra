"""Shared helpers for the spectral-matching study (preprocessing, pair scoring, Task B construction)."""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import smlib

MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163, S=31.97207100, F=18.99840322,
            Cl=34.96885268, Br=78.9183371, I=126.904473, Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213)


def fmass(f):
    try:
        return sum(MASS[e] * int(n or 1) for e, n in re.findall(r'([A-Z][a-z]?)(\d*)', f))
    except KeyError:
        return np.nan


def prep_all(rows, cfg):
    return [smlib.preprocess(mz, it, p, floor=cfg['floor'], topk=cfg['topk'], rm_prec=cfg['rm_prec'], power=cfg['power'],
                             ent_weight=cfg['ent_weight'], metric=cfg['metric'], min_mz=cfg['min_mz'], centroid_da=cfg['centroid_da'])
            for mz, it, p in rows]


def nl_all(rows, prepped):
    out = []
    for (mz, it, p), (m, i) in zip(rows, prepped):
        nm = (p - m[::-1]).astype(np.float32); ni = i[::-1].copy()
        k = nm > 0
        out.append((nm[k], ni[k]))
    return out


def run_pairs(pq_, pr_, qrows, rrows, cfg):
    qs = prep_all(qrows, cfg); rs = prep_all(rrows, cfg)
    qprec = np.array([r[2] for r in qrows], np.float64); rprec = np.array([r[2] for r in rrows], np.float64)
    qo, qm, qi = smlib.pack(qs); ro, rm, ri = smlib.pack(rs)
    s, n = smlib.score_pairs(pq_, pr_, qo, qm, qi, qprec, ro, rm, ri, rprec, cfg['tol_da'], cfg['tol_ppm'], cfg['metric'], cfg['shift'])
    if cfg.get('nl', 0) > 0:
        qo, qm, qi = smlib.pack(nl_all(qrows, qs)); ro, rm, ri = smlib.pack(nl_all(rrows, rs))
        s2, _ = smlib.score_pairs(pq_, pr_, qo, qm, qi, qprec, ro, rm, ri, rprec, cfg['tol_da'], cfg['tol_ppm'], cfg['metric'], 0)
        s = (s + cfg['nl'] * s2) / (1 + cfg['nl'])
    return s, n


def build_task_b(data, n_query, n_lib, seed=7, window=200.0):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    data = Path(data)
    sp = pd.read_parquet(data / 'spectra.parquet'); roles = json.load(open(data / 'roles.json'))
    rng = np.random.default_rng(seed)
    env = sp[sp.ingest_lib == 'enveda-180']

    def reps(df):
        df = df.copy()
        df['pol'] = np.where(df.adduct.str.endswith('+'), 1, -1)
        df['pri'] = df.collision_energy_orig.str.contains(',').astype(int) * 2 + df.adduct.isin(['[M+H]+', '[M-H]-']).astype(int)
        df = df.sort_values(['inchikey14', 'pol', 'pri'], ascending=[True, True, False])
        return df.drop_duplicates(['inchikey14', 'pol']).reset_index(drop=True)
    aq = np.array(roles['analog_query']); rng.shuffle(aq); aq = set(aq[:n_query])
    al = np.array(roles['analog_library']); rng.shuffle(al); al = set(al[:n_lib])
    q = reps(env[env.inchikey14.isin(aq)])
    q = q.sort_values('pri', ascending=False).drop_duplicates('inchikey14').reset_index(drop=True)
    lib = reps(env[env.inchikey14.isin(al)])
    q['nm'] = q.molecular_formula.map(fmass); lib['nm'] = lib.molecular_formula.map(fmass)
    q = q[np.isfinite(q.nm)].reset_index(drop=True); lib = lib[np.isfinite(lib.nm)].reset_index(drop=True)

    def fp(smi):
        m = Chem.MolFromSmiles(smi)
        return np.zeros(2048, bool) if m is None else gen.GetFingerprintAsNumPy(m).astype(bool)
    allsmi = pd.concat([q[['inchikey14', 'normalized_smiles']], lib[['inchikey14', 'normalized_smiles']]]).drop_duplicates('inchikey14')
    FP = {k: fp(s) for k, s in zip(allsmi.inchikey14, allsmi.normalized_smiles)}
    qfp = np.stack([FP[k] for k in q.inchikey14]); lfp = np.stack([FP[k] for k in lib.inchikey14])
    pqs, prs = [], []
    for pol in (1, -1):
        li = lib.index[lib.pol == pol].to_numpy(); lm = lib.nm.to_numpy()[li]; o = np.argsort(lm); li, lm = li[o], lm[o]
        for qi_ in q.index[q.pol == pol]:
            lo, hi = np.searchsorted(lm, q.nm[qi_] - window), np.searchsorted(lm, q.nm[qi_] + window, 'right')
            pqs.append(np.full(hi - lo, qi_)); prs.append(li[lo:hi])
    pq_ = np.concatenate(pqs).astype(np.int64); pr_ = np.concatenate(prs).astype(np.int64)
    tan = tanimoto_pairs(np.packbits(qfp, axis=1), np.packbits(lfp, axis=1), pq_, pr_)
    return q, lib, qfp, lfp, pq_, pr_, tan


from numba import njit, prange


@njit(cache=True, parallel=True)
def tanimoto_pairs(qb, lb, pq_, pr_):
    pc = np.zeros(256, np.int64)
    for v in range(256):
        c = 0; x = v
        while x:
            c += x & 1; x >>= 1
        pc[v] = c
    out = np.zeros(len(pq_), np.float32)
    for k in prange(len(pq_)):
        a = qb[pq_[k]]; b = lb[pr_[k]]; i = 0; u = 0
        for t in range(len(a)):
            i += pc[a[t] & b[t]]; u += pc[a[t] | b[t]]
        out[k] = i / u if u > 0 else 0.0
    return out
