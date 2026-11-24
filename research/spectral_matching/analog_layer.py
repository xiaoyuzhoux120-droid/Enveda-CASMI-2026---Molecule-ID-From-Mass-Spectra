"""Training-free analog-evidence re-ranking layer (ranks 2-25), built on the spectral-matching study.

For each test molecule:
  1. every query spectrum is compared with one representative spectrum per (structure, polarity) of
     train.parquet whose neutral mass is within +-200 Da, using one-to-one modified (hybrid) weighted
     entropy, peaks >= precursor - 1.5 removed, tolerance 0.002 Da + 10 ppm (REPORT.md Task B);
  2. the top-80 analogs (max over query spectra) give each candidate an evidence score
     sum(sim^3 * Tanimoto(candidate, analog)) / sum(sim^3)   (Morgan r2, 2048 bits; Task C);
  3. ranks 2..25 are re-sorted by z(-log rank) + lam * z(evidence); rank 1 never moves; candidates without
     a parsable structure get evidence z = 0.
Self-contained: numpy, pandas, pyarrow, numba, rdkit.
"""
import math, re, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
from numba import njit, prange

TOL_DA, TOL_PPM, FLOOR, TOPK, WINDOW, N_ANALOG, P_SIM = 0.002, 10.0, 0.002, 256, 200.0, 80, 3.0
_LN4 = math.log(4.0)
MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163, S=31.97207100, F=18.99840322,
            Cl=34.96885268, Br=78.9183371, I=126.904473, Na=22.9897692809, K=38.96370668, Si=27.9769265325,
            B=11.0093054, Se=79.9165213)
E = 0.00054857990; PROTON = MASS['H'] - E; H2O = 2 * MASS['H'] + MASS['O']
ADDUCT = {"[M+H]+": PROTON, "[M+NH4]+": MASS['N'] + 4 * MASS['H'] - E, "[M+Na]+": MASS['Na'] - E,
          "[M+K]+": MASS['K'] - E, "[M-H2O+H]+": PROTON - H2O, "[M-H]-": -PROTON, "[M-H2O-H]-": -PROTON - H2O,
          "[M+CH2O2-H]-": MASS['C'] + 2 * MASS['H'] + 2 * MASS['O'] - PROTON, "[M+Cl]-": MASS['Cl'] + E,
          "[M+C2H4O2-H]-": 2 * MASS['C'] + 4 * MASS['H'] + 2 * MASS['O'] - PROTON}


def formula_mass(f):
    if not isinstance(f, str):
        return np.nan
    try:
        return sum(MASS[e] * int(n or 1) for e, n in re.findall(r'([A-Z][a-z]?)(\d*)', f))
    except KeyError:
        return np.nan


def prep(mz, it, prec):
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    k = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz < prec - 1.5)
    mz, it = mz[k], it[k]
    if len(mz) == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    k = it >= FLOOR * it.max(); mz, it = mz[k], it[k]
    if len(mz) > TOPK:
        k = np.sort(np.argsort(-it, kind='stable')[:TOPK]); mz, it = mz[k], it[k]
    o = np.argsort(mz, kind='stable'); mz, it = mz[o], it[o]
    it = it / it.sum()
    S = -np.sum(it * np.log(it))
    if S < 3.0:
        it = it ** (0.25 + 0.25 * S); it = it / it.sum()
    return mz.astype(np.float32), it.astype(np.float32)


@njit(cache=True)
def _collect(qm, qp, rm, rp, shift, bs, bi, bj, nb):
    m = len(rm); j0 = 0
    for i in range(len(qm)):
        t = qm[i] - shift; tol = TOL_DA + TOL_PPM * 1e-6 * qm[i]
        while j0 < m and rm[j0] < t - tol:
            j0 += 1
        j = j0
        while j < m and rm[j] <= t + tol:
            if nb < len(bs):
                a = qp[i]; b = rp[j]; s = a + b
                bs[nb] = (s * np.log(s) - a * np.log(a) - b * np.log(b)) / _LN4; bi[nb] = i; bj[nb] = j; nb += 1
            j += 1
    return nb


@njit(cache=True)
def _sim(qm, qp, rm, rp, shift):
    cap = 4 * (len(qm) + len(rm)) + 64
    bs = np.empty(cap, np.float64); bi = np.empty(cap, np.int64); bj = np.empty(cap, np.int64)
    nb = _collect(qm, qp, rm, rp, 0.0, bs, bi, bj, 0)
    if abs(shift) >= 1e-3:
        nb = _collect(qm, qp, rm, rp, shift, bs, bi, bj, nb)
    if nb == 0:
        return 0.0
    order = np.argsort(-bs[:nb]); uq = np.zeros(len(qm), np.bool_); ur = np.zeros(len(rm), np.bool_); tot = 0.0
    for k in order:
        i = bi[k]; j = bj[k]
        if uq[i] or ur[j]:
            continue
        uq[i] = True; ur[j] = True; tot += bs[k]
    return tot


@njit(cache=True, parallel=True)
def _search(qm, qp, qprec, idx, roff, rmz, rit, rprec):
    out = np.zeros(len(idx), np.float32)
    for k in prange(len(idx)):
        r = idx[k]; a = roff[r]; b = roff[r + 1]
        if b > a:
            out[k] = _sim(qm, qp, rmz[a:b], rit[a:b], qprec - rprec[r])
    return out


def build_library(train_path, exclude=frozenset(), log=print):
    """One representative spectrum per (inchikey14, polarity): enveda merged-CE first, then [M+H]+/[M-H]-, then most peaks."""
    cols = ['inchikey14', 'normalized_smiles', 'molecular_formula', 'adduct', 'precursor_mz', 'collision_energy_orig',
            'ingest_lib', 'num_peaks']
    meta = pq.read_table(train_path, columns=cols).to_pandas()
    meta['row'] = np.arange(len(meta))
    meta = meta[~meta.inchikey14.isin(exclude) & meta.adduct.isin(ADDUCT) & meta.precursor_mz.notna()]
    meta['pol'] = np.where(meta.adduct.str.endswith('+'), 1, -1)
    meta['pri'] = ((meta.ingest_lib == 'enveda-180') & meta.collision_energy_orig.astype(str).str.contains(',')).astype(int) * 4 \
        + meta.adduct.isin(['[M+H]+', '[M-H]-']).astype(int) * 2
    meta = meta.sort_values(['inchikey14', 'pol', 'pri', 'num_peaks'], ascending=[True, True, False, False], kind='stable')
    reps = meta.drop_duplicates(['inchikey14', 'pol']).sort_values('row').reset_index(drop=True)
    reps['nm'] = reps.molecular_formula.map(formula_mass)
    reps = reps[np.isfinite(reps.nm)].reset_index(drop=True)
    want = set(reps.row.tolist()); peaks = {}
    pf = pq.ParquetFile(train_path); start = 0
    for b in pf.iter_batches(batch_size=65536, columns=['ms2_mzs', 'ms2_normalized_intensities']):
        n = b.num_rows
        sel = [i for i in range(start, start + n) if i in want]
        if sel:
            mz = b.column(0); it = b.column(1)
            for i in sel:
                peaks[i] = (mz[i - start].values.to_numpy(zero_copy_only=False), it[i - start].values.to_numpy(zero_copy_only=False))
        start += n
    specs = [prep(*peaks[r], p) for r, p in zip(reps.row, reps.precursor_mz)]
    lens = np.array([len(s[0]) for s in specs], np.int64)
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    lib = dict(off=off, mz=np.concatenate([s[0] for s in specs]), it=np.concatenate([s[1] for s in specs]),
               prec=reps.precursor_mz.to_numpy(np.float64), nm=reps.nm.to_numpy(np.float64), pol=reps.pol.to_numpy(),
               ik=reps.inchikey14.to_numpy(), smi=reps.normalized_smiles.to_numpy())
    log(f'[analog] library reps {len(reps):,}, peaks {off[-1]:,}')
    return lib


def analogs(lib, group):
    """group: rows of one test molecule (ms2_mzs, ms2_normalized_intensities, precursor_mz, adduct). Returns [(lib index, sim)]."""
    best = {}
    for r in group.itertuples():
        if r.adduct not in ADDUCT or not np.isfinite(r.precursor_mz):
            continue
        qm, qp = prep(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
        if len(qm) == 0:
            continue
        pol = 1 if r.adduct.endswith('+') else -1; nm = r.precursor_mz - ADDUCT[r.adduct]
        idx = np.flatnonzero((lib['pol'] == pol) & (np.abs(lib['nm'] - nm) <= WINDOW)).astype(np.int64)
        if len(idx) == 0:
            continue
        s = _search(qm, qp, float(r.precursor_mz), idx, lib['off'], lib['mz'], lib['it'], lib['prec'])
        top = np.argsort(-s, kind='stable')[:4 * N_ANALOG]
        for k in top:
            ik = lib['ik'][idx[k]]
            if s[k] > best.get(ik, (0.0, -1))[0]:
                best[ik] = (float(s[k]), int(idx[k]))
    out = sorted(best.values(), key=lambda x: (-x[0], x[1]))[:N_ANALOG]
    return [(i, s) for s, i in out]


_FPC = {}


def fingerprint(smi):
    if smi in _FPC:
        return _FPC[smi]
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    if 'gen' not in _FPC:
        _FPC['gen'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    v = None if m is None else _FPC['gen'].GetFingerprintAsNumPy(m).astype(np.float32)
    _FPC[smi] = v
    return v


def evidence(lib, an, cand_smiles):
    """Analog evidence per candidate (NaN where the structure does not parse or no analogs)."""
    out = np.full(len(cand_smiles), np.nan)
    if not an:
        return out
    A = np.stack([fingerprint(lib['smi'][i]) if fingerprint(lib['smi'][i]) is not None else np.zeros(2048, np.float32) for i, _ in an])
    w = np.maximum(np.array([s for _, s in an]), 0) ** P_SIM
    if w.sum() <= 0:
        return out
    asum = A.sum(1)
    for k, smi in enumerate(cand_smiles):
        f = fingerprint(smi)
        if f is None:
            continue
        inter = A @ f; T = inter / np.maximum(asum + f.sum() - inter, 1)
        out[k] = float(T @ w / w.sum())
    return out


def rerank(cands, ev, lam, keep_top=1, depth=25):
    """cands: ranked list; ev: evidence aligned with cands. Returns new order (indices)."""
    n = min(len(cands), depth)
    if n <= keep_top + 1 or lam <= 0 or np.all(~np.isfinite(ev[:n])):
        return list(range(len(cands)))
    head = list(range(keep_top)); idx = np.arange(keep_top, n)
    base = -np.log(idx + 1.0)
    def z(x):
        x = np.asarray(x, float); m = np.isfinite(x)
        out = np.zeros(len(x))
        if m.sum() >= 2 and np.std(x[m]) > 1e-12:
            out[m] = (x[m] - x[m].mean()) / x[m].std()
        return out
    key = z(base) + lam * z(ev[idx])
    order = idx[np.lexsort((idx, -key))]
    return head + order.tolist() + list(range(n, len(cands)))
