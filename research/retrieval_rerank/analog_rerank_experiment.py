"""Class-2 proxy: rerank database candidates for molecules that have no reference spectrum.

Question: given a query molecule whose structure is in COCONUT but has no spectrum in the
reference library, which neighbour search and which score fusion rank the true structure best?

Protocol (all deterministic, CPU only):
* Reference library = spectra of train-split molecules only (split_key hash, as casmi_ml.data),
  at most REF_PER_MOL spectra per molecule.
* Queries = dev-split (selection) or holdout-split (confirmation) molecules whose InChIKey14 is in
  COCONUT; up to 4 spectra each, [M+H]+/[M-H]- only. The query structure therefore never has a
  reference spectrum, i.e. "class 2" (known structure, not in the spectral library).
* Candidate pool = COCONUT + train-split structures within max(15 ppm, 0.004 Da) of the median
  inferred neutral mass. No query label is used to build pools or neighbour sets.
* Metric = MRR@25 on InChIKey14 (dataset key; the official metric additionally tautomer-
  canonicalises, which we do not replicate here).

Scorers compared (each is a candidate score; higher is better):
  iso_hist   historical hybrid.py analog score, neighbours = library isomers (35 ppm window)
  ana_hist   same formula, neighbours = global analog search (fragment + neutral-loss cosine, any mass)
  frag_hist  same formula, neighbours = global fragment-only cosine (ablation of the loss channel)
  ana_knn    weighted kNN vote sum_j s_j^4 * Tanimoto(c, n_j) / sum_j s_j^4 over global analogs
  prior      COCONUT annotation level (no spectrum)
and fusions of them (reciprocal-rank fusion and z-score sums).
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseline import ADDUCT_MASS, formula_mass  # noqa: E402
from casmi_ml.data import digest, split_key  # noqa: E402

RDLogger.DisableLog('rdApp.*')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
NBINS = 12501  # 0.1 Da bins up to 1250 Da
REF_PER_MOL = 3
QUERY_PER_MOL = 4
QUERY_ADDUCTS = {'[M+H]+', '[M-H]-'}
COLS = ['inchikey14', 'normalized_smiles', 'molecular_formula', 'adduct', 'precursor_mz',
        'ms2_mzs', 'ms2_normalized_intensities', 'instrument_type']


def peaks(mzs, ints, max_peaks=60):
    mzs = np.asarray(mzs, float)
    ints = np.asarray(ints, float)
    ok = np.isfinite(mzs) & np.isfinite(ints) & (ints >= .01) & (mzs >= 40) & (mzs < 1250)
    mzs, ints = mzs[ok], ints[ok]
    if len(mzs) > max_peaks:
        top = np.argsort(-ints, kind='stable')[:max_peaks]
        mzs, ints = mzs[top], ints[top]
    return mzs, ints


def vec(bins, weights):
    """Max-pool weights per bin, L2-normalise; returns (int32 bins, float32 values)."""
    bins = np.asarray(bins, np.int64)
    weights = np.asarray(weights, np.float64)
    ok = (bins >= 0) & (bins < NBINS) & (weights > 0)
    bins, weights = bins[ok], weights[ok]
    if not len(bins):
        return np.zeros(0, np.int32), np.zeros(0, np.float32)
    order = np.lexsort((-weights, bins))
    bins, weights = bins[order], weights[order]
    first = np.r_[True, bins[1:] != bins[:-1]]
    bins, weights = bins[first], weights[first]
    weights = weights / np.sqrt((weights ** 2).sum())
    return bins.astype(np.int32), weights.astype(np.float32)


def spectrum_vectors(mzs, ints, precursor):
    """Fragment vector (as baseline.vectorize) and neutral-loss vector (precursor - fragment)."""
    mzs, ints = peaks(mzs, ints)
    w = np.sqrt(ints) * np.sqrt(mzs / 100)
    frag = vec(np.rint(mzs * 10).astype(int), w)
    loss_mz = precursor - mzs
    keep = loss_mz > 0.5
    loss = vec(np.rint(loss_mz[keep] * 10).astype(int), np.sqrt(ints[keep]))
    return frag, loss


def combine(frag, loss):
    s = np.float32(1 / math.sqrt(2))
    return np.r_[frag[0], loss[0] + NBINS].astype(np.int32), np.r_[frag[1], loss[1]] * s


def to_csr(vectors, width=NBINS):
    rows = np.concatenate([np.full(len(b), i, np.int64) for i, (b, _) in enumerate(vectors)])
    cols = np.concatenate([b for b, _ in vectors]).astype(np.int64)
    vals = np.concatenate([v for _, v in vectors]).astype(np.float32)
    idx = torch.from_numpy(np.stack([rows, cols]))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(vals),
                                   (len(vectors), width)).coalesce().to_sparse_csr()


def load(train_path, query_split, n_queries, coconut_keys, max_batches=None):
    ref_meta, ref_frag, ref_loss, ref_count = [], [], [], defaultdict(int)
    structures = {}
    query_pool = defaultdict(list)
    rng = np.random.default_rng(7)
    seen = defaultdict(int)
    split_cache = {}
    pf = pq.ParquetFile(train_path)
    for bi, batch in enumerate(pf.iter_batches(batch_size=20000, columns=COLS)):
        df = batch.to_pandas()
        for r in df.itertuples(index=False):
            key = r.inchikey14
            if not isinstance(key, str) or len(key) != 14 or r.adduct not in ADDUCT_MASS:
                continue
            sp = split_cache.get(key)
            if sp is None:
                sp = split_cache[key] = split_key(key)
            if sp == 'train':
                if key not in structures:
                    structures[key] = (r.normalized_smiles, formula_mass(r.molecular_formula))
                if ref_count[key] >= REF_PER_MOL or r.ms2_mzs is None:
                    continue
                f, l = spectrum_vectors(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
                if len(f[0]) < 3:
                    continue
                ref_count[key] += 1
                ref_meta.append((key, float(r.precursor_mz) - ADDUCT_MASS[r.adduct]))
                ref_frag.append(f)
                ref_loss.append(l)
            elif sp == query_split and key in coconut_keys and r.adduct in QUERY_ADDUCTS:
                if r.ms2_mzs is None or len(r.ms2_mzs) < 3:
                    continue
                seen[key] += 1
                item = (r.adduct, float(r.precursor_mz), np.asarray(r.ms2_mzs),
                        np.asarray(r.ms2_normalized_intensities), r.normalized_smiles,
                        r.instrument_type)
                pool = query_pool[key]
                if len(pool) < QUERY_PER_MOL:
                    pool.append(item)
                else:
                    j = int(rng.integers(seen[key]))
                    if j < QUERY_PER_MOL:
                        pool[j] = item
        if max_batches and bi + 1 >= max_batches:
            break
        if bi % 20 == 0:
            print(f'batch {bi}: refs {len(ref_meta):,} query mols {len(query_pool):,}', flush=True)
    keys = sorted(query_pool, key=digest)[:n_queries]
    return ref_meta, ref_frag, ref_loss, structures, {k: query_pool[k] for k in keys}


class FPCache:
    def __init__(self):
        self.c = {}

    def __call__(self, smi):
        if smi not in self.c:
            m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            self.c[smi] = GEN.GetFingerprint(m) if m is not None and m.GetNumAtoms() else None
        return self.c[smi]


def ranks_from(scores):
    order = np.argsort(-scores, kind='stable')
    r = np.empty(len(scores))
    r[order] = np.arange(1, len(scores) + 1)
    return r


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.parquet')
    ap.add_argument('--coconut', default='external/coconut_dl/coconut_structures.parquet')
    ap.add_argument('--split', default='dev', choices=['dev', 'holdout'])
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--k', type=int, default=25)
    ap.add_argument('--max-batches', type=int, default=None)
    ap.add_argument('--out', default='research/retrieval_rerank/results')
    args = ap.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    coco = pd.read_parquet(args.coconut)
    coco = coco[coco.inchikey.str.len() >= 14].copy()
    coco['k14'] = coco.inchikey.str[:14]
    coco = coco.sort_values('annotation_level', ascending=False).drop_duplicates('k14')
    coconut_keys = set(coco.k14)

    ref_meta, ref_frag, ref_loss, structures, queries = load(args.train, args.split, args.n,
                                                             coconut_keys, args.max_batches)
    print(f'loaded {len(ref_meta):,} reference spectra, {len(structures):,} train structures, '
          f'{len(queries)} queries in {time.time() - t0:.0f}s', flush=True)

    # Candidate structure table: COCONUT plus train structures (train wins smiles if duplicated).
    cand = coco[['k14', 'canonical_smiles', 'exact_mass', 'annotation_level']].rename(
        columns={'canonical_smiles': 'smiles', 'exact_mass': 'mass', 'annotation_level': 'annot'})
    tr = pd.DataFrame([(k, s, m) for k, (s, m) in structures.items()], columns=['k14', 'smiles', 'mass'])
    tr = tr[np.isfinite(tr.mass)]
    tr['annot'] = 0.0
    cand = pd.concat([cand, tr[~tr.k14.isin(coconut_keys)]], ignore_index=True)
    cand['in_lib'] = cand.k14.isin(structures)
    cand = cand.sort_values('mass').reset_index(drop=True)
    cmass = cand.mass.to_numpy()

    ref_keys = np.array([k for k, _ in ref_meta])
    ref_mass = np.array([m for _, m in ref_meta])
    ref_order = np.argsort(ref_mass)
    ref_sorted = ref_mass[ref_order]
    R_frag = to_csr(ref_frag)
    # Combined [fragment ; loss] / sqrt(2) so that a dot product = mean of both cosines.
    R_comb = to_csr([combine(f, l) for f, l in zip(ref_frag, ref_loss)], width=2 * NBINS)
    del ref_frag, ref_loss
    print(f'reference matrices built {time.time() - t0:.0f}s', flush=True)

    fp = FPCache()
    key_smiles = {k: s for k, (s, _) in structures.items()}
    rows = []
    per_query = []
    for qi, (qkey, spectra) in enumerate(queries.items()):
        neutral = np.median([p - ADDUCT_MASS[a] for a, p, *_ in spectra])
        qf, ql = zip(*[spectrum_vectors(mz, it, p) for a, p, mz, it, *_ in spectra])
        Qf = to_csr(list(qf)).to_dense()
        Qc = to_csr([combine(f, l) for f, l in zip(qf, ql)], width=2 * NBINS).to_dense()

        # Global analog search (any mass).
        S_comb = torch.sparse.mm(R_comb, Qc.T).numpy()          # [n_ref, n_q]
        S_frag = torch.sparse.mm(R_frag, Qf.T).numpy()

        def top_keys(S, idx=None, k=args.k, consensus=False):
            if idx is not None:
                S = S[idx]
                keys = ref_keys[idx]
            else:
                keys = ref_keys
            best = S.max(axis=1)
            take = np.argsort(-best, kind='stable')[:k * REF_PER_MOL * 4]
            agg = {}
            for i in take:
                key = keys[i]
                if consensus:
                    agg.setdefault(key, []).append(S[i])
                else:
                    agg[key] = max(agg.get(key, 0.0), float(best[i]))
            if consensus:
                out = {}
                for key, mats in agg.items():
                    per_q = np.max(np.stack(mats), axis=0)
                    out[key] = .65 * float(per_q.max()) + .35 * float(np.mean(np.sort(per_q)[-3:]))
                agg = out
            return sorted(agg.items(), key=lambda x: -x[1])[:k]

        d35 = max(neutral * 35e-6, .006)
        iso_idx = ref_order[np.searchsorted(ref_sorted, neutral - d35):
                            np.searchsorted(ref_sorted, neutral + d35, side='right')]
        neigh = {
            'iso': top_keys(S_frag, iso_idx, consensus=True) if len(iso_idx) else [],
            'ana': top_keys(S_comb),
            'frag': top_keys(S_frag),
        }

        d = max(neutral * 15e-6, .004)
        lo, hi = np.searchsorted(cmass, neutral - d), np.searchsorted(cmass, neutral + d, side='right')
        pool = cand.iloc[lo:hi]
        if pool.empty:
            per_query.append(dict(key=qkey, n_cand=0, in_pool=False))
            continue
        cfps = [fp(s) for s in pool.smiles]
        ok = np.array([f is not None for f in cfps])
        pool = pool[ok]
        cfps = [f for f in cfps if f is not None]
        n = len(pool)
        in_pool = bool((pool.k14 == qkey).any())
        ppm = np.abs(pool.mass.to_numpy() - neutral) / neutral * 1e6
        gauss = np.exp(-.5 * (ppm / 15) ** 2)

        def sims(nb):
            nfps, ns = [], []
            for key, s in nb:
                f = fp(key_smiles.get(key))
                if f is not None:
                    nfps.append(f)
                    ns.append(s)
            if not nfps:
                return None, None
            T = np.array([DataStructs.BulkTanimotoSimilarity(f, nfps) for f in cfps])
            return T, np.array(ns)

        scores = {}
        for name, nb in neigh.items():
            T, s = sims(nb)
            if T is None:
                scores[f'{name}_hist'] = np.zeros(n)
                if name == 'ana':
                    scores['ana_knn'] = np.zeros(n)
                    scores['ana_knn_top5'] = np.zeros(n)
                continue
            scores[f'{name}_hist'] = (T * (.5 + .5 * s)).max(axis=1) * gauss
            if name == 'ana':
                w = s ** 4
                scores['ana_knn'] = (T * w).sum(1) / max(w.sum(), 1e-12)
                w5 = np.where(np.arange(len(s)) < 5, w, 0)
                scores['ana_knn_top5'] = (T * w5).sum(1) / max(w5.sum(), 1e-12)
            if name == 'iso':
                w = s ** 4
                scores['iso_knn'] = (T * w).sum(1) / max(w.sum(), 1e-12)
        scores.setdefault('iso_knn', np.zeros(n))
        scores['prior'] = pool.annot.fillna(0).to_numpy() + 1e-3 * gauss
        scores['in_lib'] = pool.in_lib.to_numpy().astype(float)

        base = ['iso_hist', 'ana_hist', 'ana_knn', 'prior']
        R = {k: ranks_from(v) for k, v in scores.items()}
        Z = {k: zscore(v) for k, v in scores.items()}
        fused = {
            'rrf60_iso+ana': 1 / (60 + R['iso_hist']) + 1 / (60 + R['ana_hist']),
            'rrf3_iso+ana': 1 / (3 + R['iso_hist']) + 1 / (3 + R['ana_hist']),
            'rrf3_iso+anaknn': 1 / (3 + R['iso_hist']) + 1 / (3 + R['ana_knn']),
            'z_iso+ana': Z['iso_hist'] + Z['ana_hist'],
            'z_iso+anaknn': Z['iso_hist'] + Z['ana_knn'],
            'z_iso+anaknn+.25prior': Z['iso_hist'] + Z['ana_knn'] + .25 * Z['prior'],
            'z_iso+anaknn+.5prior': Z['iso_hist'] + Z['ana_knn'] + .5 * Z['prior'],
            'z_isoknn+anaknn+.25prior': Z['iso_knn'] + Z['ana_knn'] + .25 * Z['prior'],
            'z_iso+.25prior': Z['iso_hist'] + .25 * Z['prior'],
            'z_anaknn+.25prior': Z['ana_knn'] + .25 * Z['prior'],
        }
        allscores = {**scores, **fused}
        rec = dict(key=qkey, n_cand=n, in_pool=in_pool, n_spec=len(spectra),
                   n_iso_neigh=len(neigh['iso']),
                   top_ana=neigh['ana'][0][1] if neigh['ana'] else 0.,
                   top_iso=neigh['iso'][0][1] if neigh['iso'] else 0.)
        if in_pool:
            true_i = int(np.flatnonzero(pool.k14.to_numpy() == qkey)[0])
            # Tanimoto of the true structure to its best neighbour (how informative neighbours are)
            tf = cfps[true_i]
            for nm in ['iso', 'ana']:
                nf = [fp(key_smiles.get(k)) for k, _ in neigh[nm]]
                nf = [f for f in nf if f is not None]
                rec[f'best_tani_{nm}'] = max(DataStructs.BulkTanimotoSimilarity(tf, nf)) if nf else 0.
            for name, v in allscores.items():
                # pessimistic tie handling: ties with the true candidate rank ahead of it
                rank = int((v > v[true_i]).sum() + (v == v[true_i]).sum())
                rec[f'rank_{name}'] = rank
        per_query.append(rec)
        if qi % 50 == 0:
            print(f'{qi + 1}/{len(queries)} {time.time() - t0:.0f}s', flush=True)

    df = pd.DataFrame(per_query)
    df.to_csv(out / f'per_query_{args.split}.csv', index=False)
    rank_cols = [c for c in df.columns if c.startswith('rank_')]
    summary = {'split': args.split, 'queries': int(len(df)), 'in_pool': int(df.in_pool.sum()),
               'median_candidates': float(df.n_cand.median()), 'seconds': round(time.time() - t0),
               'reference_spectra': len(ref_meta), 'methods': {}}
    rng = np.random.default_rng(0)
    ip = df[df.in_pool]
    rr = {c: np.where(ip[c] <= 25, 1 / ip[c], 0.) for c in rank_cols}
    for c in rank_cols:
        x = rr[c]
        boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
        diff = x - rr['rank_iso_hist']
        db = [diff[i].mean() for i in [rng.integers(0, len(x), len(x)) for _ in range(1000)]]
        summary['methods'][c[5:]] = {
            'mrr_in_pool': round(float(x.mean()), 4),
            'ci95': [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            'top1': round(float((ip[c] == 1).mean()), 4),
            'top5': round(float((ip[c] <= 5).mean()), 4),
            'top25': round(float((ip[c] <= 25).mean()), 4),
            'delta_vs_iso_hist': round(float(diff.mean()), 4),
            'delta_ci95': [round(float(np.percentile(db, 2.5)), 4), round(float(np.percentile(db, 97.5)), 4)],
        }
    (out / f'summary_{args.split}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
