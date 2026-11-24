"""Benchmark spectral-similarity variants on two tasks built from train.parquet.

Task A  cross-instrument identity: timsTOF (enveda-180) spectra of a molecule are queries; the library is
        every non-timsTOF spectrum with the same adduct within 10 ppm + 0.002 Da of the query precursor.
        A structure (inchikey14) scores the max similarity over its spectra x query spectra; MRR / top-1
        of the true structure among molecules whose structure is present with >=2 candidate structures.
Task B  analog retrieval (structure held out): one merged-CE timsTOF spectrum per query molecule, library =
        merged-CE spectra of 40k other enveda-180 molecules, same polarity, |neutral mass diff| <= 200 Da.
        Quality of the hits as structural evidence: Morgan-2 Tanimoto of top-1, mean of top-5, best of
        top-10, and soft-Tanimoto between the sim^3-weighted mean fingerprint of the top-80 and the truth
        (the 0.495 pipeline's analog features use N_ANALOG 80, SIM_POWER 3).
Paired bootstrap (2,000 resamples over molecules) gives CI95 of each variant's difference to the reference.
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import smlib
import bench_common as bc

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--task', choices=['A', 'B'], required=True)
ap.add_argument('--variants', default='all')
ap.add_argument('--n-query', type=int, default=1000)
ap.add_argument('--n-lib', type=int, default=40000)
ap.add_argument('--ref', default=None)
a = ap.parse_args()

BASE = dict(metric=1, power=1.0, ent_weight=True, floor=0.002, topk=256, rm_prec=False, tol_da=0.01, tol_ppm=0.0,
            shift=1, centroid_da=0.0, min_mz=0.0, nl=0.0)
V = {
    # the 0.495 pipeline settings (pv.CFG: entropy weighted, floor .002, top 256, 0.01 Da, max(unshifted, shifted))
    'pipeline_entropy': {},
    # repo baseline.py / extlib overlay style
    'cos_sqrt_0.015': dict(metric=0, power=0.5, ent_weight=False, floor=0.0, topk=0, shift=0, tol_da=0.015),
    'cos_sqrt_0.01': dict(metric=0, power=0.5, ent_weight=False, floor=0.0, topk=0, shift=0),
    'cos_raw_0.01': dict(metric=0, power=1.0, ent_weight=False, floor=0.0, topk=0, shift=0),
    'entropy_unweighted': dict(ent_weight=False),
    # tolerance
    'ent_tol_0.02': dict(tol_da=0.02),
    'ent_tol_0.005': dict(tol_da=0.005),
    'ent_tol_0.003': dict(tol_da=0.003),
    'ent_tol_10ppm+0.002': dict(tol_da=0.002, tol_ppm=10.0),
    'ent_tol_20ppm+0.002': dict(tol_da=0.002, tol_ppm=20.0),
    'ent_tol_5ppm+0.001': dict(tol_da=0.001, tol_ppm=5.0),
    'cos_sqrt_10ppm+0.002': dict(metric=0, power=0.5, ent_weight=False, floor=0.0, topk=0, shift=0, tol_da=0.002, tol_ppm=10.0),
    # peak processing
    'ent_rm_prec': dict(rm_prec=True),
    'ent_floor_0.01': dict(floor=0.01),
    'ent_floor_0': dict(floor=0.0),
    'ent_top64': dict(topk=64),
    'ent_top128': dict(topk=128),
    'ent_centroid_0.005': dict(centroid_da=0.005),
    'ent_minmz_50': dict(min_mz=50.0),
    # shift handling
    'ent_noshift': dict(shift=0),
    'ent_modified': dict(shift=2),
    # combined candidates
    'ent_10ppm_rmprec': dict(tol_da=0.002, tol_ppm=10.0, rm_prec=True),
    'ent_10ppm_rmprec_modified': dict(tol_da=0.002, tol_ppm=10.0, rm_prec=True, shift=2),
    'ent_10ppm_modified': dict(tol_da=0.002, tol_ppm=10.0, shift=2),
    'ent_modified_rmprec': dict(shift=2, rm_prec=True),
    'cos_sqrt_modified_0.01_rmprec': dict(metric=0, power=0.5, ent_weight=False, floor=0.0, topk=0, shift=2, rm_prec=True),
    # neutral-loss fusion (fragment sim + nl * NL sim)
    'ent_nl_0.5': dict(nl=0.5, rm_prec=True),
}
if a.variants != 'all':
    V = {k: V[k] for k in a.variants.split(',')}
ref = a.ref or next(iter(V))

data = Path(a.data)
run_pairs = bc.run_pairs


def boot(diff, n=2000):
    r = np.random.default_rng(0)
    idx = r.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


results = {}
if a.task == 'A':
    sp = pd.read_parquet(data / 'spectra.parquet'); roles = json.load(open(data / 'roles.json'))
    cq = set(roles['cross_query'])
    q = sp[(sp.ingest_lib == 'enveda-180') & sp.inchikey14.isin(cq)].reset_index(drop=True)
    lib = sp[sp.instrument_type != 'timsTOF'].reset_index(drop=True)
    lib = lib[lib.ms2_mzs.map(len) > 0].reset_index(drop=True)
    pqs, prs = [], []
    lib_by_ad = {ad: (g.precursor_mz.to_numpy(), g.index.to_numpy()) for ad, g in lib.groupby('adduct')}
    for ad in lib_by_ad:
        o = np.argsort(lib_by_ad[ad][0]); lib_by_ad[ad] = (lib_by_ad[ad][0][o], lib_by_ad[ad][1][o])
    for qi_, r in q.iterrows():
        if r.adduct not in lib_by_ad: continue
        pm, ix = lib_by_ad[r.adduct]
        tol = r.precursor_mz * 10e-6 + 0.002
        lo, hi = np.searchsorted(pm, r.precursor_mz - tol), np.searchsorted(pm, r.precursor_mz + tol, 'right')
        for j in ix[lo:hi]:
            pqs.append(qi_); prs.append(j)
    pq_ = np.array(pqs, np.int64); pr_ = np.array(prs, np.int64)
    qik = q.inchikey14.to_numpy()[pq_]; rik = lib.inchikey14.to_numpy()[pr_]
    # eligible molecules: true structure among candidates and >=2 candidate structures
    cand = pd.DataFrame(dict(m=qik, c=rik)).drop_duplicates()
    g = cand.groupby('m').c.agg(lambda s: set(s))
    elig = sorted(m for m, cs in g.items() if m in cs and len(cs) >= 2)
    print(f'task A: {len(q)} query spectra, {len(lib)} library spectra, {len(pq_):,} pairs, eligible molecules {len(elig)}', flush=True)
    qrows = list(zip(q.ms2_mzs, q.ms2_normalized_intensities, q.precursor_mz))
    rrows = list(zip(lib.ms2_mzs, lib.ms2_normalized_intensities, lib.precursor_mz))
    per = {}
    for name, over in V.items():
        cfg = {**BASE, **over}; t0 = time.time()
        s, n = run_pairs(pq_, pr_, qrows, rrows, cfg)
        # per query spectrum (single spectrum, harder) and per molecule (max over its spectra)
        df = pd.DataFrame(dict(q=pq_, m=qik, c=rik, s=s))
        df = df[df.m.isin(set(elig))]
        ds = df.groupby(['q', 'm', 'c']).s.max().reset_index()
        ok_q = ds[ds.m == ds.c].q.unique(); ds = ds[ds.q.isin(set(ok_q))]
        ds['rank'] = ds.groupby('q').s.rank(ascending=False, method='average')
        rs = ds[ds.m == ds.c].set_index('q')
        spec_rr = rs.groupby('m')['rank'].apply(lambda r: float(np.mean(1.0 / r))).reindex(elig)  # molecule-averaged
        dm = df.groupby(['m', 'c']).s.max().reset_index()
        dm['rank'] = dm.groupby('m').s.rank(ascending=False, method='average')
        rr = dm[dm.m == dm.c].set_index('m')['rank'].reindex(elig)
        rec = spec_rr.to_numpy()
        per[name] = rec
        results[name] = dict(spectrum_mrr=float(rec.mean()), spectrum_top1=float((rs['rank'] == 1).mean()), n_spectra=int(len(rs)),
                             molecule_mrr=float((1.0 / rr).mean()), molecule_top1=float((rr == 1).mean()), n=len(rec),
                             secs=round(time.time() - t0, 1), cfg=cfg)
        print(name, json.dumps({k: v for k, v in results[name].items() if k != 'cfg'}), flush=True)
    for name in per:
        d = per[name] - per[ref]
        results[name]['d_mrr_vs_ref'] = float(d.mean()); results[name]['ci95'] = boot(d)
else:
    q, lib, qfp, lfp, pq_, pr_, tan = bc.build_task_b(a.data, a.n_query, a.n_lib)
    print(f'task B: {len(q)} queries, {len(lib)} library spectra, {len(pq_):,} pairs', flush=True)
    qrows = list(zip(q.ms2_mzs, q.ms2_normalized_intensities, q.precursor_mz))
    rrows = list(zip(lib.ms2_mzs, lib.ms2_normalized_intensities, lib.precursor_mz))
    bounds = np.r_[0, np.cumsum(np.bincount(pq_, minlength=len(q)))]
    order_q = np.argsort(pq_, kind='stable')
    pq_s, pr_s, tan_s = pq_[order_q], pr_[order_q], tan[order_q]
    # oracle / random references
    orc, rnd = [], []
    for k in range(len(q)):
        t = tan_s[bounds[k]:bounds[k + 1]]
        if len(t): orc.append(t.max()); rnd.append(t.mean())
    results['_reference'] = dict(oracle_best_tanimoto=float(np.mean(orc)), random_mean_tanimoto=float(np.mean(rnd)))
    per = {}
    for name, over in V.items():
        cfg = {**BASE, **over}; t0 = time.time()
        s, n = run_pairs(pq_s, pr_s, qrows, rrows, cfg)
        m1, m5, b10, fr = [], [], [], []
        for k in range(len(q)):
            a_, b_ = bounds[k], bounds[k + 1]
            if b_ <= a_: continue
            ss = s[a_:b_]; tt = tan_s[a_:b_]
            o = np.lexsort((pr_s[a_:b_], -ss))
            m1.append(tt[o[0]]); m5.append(tt[o[:5]].mean()); b10.append(tt[o[:10]].max())
            top = o[:80]; w = np.maximum(ss[top], 0) ** 3
            if w.sum() <= 0: fr.append(0.0); continue
            f = (w[:, None] * lfp[pr_s[a_:b_][top]]).sum(0) / w.sum(); tr = qfp[k]
            fr.append(np.minimum(f, tr).sum() / np.maximum(f, tr).sum())
        per[name] = dict(t1=np.array(m1), t5=np.array(m5), b10=np.array(b10), fr=np.array(fr))
        results[name] = dict(tan_top1=float(np.mean(m1)), tan_top5=float(np.mean(m5)), best_tan_top10=float(np.mean(b10)),
                             soft_tan_weighted_fp=float(np.mean(fr)), frac_top1_tan_ge_0p7=float(np.mean(np.array(m1) >= 0.7)),
                             n=len(m1), secs=round(time.time() - t0, 1), cfg=cfg)
        print(name, json.dumps({k: v for k, v in results[name].items() if k != 'cfg'}), flush=True)
    for name in per:
        for key in ('t1', 'fr'):
            d = per[name][key] - per[ref][key]
            results[name][f'd_{key}_vs_ref'] = float(d.mean()); results[name][f'ci95_{key}'] = boot(d)
results['_meta'] = dict(task=a.task, ref=ref, n_query=a.n_query, n_lib=a.n_lib)
json.dump(results, open(a.out, 'w'), indent=1)
