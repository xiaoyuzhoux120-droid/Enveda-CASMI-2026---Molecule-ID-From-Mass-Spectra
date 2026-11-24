"""Empirical fragment m/z error of enveda-180 timsTOF spectra via sub-formula annotation.

For each peak (top-N by intensity, m/z >= 50, below precursor - 1) the closest sub-formula ion of the
known molecular formula is found (C,H,N,O,S,P,Cl subsets; H free; even/odd electron both allowed).
The error histogram within +-30 ppm is a mixture of true annotations (peaked) and chance matches
(roughly flat); the flat part is estimated from the 20-30 ppm shoulders and subtracted.
Also reports the replicate error: same molecule, different collision energies, nearest peak pairs.
"""
import sys, re, json, itertools
import numpy as np, pandas as pd
M = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, S=31.97207100, P=30.97376163, Cl=34.96885268)
E = 0.00054857990

def parse(f):
    d = {}
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', f):
        d[el] = d.get(el, 0) + int(n or 1)
    return d

def annotate(f, mzs, mode):
    d = parse(f)
    if set(d) - set(M): return None
    ranges = [range(d.get(el, 0) + 1) for el in ('C', 'N', 'O', 'S', 'P', 'Cl')]
    heavy = np.array([c * M['C'] + n * M['N'] + o * M['O'] + s * M['S'] + p * M['P'] + cl * M['Cl']
                      for c, n, o, s, p, cl in itertools.product(*ranges)])
    if len(heavy) > 400000: return None
    hmax = d.get('H', 0) + 1
    out = []
    for mz in mzs:
        neutral = mz + E if mode > 0 else mz - E
        h = np.rint((neutral - heavy) / M['H'])
        ok = (h >= 0) & (h <= hmax)
        if not ok.any(): out.append(np.nan); continue
        theo = heavy[ok] + h[ok] * M['H']
        k = np.argmin(abs(theo - neutral))
        th = theo[k] - E if mode > 0 else theo[k] + E
        out.append((mz - th) / th * 1e6)
    return np.array(out)

sp = pd.read_parquet(sys.argv[1])
roles = json.load(open(sys.argv[2]))
q = sp[(sp.ingest_lib == 'enveda-180') & sp.inchikey14.isin(set(roles['analog_query']))]
q = q[q.adduct.isin(['[M+H]+', '[M-H]-'])]
rng = np.random.default_rng(0)
mols = rng.choice(q.inchikey14.unique(), 400, replace=False)
errs, mzl = [], []
rep = []
for ik in mols:
    g = q[q.inchikey14 == ik]
    for _, r in g.iterrows():
        mz = np.asarray(r.ms2_mzs); it = np.asarray(r.ms2_normalized_intensities)
        keep = (mz >= 50) & (mz < r.precursor_mz - 1)
        mz, it = mz[keep], it[keep]
        top = np.argsort(-it)[:20]
        e = annotate(r.molecular_formula, mz[top], 1 if r.adduct.endswith('+') else -1)
        if e is None: continue
        errs.extend(e); mzl.extend(mz[top])
    # replicate error between single-CE spectra of same adduct
    for ad, gg in g.groupby('adduct'):
        singles = gg[~gg.collision_energy_orig.str.contains(',')]
        if len(singles) < 2: continue
        a, b = singles.iloc[0], singles.iloc[1]
        am, ai = np.asarray(a.ms2_mzs), np.asarray(a.ms2_normalized_intensities)
        bm = np.sort(np.asarray(b.ms2_mzs))
        for m in am[np.argsort(-ai)[:20]]:
            j = np.searchsorted(bm, m)
            c = [bm[k] for k in (j - 1, j) if 0 <= k < len(bm)]
            if c:
                n = min(c, key=lambda x: abs(x - m))
                rep.append(((n - m) / m * 1e6, m))
errs = np.array(errs); mzl = np.array(mzl)
ok = np.isfinite(errs) & (abs(errs) <= 30)
e = errs[ok]; m = mzl[ok]
bins = np.arange(-30, 31, 2)
h, _ = np.histogram(e, bins)
flat = np.mean(np.r_[h[:5], h[-5:]])
sig = np.clip(h - flat, 0, None)
res = dict(n_peaks=int(len(errs)), within30=int(ok.sum()), hist=dict(zip([f'{x:+.0f}' for x in bins[:-1]], h.tolist())),
           flat_per_2ppm=float(flat))
c = (bins[:-1] + bins[1:]) / 2
cs = np.cumsum(sig) / sig.sum()
res['signal_q05_q50_q95_ppm'] = [float(np.interp(q_, cs, c)) for q_ in (.05, .5, .95)]
core = e[abs(e) <= 10]
res['core_median_ppm'] = float(np.median(core)); res['core_mad_ppm'] = float(np.median(abs(core - np.median(core))))
for lo, hi in ((50, 150), (150, 300), (300, 600), (600, 1200)):
    s = (m >= lo) & (m < hi) & (abs(e) <= 10)
    if s.sum() > 20:
        res[f'core_abs_err_q95_Da_{lo}_{hi}'] = float(np.quantile(abs(e[s]) * m[s] * 1e-6, .95))
        res[f'core_q95_ppm_{lo}_{hi}'] = float(np.quantile(abs(e[s] - np.median(core)), .95))
rep = np.array(rep)
r = rep[abs(rep[:, 0]) <= 30]
res['replicate_pairs'] = int(len(rep))
res['replicate_within_5ppm'] = float(np.mean(abs(rep[:, 0]) <= 5)); res['replicate_within_10ppm'] = float(np.mean(abs(rep[:, 0]) <= 10))
res['replicate_q50_q90_q95_absppm_within30'] = [float(np.quantile(abs(r[:, 0]), q_)) for q_ in (.5, .9, .95)]
rr = rep[abs(rep[:, 0]) <= 10]
res['replicate_q95_abs_Da'] = float(np.quantile(abs(rr[:, 0]) * rr[:, 1] * 1e-6, .95))
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[3], 'w'), indent=1)
