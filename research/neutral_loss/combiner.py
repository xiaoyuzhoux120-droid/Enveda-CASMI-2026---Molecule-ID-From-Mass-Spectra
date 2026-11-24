"""Experiment 2: learned isomer reranker (GBDT) on MetFrag-lite + neutral-loss <-> substructure agree/clash features.

Arms (identical model / training panels, only the loss features differ):
  mf_only   MetFrag-lite scores (DEFAULT + BASELINE cfg) + candidate descriptors
  rr8_13    + agree/clash for the 13 hand-written losses of the notebook's RR8 reranker
  ext_31    + agree/clash for an extended, mechanistic loss dictionary (31 losses, count-aware requirements)

Training panels: 'fit'-split molecules with a non-enveda-180 spectrum (natural-product-like), never the
evaluation molecules. Evaluation: the 'eval' (2000) and 'np' (enveda-np-examples) panels of rank_exp.py.

Usage: python combiner.py WORK_DIR FRAG_MODULE_DIR [n_train_molecules]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent))
from analyze import boot_ci, exp_rr  # noqa: E402
from rank_exp import RR8_LOSSES  # noqa: E402

# name: (loss mass, SMARTS the candidate must contain, minimum number of matches)
EXT_LOSSES = {n: (m, s, 1) for n, (m, s) in RR8_LOSSES.items()}
EXT_LOSSES['h2o'] = (18.010565, '[OX2H1]', 1)
EXT_LOSSES.update({
    '2h2o': (36.021129, '[OX2H1]', 2),
    'ch2o': (30.010565, '[$([CH2][OX2H1]),$([CH3][OX2])]', 1),
    'c2h4o2': (60.021129, '[CH3]C(=O)O[#6]', 1),
    'c4h8': (56.062600, '[CH3]C([CH3])=[CH][CH2,c]', 1),
    'c5h8': (68.062600, '[CH3]C([CH3])=[CH][CH2]', 1),
    'hcn': (27.010899, '[$(n),$(C#N),$([NX3;H2]c)]', 1),
    'ch5n': (31.042199, '[CH3][NX3;!$(NC=O)]', 1),
    'c2h7n': (45.057849, '[CH3][NX3;!$(NC=O)][CH3]', 1),
    'c3h9n': (59.073499, '[CH3][NX3,NX4+]([CH3])[CH3]', 1),
    'hex_h2o': (180.063388, '[OX2][CX4;R]1[OX2;R][CX4;R][CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]', 1),
    'dhex_h2o': (164.068473, 'C[CX4;R]1[OX2;R][CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]', 1),
    'coumaroyl': (146.036779, 'O=C(C=Cc1ccc([OX2])cc1)[O,N]', 1),
    'caffeoyl': (162.031694, 'O=C(C=Cc1ccc([OX2])c([OX2])c1)[O,N]', 1),
    'feruloyl': (176.047344, 'O=C(C=Cc1ccc([OX2])c(O[CH3])c1)[O,N]', 1),
    'galloyl': (152.010959, 'O=C(c1cc(O)c(O)c(O)c1)[O]', 1),
    'malonyl': (86.000394, 'O=C([OX2H1,OX1-])[CH2]C(=O)O[#6]', 1),
    'h2s': (33.987721, '[SX2]', 1),
    'hcl': (35.976678, 'Cl', 1),
})
SETS = {'mf_only': [], 'rr8_13': list(RR8_LOSSES), 'ext_31': list(EXT_LOSSES)}
assert len(EXT_LOSSES) == 31, len(EXT_LOSSES)
_P = {n: Chem.MolFromSmarts(v[1]) for n, v in EXT_LOSSES.items()}
_P13 = {n: Chem.MolFromSmarts(v[1]) for n, v in RR8_LOSSES.items()}
assert all(p is not None for p in _P.values())


def cand_facts(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    ext = {n: int(len(m.GetSubstructMatches(p)) >= EXT_LOSSES[n][2]) for n, p in _P.items()}
    r13 = {n: int(m.HasSubstructMatch(p)) for n, p in _P13.items()}
    heavy = m.GetNumHeavyAtoms()
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    desc = [heavy, m.GetRingInfo().NumRings(), sum(a.GetIsAromatic() for a in m.GetAtoms()) / max(heavy, 1),
            sym.count('O'), sym.count('N')]
    return ext, r13, desc


def observed(spectra, tol=0.006):
    out = {}
    for n, (mass, _, _) in EXT_LOSSES.items():
        v = 0.
        for sp in spectra:
            v += sp['it'][np.abs(sp['mz'] - (sp['prec'] - mass)) <= tol].sum()
        out[n] = v / max(len(spectra), 1)
    return out


def panel_features(spectra, cands, frag, fcache, facts):
    mf1 = frag.score_molecule(spectra, cands, cfg=None, cache=fcache)
    mf2 = frag.score_molecule(spectra, cands, cfg=frag.BASELINE_CFG, cache=fcache)
    obs = observed(spectra)
    feats = {k: [] for k in SETS}
    for s, a, b in zip(cands, mf1, mf2):
        f = facts.get(s)
        if f is None:
            f = facts[s] = cand_facts(s)
        if f is None or a is None or b is None:
            for k in SETS:
                feats[k].append(None)
            continue
        ext, r13, desc = f
        base = [a, b] + desc
        for k, names in SETS.items():
            extra = []
            for n in names:
                has = r13[n] if k == 'rr8_13' else ext[n]
                extra += [obs[n] * has, obs[n] * (1 - has)]
            feats[k].append(base + extra)
    out = {}
    for k, rows in feats.items():
        d = len(next((r for r in rows if r is not None), [0] * (7 + 2 * len(SETS[k]))))
        X = np.array([r if r is not None else [np.nan] * d for r in rows], float)
        with np.errstate(all='ignore'):
            Xc = X - np.nanmean(X, axis=0, keepdims=True)
        out[k] = np.hstack([X, Xc])
    return out


def spectra_of(sub):
    return [dict(mz=np.asarray(r.ms2_mzs, float), it=np.asarray(r.ms2_normalized_intensities, float) /
                 max(np.max(r.ms2_normalized_intensities), 1e-12), adduct=r.adduct, prec=float(r.precursor_mz),
                 mode=1 if r.adduct.endswith('+') else -1) for r in sub.itertuples(index=False)]


def main(work, frag_dir, n_train=4000, max_cand=40):
    work = Path(work)
    sys.path.insert(0, frag_dir)
    import frag_rescore as frag
    frag.warmup()
    df = pd.read_parquet(work / 'spectra.parquet')
    cat = pd.read_parquet(work / 'catalog.parquet')
    by_formula = cat.groupby('molecular_formula')
    fcache, facts = {}, {}
    rng = np.random.default_rng(11)

    def build(keys_frame, tag):
        panels = []
        for q in keys_frame.itertuples(index=False):
            if q.molecular_formula not in by_formula.groups:
                continue
            iso = by_formula.get_group(q.molecular_formula)
            iso = iso[iso.inchikey14 != q.inchikey14]
            if len(iso) == 0:
                continue
            if len(iso) > max_cand - 1:
                iso = iso.iloc[rng.choice(len(iso), max_cand - 1, replace=False)]
            cands = [q.normalized_smiles] + list(iso.normalized_smiles)
            sub = df[(df.inchikey14 == q.inchikey14) & (df.split == q.split)]
            if tag == 'train':
                sub = sub[sub.ingest_lib != 'enveda-180']
            if len(sub) == 0:
                continue
            F = panel_features(spectra_of(sub), cands, frag, fcache, facts)
            panels.append(dict(key=q.inchikey14, split=q.split, F=F))
            if len(fcache) > 20000:
                fcache.clear()
        print(tag, 'panels', len(panels), flush=True)
        return panels

    nat_fit = df[(df.split == 'fit') & (df.ingest_lib != 'enveda-180')].drop_duplicates('inchikey14')
    nform = cat.set_index('inchikey14').molecular_formula.map(cat.groupby('molecular_formula').size())
    nat_fit = nat_fit[nat_fit.inchikey14.map(nform).fillna(0) >= 2].head(n_train)
    train = build(nat_fit[['inchikey14', 'normalized_smiles', 'molecular_formula', 'split']], 'train')
    test_q = df[df.split.isin(['eval', 'np'])].drop_duplicates(['inchikey14', 'split'])
    test = build(test_q[['inchikey14', 'normalized_smiles', 'molecular_formula', 'split']], 'test')
    res = {}
    rr = {'metfrag_lite': {s: [] for s in ('eval', 'np')}}
    for p in test:
        rr['metfrag_lite'][p['split']].append(exp_rr(np.nan_to_num(p['F']['mf_only'][:, 0], nan=-1e9))[0])
    for arm in SETS:
        X = np.vstack([p['F'][arm] for p in train])
        y = np.concatenate([[1] + [0] * (len(p['F'][arm]) - 1) for p in train])
        ok = ~np.isnan(X[:, 0])
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(X[ok], y[ok])
        rr[arm] = {s: [] for s in ('eval', 'np')}
        for p in test:
            sc = clf.predict_proba(np.nan_to_num(p['F'][arm], nan=0))[:, 1]
            sc[np.isnan(p['F'][arm][:, 0])] = -1
            rr[arm][p['split']].append(exp_rr(sc)[0])
    for s in ('eval', 'np'):
        line = {}
        for arm in rr:
            line[arm] = round(float(np.mean(rr[arm][s])), 4)
        for a, b in (('mf_only', 'metfrag_lite'), ('rr8_13', 'metfrag_lite'), ('ext_31', 'metfrag_lite'),
                     ('rr8_13', 'mf_only'), ('ext_31', 'mf_only'), ('ext_31', 'rr8_13')):
            m, lo, hi = boot_ci(rr[a][s], rr[b][s])
            line[f'{a}-{b}'] = [round(m, 4), round(lo, 4), round(hi, 4)]
        line['panels'] = len(rr['mf_only'][s])
        res[s] = line
        print(s, json.dumps(line), flush=True)
    json.dump(res, open(work / 'combiner_summary.json', 'w'), indent=1)
    json.dump(rr, open(work / 'combiner_per_panel.json', 'w'))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:]))
