"""Retrain the FRP5 notebook's RR8 isomer reranker (LightGBM lambdarank) on natural-product-like panels, with the
original 13-loss features and with the extended 31-loss dictionary, and compare against the shipped RR8 booster.

Needs, in MOD_DIR: frag_rescore.py, rr_features.py, rr_overlay.py (exported verbatim from the notebook's
`_physical`, `_RR8_FEATURES`, `_RR8_OVERLAY` strings), rr_features_ext.py (31-loss variant), rr8_model.txt
(the shipped booster, decoded from `_RR8_MODEL_Z`).

Usage: python rr8_retrain.py WORK_DIR MOD_DIR [n_train_molecules]
Writes MOD_DIR/rr31_model.txt, MOD_DIR/rr13_model.txt and WORK_DIR/rr8_retrain_summary.json.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze import boot_ci, exp_rr  # noqa: E402

PARAMS = dict(objective='lambdarank', learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              lambdarank_truncation_level=40, verbose=-1, seed=0, num_threads=4)
N_ROUNDS = 200


def spectra_of(sub):
    out = []
    for r in sub.itertuples(index=False):
        it = np.asarray(r.ms2_normalized_intensities, float)
        out.append(dict(mz=np.asarray(r.ms2_mzs, float), it=it / max(it.max(), 1e-12), prec=float(r.precursor_mz),
                        adduct=str(r.adduct), mode=1 if str(r.adduct).endswith('+') else -1, ce=float('nan')))
    return out


def main(work, mod_dir, n_train=8000, max_cand=40):
    work = Path(work)
    sys.path.insert(0, mod_dir)
    import lightgbm as lgb
    frag = importlib.import_module('frag_rescore')
    ov = importlib.import_module('rr_overlay')
    rf13 = importlib.import_module('rr_features')
    rf31 = importlib.import_module('rr_features_ext')
    frag.warmup()
    df = pd.read_parquet(work / 'spectra.parquet', columns=['inchikey14', 'normalized_smiles', 'molecular_formula',
                                                            'ingest_lib', 'adduct', 'precursor_mz', 'ms2_mzs',
                                                            'ms2_normalized_intensities', 'split'])
    cat = pd.read_parquet(work / 'catalog.parquet')
    by_formula = cat.groupby('molecular_formula')
    nform = cat.groupby('molecular_formula').size()
    rng = np.random.default_rng(23)
    spec_by = {k: g for k, g in df.groupby(['inchikey14', 'split'])}

    def panels(queries, nat_only):
        out = []
        cache13, cache31, fcache = {}, {}, {}
        for q in queries.itertuples(index=False):
            iso = by_formula.get_group(q.molecular_formula)
            iso = iso[iso.inchikey14 != q.inchikey14]
            if len(iso) == 0:
                continue
            if len(iso) > max_cand - 1:
                iso = iso.iloc[rng.choice(len(iso), max_cand - 1, replace=False)]
            cands = [q.normalized_smiles] + list(iso.normalized_smiles)
            sub = spec_by[(q.inchikey14, q.split)]
            if nat_only:
                sub = sub[sub.ingest_lib != 'enveda-180']
            if len(sub) == 0:
                continue
            sp = spectra_of(sub)
            F13 = ov.model_features(rf13, frag, sp, cands, cache13, fcache)
            F31 = ov.model_features(rf31, frag, sp, cands, cache31, fcache)
            mf = np.array([np.nan if v is None else v for v in frag.score_molecule(sp, cands, cfg=None, cache=fcache)])
            out.append(dict(key=q.inchikey14, split=q.split, F13=F13, F31=F31, mf=mf))
            for c in (cache13, cache31, fcache):
                if len(c) > 30000:
                    c.clear()
            if len(out) % 1000 == 0:
                print('  panels', len(out), flush=True)
        return out

    mols = df.drop_duplicates(['inchikey14', 'split'])
    mols = mols[mols.molecular_formula.map(nform).fillna(0) >= 2]
    train_q = mols[(mols.split == 'fit') & (mols.ingest_lib != 'enveda-180')]
    # a molecule qualifies if any of its fit spectra is non-enveda-180
    nat_keys = set(df.loc[(df.split == 'fit') & (df.ingest_lib != 'enveda-180'), 'inchikey14'])
    train_q = mols[(mols.split == 'fit') & mols.inchikey14.isin(nat_keys)].head(n_train)
    test_q = mols[mols.split.isin(['eval', 'np'])]
    print('train queries', len(train_q), 'test queries', len(test_q), flush=True)
    train = panels(train_q, True)
    test = panels(test_q, False)

    boosters = {}
    for tag in ('F13', 'F31'):
        X = np.vstack([p[tag] for p in train])
        y = np.concatenate([[1] + [0] * (len(p[tag]) - 1) for p in train])
        grp = [len(p[tag]) for p in train]
        ds = lgb.Dataset(X, y, group=grp)
        boosters[tag] = lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)
    shipped = lgb.Booster(model_file=str(Path(mod_dir) / 'rr8_model.txt'))
    arms = {'metfrag_lite': lambda p: p['mf'], 'shipped_rr8_13': lambda p: shipped.predict(p['F13']),
            'retrained_13': lambda p: boosters['F13'].predict(p['F13']),
            'retrained_31': lambda p: boosters['F31'].predict(p['F31'])}
    rr = {a: {'eval': [], 'np': []} for a in arms}
    for p in test:
        bad = np.isnan(p['F13'][:, 0])
        for a, f in arms.items():
            s = np.asarray(f(p), float)
            s = np.where(bad | ~np.isfinite(s), -1e9, s)
            rr[a][p['split']].append(exp_rr(s)[0])
    res = {}
    for s in ('eval', 'np'):
        line = {a: round(float(np.mean(rr[a][s])), 4) for a in arms}
        for a, b in (('retrained_13', 'shipped_rr8_13'), ('retrained_31', 'shipped_rr8_13'),
                     ('retrained_31', 'retrained_13'), ('shipped_rr8_13', 'metfrag_lite'),
                     ('retrained_31', 'metfrag_lite')):
            m, lo, hi = boot_ci(rr[a][s], rr[b][s])
            line[f'{a}-{b}'] = [round(m, 4), round(lo, 4), round(hi, 4)]
        line['panels'] = len(rr['metfrag_lite'][s])
        res[s] = line
        print(s, json.dumps(line), flush=True)
    res['train_panels'] = len(train)
    res['params'] = dict(PARAMS, num_boost_round=N_ROUNDS)
    json.dump(res, open(work / 'rr8_retrain_summary.json', 'w'), indent=1)
    boosters['F13'].save_model(str(Path(mod_dir) / 'rr13_model.txt'))
    boosters['F31'].save_model(str(Path(mod_dir) / 'rr31_model.txt'))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:]))
