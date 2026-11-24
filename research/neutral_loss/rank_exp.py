"""Same-formula isomer ranking: does neutral-loss / substructure prior knowledge separate isomers that mass alone
cannot?  Compares (all untrained on the evaluation molecules):

  random         expected MRR of a random order
  rr8_losses     13 hand-written loss<->SMARTS agree/clash terms (the inactive RR8 reranker's loss features)
  metfrag_lite   bond-cleavage explained intensity (frag_rescore DEFAULT_CFG, the 0.495 notebook's physical check)
  nn_<feat>_<tgt> Bernoulli log-likelihood of the candidate's substructure vector under an MLP trained on 'fit'
                 feat: int = integer m/z + integer loss histograms (casmi_ml representation)
                       hr  = int + mined high-resolution loss/fragment vocabulary (+/- 4 mDa)
                 tgt:  chem = 41 NP SMARTS groups + 166 MACCS keys; morgan = Morgan r2 2048
  combos         z(metfrag) + w * z(nn), w chosen on half A of 'eval', reported on half B and on 'np'.

Usage: python rank_exp.py WORK_DIR FRAG_MODULE_DIR [max_candidates]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import chem  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(0)
RR8_LOSSES = {
    'h2o': (18.010565, '[OX2H1]'), 'co2': (43.989829, '[CX3](=O)[OX2H1,OX1-]'),
    'hcooh': (46.005479, '[CX3](=O)[OX2H1,OX1-]'), 'nh3': (17.026549, '[NX3;H2,H1;!$(NC=O)]'),
    'ch3': (15.023475, '[CH3][OX2,NX3]'), 'ch3oh': (32.026215, '[CH3][OX2]'), 'c2h2o': (42.010565, '[CH3]C(=O)[O,N]'),
    'hexose': (162.052824, '[OX2;R][CX4;R]1[OX2;R][CX4;R][CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]'),
    'dhex': (146.057909, 'C[CX4;R]1[OX2;R][CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]'),
    'pentose': (132.042259, '[OX2][CX4;R]1[OX2;R][CX4;R][CX4;R]([OX2])[CX4;R]1[OX2]'),
    'glcA': (176.032088, 'OC(=O)[CX4;R]1[OX2;R][CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]'),
    'so3': (79.956815, 'S(=O)(=O)[OX2H1,OX1-,OX2]'), 'co': (27.994915, '[CX3]=O'),
}


def train_mlp(X, Y, epochs=8, hidden=512, lr=2e-3, bs=1024, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                              torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                              torch.nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(X) + bs - 1) // bs))
    lossf = torch.nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X), generator=g)
        tot = 0.
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(X[b]), Y[b])
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(b)
        print(f'    epoch {ep} bce {tot / len(X):.4f}', flush=True)
    net.eval()
    return net


def predict(net, X):
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(X, dtype=torch.float32))).numpy()


def loglik(P, Y):
    P = np.clip(P, 1e-4, 1 - 1e-4)
    return (Y * np.log(P) + (1 - Y) * np.log(1 - P)).sum(1)


def rank_of_true(scores, true_idx):
    s = np.asarray(scores, float)
    s = np.where(np.isfinite(s), s, -np.inf)
    # pessimistic tie handling: ties with the true candidate count against it (no free lunch from constant scores)
    return 1 + int((s > s[true_idx]).sum()) + int(((s == s[true_idx]).sum() - 1))


def z(v):
    v = np.asarray(v, float)
    v = np.where(np.isfinite(v), v, np.nanmin(np.where(np.isfinite(v), v, np.nan)) if np.isfinite(v).any() else 0)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)


def main(work, frag_dir, max_cand=40):
    work = Path(work)
    sys.path.insert(0, frag_dir)
    import frag_rescore as frag
    from rdkit import Chem
    frag.warmup()
    D = np.load(work / 'features.npz', allow_pickle=True)
    df = pd.read_parquet(work / 'spectra.parquet')
    split, keys = D['split'], D['keys']
    assert len(df) == len(split), 'spectra/features misaligned'
    assert (df.inchikey14.values.astype(str) == keys).all()
    fit = split == 'fit'
    Xint = D['Xint'].astype(np.float32)
    Xhr = D['Xhr'].astype(np.float32)
    mode = D['mode']
    feats = {'int': np.hstack([Xint, mode]), 'hr': np.hstack([Xint, Xhr, mode])}
    tgts = {'chem': np.hstack([D['Y_np'], D['Y_maccs']]).astype(np.float32), 'morgan': D['Y_morgan'].astype(np.float32)}

    # ---- panels ----
    cat = pd.read_parquet(work / 'catalog.parquet')
    fit_keys = set(keys[fit])
    by_formula = cat.groupby('molecular_formula')
    queries = df[~fit].drop_duplicates('inchikey14')[['inchikey14', 'normalized_smiles', 'molecular_formula', 'split']]
    rng = np.random.default_rng(7)
    panels = []
    for q in queries.itertuples(index=False):
        if q.molecular_formula not in by_formula.groups:
            continue
        iso = by_formula.get_group(q.molecular_formula)
        iso = iso[iso.inchikey14 != q.inchikey14]
        if len(iso) == 0:
            continue
        if len(iso) > max_cand - 1:
            iso = iso.iloc[rng.choice(len(iso), max_cand - 1, replace=False)]
        cands = [(q.inchikey14, q.normalized_smiles)] + list(zip(iso.inchikey14, iso.normalized_smiles))
        panels.append(dict(key=q.inchikey14, split=q.split, cands=cands))
    print('panels', len(panels), pd.Series([p['split'] for p in panels]).value_counts().to_dict(),
          'median size', np.median([len(p['cands']) for p in panels]), flush=True)
    # candidate targets
    ctg = {}
    for p in panels:
        for k, s in p['cands']:
            if k not in ctg:
                ctg[k] = chem.targets(s)
    rr8_pat = {n: Chem.MolFromSmarts(v[1]) for n, v in RR8_LOSSES.items()}
    rr8_has = {}
    for p in panels:
        for k, s in p['cands']:
            if k not in rr8_has:
                m = Chem.MolFromSmiles(s)
                rr8_has[k] = None if m is None else np.array([int(m.HasSubstructMatch(rr8_pat[n])) for n in RR8_LOSSES])

    # ---- train networks ----
    preds = {}
    rows_by_key = pd.Series(np.arange(len(keys))).groupby(keys).apply(list).to_dict()
    for fn, X in feats.items():
        for tn, Y in tgts.items():
            t0 = time.time()
            print(f'train nn_{fn}_{tn} X{X[fit].shape} Y{Y.shape[1]}', flush=True)
            net = train_mlp(X[fit], Y[fit])
            P = predict(net, X[~fit])
            idx = np.nonzero(~fit)[0]
            pos_of = {r: i for i, r in enumerate(idx)}
            preds[f'nn_{fn}_{tn}'] = {k: P[[pos_of[r] for r in rows_by_key[k]]].mean(0) for k in set(keys[~fit])}
            print(f'  done {time.time() - t0:.0f}s', flush=True)

    # ---- score panels ----
    results = []
    t0 = time.time()
    fcache = {}
    for n_p, p in enumerate(panels):
        rows = rows_by_key[p['key']]
        sub = df.iloc[rows]
        spectra = [dict(mz=np.asarray(r.ms2_mzs, float), it=np.asarray(r.ms2_normalized_intensities, float) /
                        max(np.max(r.ms2_normalized_intensities), 1e-12), adduct=r.adduct, prec=float(r.precursor_mz),
                        mode=1 if r.adduct.endswith('+') else -1) for r in sub.itertuples(index=False)]
        ck = [k for k, _ in p['cands']]
        cs = [s for _, s in p['cands']]
        valid = np.array([ctg[k] is not None for k in ck])
        S = {}
        S['random'] = np.zeros(len(ck))
        # rr8 loss agreement
        obs = np.zeros(len(RR8_LOSSES))
        for sp in spectra:
            mz, it = sp['mz'], sp['it']
            for j, (n, (mass, _)) in enumerate(RR8_LOSSES.items()):
                obs[j] += it[np.abs(mz - (sp['prec'] - mass)) <= 0.006].sum() / len(spectra)
        S['rr8_losses'] = np.array([(obs * (2 * h - 1)).sum() if h is not None else -np.inf
                                    for h in (rr8_has[k] for k in ck)])
        mf = frag.score_molecule(spectra, cs, cfg=None, cache=fcache)
        S['metfrag_lite'] = np.array([-np.inf if v is None else v for v in mf])
        if len(fcache) > 20000:
            fcache.clear()
        for name, pr in preds.items():
            P = pr[p['key']]
            tn = name.split('_')[-1]
            Yc = np.stack([np.concatenate([ctg[k]['np'], ctg[k]['maccs']]) if tn == 'chem' else ctg[k]['morgan']
                           if ctg[k] is not None else np.zeros(len(P)) for k in ck]).astype(float)
            S[name] = np.where(valid, loglik(P[None, :], Yc), -np.inf)
        rec = dict(key=p['key'], split=p['split'], n=len(ck), cand_in_fit=int(sum(k in fit_keys for k in ck[1:])))
        for name, s in S.items():
            rec['rank_' + name] = rank_of_true(s, 0)
        rec['scores'] = {name: [float(x) for x in s] for name, s in S.items()}
        results.append(rec)
        if n_p % 200 == 0:
            print(f'  scored {n_p}/{len(panels)} {time.time() - t0:.0f}s', flush=True)
    with open(work / 'rank_results.json', 'w') as fh:
        json.dump(results, fh)
    print('saved', len(results))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:]))
