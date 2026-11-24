"""Mine high-resolution neutral-loss / fragment vocabularies on the 'fit' split, measure loss->substructure lift,
and cache feature matrices + targets for both splits.

Usage: python mine.py WORK_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: F401  (kept for interactive use)

sys.path.insert(0, str(Path(__file__).parent))
import chem  # noqa: E402

# Known losses for annotation of the mined vocabulary (monoisotopic, Da)
KNOWN = {'H2O': 18.010565, 'NH3': 17.026549, 'CO': 27.994915, 'C2H4': 28.0313, 'CO2': 43.989829,
         'HCOOH': 46.005479, 'CH3 (radical)': 15.023475, 'CH4O (MeOH)': 32.026215, 'C2H2O (ketene)': 42.010565,
         'C2H4O2 (AcOH)': 60.021129, 'C3H6 ': 42.04695, 'C4H8 (prenyl-H)': 56.0626, 'C5H8 (isoprene)': 68.0626,
         'HCN': 27.010899, 'CH5N (methylamine)': 31.042199, 'C2H7N (dimethylamine)': 45.057849,
         'C3H9N (trimethylamine)': 59.073499, 'H2O+CO': 46.005480, '2H2O': 36.02113, 'SO3': 79.956815,
         'H3PO4': 97.976895, 'HPO3': 79.966331, 'H2S': 33.987721, 'HCl': 35.976678,
         'C6H10O5 (hexose)': 162.052824, 'C6H10O4 (deoxyhexose)': 146.057909, 'C5H8O4 (pentose)': 132.042259,
         'C6H8O6 (glucuronic)': 176.032088, 'C6H12O6 (hexose+H2O)': 180.063388, 'C9H6O2 (coumaroyl)': 146.036779,
         'C9H6O3 (caffeoyl)': 162.031694, 'C10H8O3 (feruloyl)': 176.047344, 'C7H4O4 (galloyl)': 152.010959,
         'C3H2O3 (malonyl)': 86.000394, 'C2H2O+H2O': 60.021129, 'CH2O (formaldehyde)': 30.010565,
         'C2H5NO2 (glycine)': 75.032028, 'C6H12O5 (deoxyhexose+H2O)': 164.068473, 'C12H20O9 (2 hexose-... )': 308.110732,
         'C12H20O10 (2 hexose)': 324.105649}


def annotate(m, tol=0.003):
    best = min(KNOWN.items(), key=lambda kv: abs(kv[1] - m))
    return best[0] if abs(best[1] - m) <= tol else ''


def main(work):
    work = Path(work)
    df = pd.read_parquet(work / 'spectra.parquet')
    # targets per molecule
    mols = df.drop_duplicates('inchikey14')[['inchikey14', 'normalized_smiles', 'molecular_formula', 'split']]
    tg = {k: chem.targets(s) for k, s in zip(mols.inchikey14, mols.normalized_smiles)}
    bad = {k for k, v in tg.items() if v is None}
    df = df[~df.inchikey14.isin(bad)].reset_index(drop=True)
    mols = mols[~mols.inchikey14.isin(bad)]
    print('molecules', len(mols), 'unparsable', len(bad))
    cleaned = [chem.clean(m, i, p) for m, i, p in zip(df.ms2_mzs, df.ms2_normalized_intensities, df.precursor_mz)]
    fit = (df.split == 'fit').values
    pos = (df.adduct == '[M+H]+').values
    nat = (df.ingest_lib != 'enveda-180').values  # vocabulary + lift are mined on natural-product-like libraries
    # vocab mined on fit/positive, per molecule (union over its spectra)
    per_mol_loss, per_mol_frag = {}, {}
    for i in np.nonzero(fit & pos & nat)[0]:
        mz, it = cleaned[i]
        k = df.inchikey14[i]
        per_mol_loss.setdefault(k, []).append(df.precursor_mz[i] - mz)
        per_mol_frag.setdefault(k, []).append(mz)
    L = [np.concatenate(v) for v in per_mol_loss.values()]
    F = [np.concatenate(v) for v in per_mol_frag.values()]
    loss_vocab = chem.mine_vocab(L, 2.0, 500.0, max_n=400)
    frag_vocab = chem.mine_vocab(F, 40.0, 500.0, max_n=400)
    lv = np.array([c for c, _ in loss_vocab])
    fv = np.array([c for c, _ in frag_vocab])
    print('vocab sizes', len(lv), len(fv))
    # negative-mode losses too (few spectra; small vocab)
    negL = []
    for i in np.nonzero(fit & ~pos & nat)[0]:
        mz, it = cleaned[i]
        negL.append(df.precursor_mz[i] - mz)
    neg_vocab = chem.mine_vocab(negL, 2.0, 500.0, max_n=150, min_frac=0.005)
    nv = np.array([c for c, _ in neg_vocab])
    json.dump({'loss_pos': loss_vocab, 'frag_pos': frag_vocab, 'loss_neg': neg_vocab},
              open(work / 'vocab.json', 'w'), indent=1)

    # features
    Xint = np.stack([chem.int_features(mz, it, p) for (mz, it), p in zip(cleaned, df.precursor_mz)])
    Xhr = np.stack([chem.vocab_features(mz, it, p, lv, fv) if a == '[M+H]+' else
                    np.concatenate([chem.vocab_features(mz, it, p, nv, np.zeros(0)),
                                    np.zeros(len(lv) + len(fv) - len(nv), np.float32)])
                    for (mz, it), p, a in zip(cleaned, df.precursor_mz, df.adduct)])
    # note: negative-mode rows reuse the first len(nv) columns with a separate vocabulary; a mode flag disambiguates
    mode = (~pos).astype(np.float32)[:, None]
    Y = {t: np.stack([tg[k][t] for k in df.inchikey14]) for t in ('np', 'maccs', 'morgan')}
    np.savez_compressed(work / 'features.npz', Xint=Xint.astype(np.float16), Xhr=Xhr.astype(np.float16), mode=mode,
                        fit=fit, pos=pos, nat=nat, split=df.split.values.astype(str), keys=df.inchikey14.values.astype(str), **{f'Y_{t}': v for t, v in Y.items()})

    # ---- lift analysis: P(group | loss observed) / P(group), molecule level, fit vs eval replication ----
    groups = list(chem.NP_SMARTS)
    G = Y['np']
    rows = []
    loss_present = (Xhr[:, :len(lv)] > np.sqrt(0.02)).astype(np.uint8)  # peak >= 2 % rel. intensity
    for split_name, msk in (('fit', fit & pos & nat), ('eval', ~fit & pos)):
        keys = df.inchikey14.values[msk]
        lp = pd.DataFrame(loss_present[msk]).groupby(keys).max()
        gp = pd.DataFrame(G[msk]).groupby(keys).max().loc[lp.index]
        lpv, gpv = lp.values.astype(float), gp.values.astype(float)
        n = len(lpv)
        base = gpv.mean(0)
        both = lpv.T @ gpv  # [loss, group]
        nl = lpv.sum(0)
        for j in range(len(lv)):
            for g in range(len(groups)):
                if nl[j] < 1:
                    continue
                rows.append((split_name, j, lv[j], g, groups[g], nl[j], both[j, g], base[g],
                             both[j, g] / nl[j]))
    lift = pd.DataFrame(rows, columns=['split', 'j', 'loss', 'g', 'group', 'n_loss', 'n_both', 'p_group', 'p_group_given_loss'])
    lift['lift'] = lift.p_group_given_loss / lift.p_group.clip(lower=1e-9)
    w = lift.pivot_table(index=['j', 'loss', 'group'], columns='split', values=['lift', 'n_loss', 'n_both', 'p_group_given_loss', 'p_group'])
    w.columns = [f'{a}_{b}' for a, b in w.columns]
    w = w.reset_index()
    w['annotation'] = w.loss.map(annotate)
    w.to_csv(work / 'lift_all.csv', index=False)
    strong = w[(w.n_loss_fit >= 50) & (w.n_both_fit >= 20) & (w.lift_fit >= 3) & (w.n_loss_eval >= 20)]
    strong = strong.sort_values('lift_fit', ascending=False)
    strong.to_csv(work / 'lift_strong.csv', index=False)
    # replication: shrunken eval lift (add-one smoothing) vs fit lift
    ev_lift = (strong.n_both_eval + strong.p_group_eval) / (strong.n_loss_eval + 1) / strong.p_group_eval.clip(lower=1e-9)
    r = np.corrcoef(np.log(strong.lift_fit), np.log(ev_lift))[0, 1] if len(strong) > 2 else float('nan')
    rep = float((ev_lift > 1.5).mean()) if len(strong) else float('nan')
    print('strong associations', len(strong), 'log-lift fit/eval corr', round(r, 3), 'replicated (eval lift>1.5)', round(rep, 3))
    top = pd.DataFrame(loss_vocab[:40], columns=['loss', 'frac_molecules'])
    top['annotation'] = top.loss.map(annotate)
    print(top.to_string())


if __name__ == '__main__':
    main(sys.argv[1])
