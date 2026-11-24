"""Shared chemistry helpers: spectrum cleaning, high-resolution loss/fragment vocabularies, substructure targets."""
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*')

# Natural-product oriented functional groups / substructures (chemical prior targets).
NP_SMARTS = {
    'hydroxyl_any': '[OX2H]',
    'phenol': 'c[OX2H]',
    'catechol': 'c([OX2H])c[OX2H]',
    'pyrogallol': 'c([OX2H])c([OX2H])c[OX2H]',
    'methoxy_aryl': 'c[OX2][CH3]',
    'methoxy_any': '[OX2][CH3]',
    'methylenedioxy': 'c1cc2OCOc2cc1',
    'carboxylic_acid': '[CX3](=O)[OX2H1,OX1-]',
    'ester': '[#6][CX3](=O)[OX2][#6]',
    'lactone': '[#6][CX3;R](=O)[OX2;R][#6]',
    'acetyl_ester': '[CH3]C(=O)O[#6]',
    'amide': '[NX3][CX3](=O)[#6]',
    'primary_amine': '[NX3;H2;!$(NC=O)]',
    'sec_tert_amine': '[NX3;H1,H0;!$(NC=O);!$(N-a);!$(N=*)]([#6])[#6]',
    'aromatic_n': 'n',
    'indole': 'c1ccc2[nH]ccc2c1',
    'aldehyde': '[CX3H1](=O)[#6]',
    'ketone': '[#6][CX3](=O)[#6]',
    'enone': '[#6]=[#6][CX3](=O)[#6]',
    'hexose_ring': '[OX2][CX4;R]1[OX2;R][CX4;R]([CX4][OX2])[CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]',
    'deoxyhexose_ring': '[CH3][CX4;R]1[OX2;R][CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]',
    'pentose_ring': '[OX2][CX4;R]1[OX2;R][CX4;R][CX4;R]([OX2])[CX4;R]1[OX2]',
    'glucuronide': 'OC(=O)[CX4;R]1[OX2;R][CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]',
    'o_glycoside': '[#6;!$(C(O)(O))][OX2][CX4;R]1[OX2;R][CX4;R][CX4;R][CX4;R]1',
    'c_glycoside': 'c[CX4;R]1[OX2;R][CX4;R][CX4;R][CX4;R][CX4;R]1',
    'flavone_core': 'O=c1cc(-c2ccccc2)oc2ccccc12',
    'flavonoid_c6c3c6': '[#6]1~[#6]~[#6](~[#6]2~[#6]~[#6]~[#6]~[#6]~[#6]~2)~[#8]~[#6]2~[#6]~[#6]~[#6]~[#6]~[#6]~2~1',
    'coumarin': 'O=c1ccc2ccccc2o1',
    'cinnamoyl': 'c/C=C/C(=O)[O,N]',
    'cinnamoyl_any': 'cC=CC(=O)[O,N]',
    'galloyl': 'O=C(O)c1cc(O)c(O)c(O)c1',
    'prenyl': '[CH3]C([CH3])=[CH][CH2]',
    'steroid_core': 'C1CCC2C(C1)CCC1C2CCC2CCCC12',
    'sulfate_sulfonate': 'S(=O)(=O)[OX2,OX1-]',
    'phosphate': 'P(=O)([OX2,OX1-])[OX2,OX1-]',
    'nitro': '[N+](=O)[O-]',
    'halogen': '[F,Cl,Br,I]',
    'thioether': '[#6][SX2][#6]',
    'epoxide': 'C1OC1',
    'terminal_alkene': '[CH2]=[CX3]',
    'quaternary_n': '[NX4+]',
    'alkaloid_n_ring': '[NX3;R;!$(NC=O)]',
}
_PATTS = {k: Chem.MolFromSmarts(v) for k, v in NP_SMARTS.items()}
assert all(p is not None for p in _PATTS.values()), [k for k, p in _PATTS.items() if p is None]
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
PROTON = 1.00727646688


def np_groups(mol):
    return np.array([int(mol.HasSubstructMatch(p)) for p in _PATTS.values()], np.uint8)


def maccs(mol):
    a = np.zeros(167, np.uint8)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), a)
    return a[1:]


def targets(smiles):
    """dict of target vectors for one SMILES, or None if unparsable."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return {'np': np_groups(mol), 'maccs': maccs(mol), 'morgan': MORGAN.GetFingerprintAsNumPy(mol).astype(np.uint8)}


def clean(mzs, its, prec, min_rel=0.005):
    mz = np.asarray(mzs, float)
    it = np.asarray(its, float)
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz < prec - 0.5)  # drop precursor region
    mz, it = mz[ok], it[ok]
    if len(it) == 0:
        return mz, it
    it = it / it.max()
    keep = it >= min_rel
    return mz[keep], it[keep]


def mine_vocab(values_per_mol, lo, hi, res=0.001, smooth=3, min_frac=0.003, max_n=400, min_sep=0.008):
    """Data-driven peak picking on a molecule-count histogram of high-resolution masses.

    values_per_mol: list of arrays (unique-per-molecule masses). Returns array of centre masses ordered by support.
    """
    nb = int(round((hi - lo) / res)) + 1
    h = np.zeros(nb)
    for v in values_per_mol:
        v = v[(v >= lo) & (v < hi)]
        idx = np.unique(np.round((v - lo) / res).astype(int))
        h[idx] += 1
    k = np.ones(2 * smooth + 1)
    hs = np.convolve(h, k, mode='same')
    n = len(values_per_mol)
    order = np.argsort(-hs)
    centres, taken = [], np.zeros(nb, bool)
    sep = int(round(min_sep / res))
    for i in order:
        if hs[i] < min_frac * n or len(centres) >= max_n:
            break
        if taken[i]:
            continue
        # centroid within the smoothing window
        w = slice(max(0, i - smooth), i + smooth + 1)
        x = lo + np.arange(nb)[w] * res
        c = float((x * h[w]).sum() / max(h[w].sum(), 1e-9))
        centres.append((c, hs[i] / n))
        taken[max(0, i - sep):i + sep + 1] = True
    return centres


def vocab_features(mz, it, prec, loss_vocab, frag_vocab, tol=0.004):
    """sqrt-intensity-weighted presence of each vocabulary loss / fragment (max over matching peaks)."""
    w = np.sqrt(it)
    out = np.zeros(len(loss_vocab) + len(frag_vocab), np.float32)
    if len(mz) == 0:
        return out
    loss = prec - mz
    for arr, vocab, off in ((loss, loss_vocab, 0), (mz, frag_vocab, len(loss_vocab))):
        o = np.argsort(arr)
        a = arr[o]
        lo = np.searchsorted(a, vocab - tol)
        hi = np.searchsorted(a, vocab + tol)
        for j in np.nonzero(hi > lo)[0]:
            out[off + j] = w[o[lo[j]:hi[j]]].max()
    return out


def int_features(mz, it, prec, nbin=1000):
    """Integer-bin m/z and neutral-loss histograms (the representation the repo's casmi_ml models use)."""
    w = np.sqrt(it)
    h = np.zeros(nbin, np.float32)
    l = np.zeros(nbin, np.float32)
    m = mz < nbin
    np.maximum.at(h, mz[m].astype(int), w[m])
    loss = prec - mz
    m = (loss >= 0) & (loss < nbin)
    np.maximum.at(l, loss[m].astype(int), w[m])
    h /= max(np.linalg.norm(h), 1e-9)
    l /= max(np.linalg.norm(l), 1e-9)
    return np.concatenate([h, l])
