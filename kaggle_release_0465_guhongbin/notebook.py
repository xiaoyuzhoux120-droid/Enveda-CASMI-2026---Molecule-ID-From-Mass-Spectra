# %% [markdown]
# CASMI26 v15 (v14 + N1A GLACIER proxies)
#
# Our v14 (LB 0.455) with N1A from Seyit Kaan Gunes (seyitkaangunes/casmi26-e006-n1a-cc-by-chebi-lb-0-452): molecules without an [M+H]+ spectrum get GLACIER scores from their [M+NH4]+ / [M-H2O+H]+ spectra converted to [M+H]+ proxies. Our v13 (LB 0.451) with engine 2's candidate pool extended by the PubChemLite CC0 structures of franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints, as in guhongbin/casmi26-v447-pubchemlite (Hongbin Gu). Our v12-sync (v11 + metric-identity E2R2) with the engine-2 ChEBI / LIPID MAPS candidate structures taken from sergiomalv/casmi-chebi-lmsd-structures (CC BY 4.0; ChEBI, EMBL-EBI; LIPID MAPS LMSD) instead of prvsiyan/chebi-lipidmaps-casmi26, as in the public 0.451 notebooks (easonyanyan, saddamaditya). Our v11 (LB 0.446) with the metric-identity E2R2 rank-2 cell of the public 0.451 notebooks (easonyanyan, saddamaditya; idea antriksharya) and no rule (base sync). Our v7c (LB 0.441) with the engine-2 rank-2 hedge of antriksharya/casmi26-e006-e2rank2 (verbatim final cell) and the PubChem promotion threshold rel > 150 (seyitkaangunes/casmi26-e006-tuned-pubchem-promo-lb-0-444 sweep). Credits: antriksharya, seyitkaangunes, canqiang, samanyu1808, flexonafft, obstacledeveloper, lszlst, Zenith, megayak, prvsiyan, dmitriigluzdov and our own casmi26 datasets. Our v7a (faithful port of the public E006 notebook, LB 0.440) plus one change: molecules with no forward-model evidence at all (no GLACIER, no ICEBERG coverage) get a MetFrag-lite fragment re-rank of their fused list (lambda 0.5, Huseyin Emre Aksoy's frag_rescore.py, Apache-2.0). Faithful port of canqiang's public "E006 window 40 ICE fallback" notebook (public LB 0.440) to our account, same pinned Python 3.12 T4 image. Only change: the optional DreaMS mass-tail dataset and the third-party RDKit wheel dataset are not attached (the simulator RDKit 2025.3.6 cp312 wheel ships inside casmi26-iceberg). Credits: canqiang (E006), flexonafft (metric-identity fusion v80), samanyu1808 (ICEBERG where GLACIER is blind), obstacledeveloper (metric audit and safe tail), bobthebot369 (Zenith), lszlst (GLACIER-only ablations), skomuro (promotion rule), prvsiyan/megayak (engine-2 assets), Dmitrii Gluzdov (popularity arrays), ms-pred (ICEBERG/GLACIER), and our own v4n engine/datasets.


# %% [markdown]
# E004: exact v80 public recipe reproduction
#
# Source: flexonafft/casmi26-public-baseline-adaptation-experiments v80, script356740494. Author submission57008151 scored0.436; this is source evidence, not our score. All 27 source code cells were recovered from that exact version's public rendered input cells. Their combined SHA256 is pinned and their executable AST was independently compared with the v82 source prefix (the v82 consensus overlay is excluded). The original code, GPU T4, image digest and 16 public dataset versions are preserved. Two function-local observers check the input/runtime and save diagnostic artifacts without changing predictions. This is a whole-recipe reproduction, not a single-factor comparison against E003. Preserve upstream attribution and component licences; see SOURCE.md.


# %% [code]

'''CASMI26: Metric-Identity Fusion — Aggressive Experiment
One vote per official metric identity per engine; compact unique ranks before the 40-candidate fusion shortlist. Preserve the existing PubChem gate and same-formula GLACIER logic. This tests duplicate-vote bias, not a coefficient search. It can change top-1 and reduce the score.
Run the complete repaired @obstacledeveloper adaptation first, then this independent overlay. submission_reference.csv is the completed baseline from the same run; experiment_audit.csv/json record changes and file hashes. This is not stacked with the other experiment. Same 17 source inputs and GPU environment; no additional dataset required. No new public score has been measured. A visible smoke run is not validation.'''


    


# %% [code]

'''Credits & Attribution
@obstacledeveloper — direct source: CASMI26 V1.1 Metric Audit and Safe Tail, version 1, scriptVersionId 356393001. Source: https://www.kaggle.com/code/obstacledeveloper/casmi26-v1-1-metric-audit-and-safe-tail?scriptVersionId=356393001
The source page displays public score 0.433; this adaptation has not reproduced that result.
@bobthebot369 — inherited Sovereign Zenith ensemble and confidence-promotion ideas: https://www.kaggle.com/code/bobthebot369/enveda-casmi-2026-v18-sovereign-zenith/notebook
@ahmedberatozer — engine, ranker, pool, PubChem, full FPNet and simulator packages: https://www.kaggle.com/ahmedberatozer
@prvsiyan — analog propagation, spectral matching and fingerprint models: https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline
@dmitriigluzdov — PubChem popularity prior: https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior
@hengck23 — inherited hengck23-dreams-enveda-casmi26 structural candidate resource: https://www.kaggle.com/hengck23
Official metric implementation, version 14: https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank
ICEBERG/GLACIER research tooling: https://github.com/coleygroup/ms-pred
Competition: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra
RDKit official metric wheel: https://www.kaggle.com/datasets/metric/rdkit-2026-3-3-wheel
ChEBI/LIPID MAPS and COCONUT resources remain credited in inherited code and input metadata.
The public notebook declares Apache-2.0; upstream datasets/components retain their own licences. Substantial code, including embedded modules, is reused verbatim. Our adaptation changes presentation and adds provenance, not authorship of the engines. Upstream comments are retained as provenance, not independently established component-performance claims.'''


    


# %% [code]

'''Experiment provenance and reasoning
Direct starting point: @obstacledeveloper, linked in Credits & Attribution below. The upstream engine/fusion/simulator code is substantially reused, not claimed as original. Our contribution here is the separately audited overlay described above. Previous analog expansion, top-1 shielding, adduct-specific merging and single-pass variants did not improve our confirmed 0.420 result. These experiments test structural ranking changes instead of repeating those parameter changes. Metric identity uses the inherited official canonicalization and heavy-atom composition helpers. GLACIER tooling: https://github.com/coleygroup/ms-pred ; research: https://arxiv.org/abs/2606.29161 . Global cross-formula ranking is our unvalidated hypothesis, not a claim made by that paper. Public leaderboard feedback does not establish private-set generalization.'''


    


# %% [code]

'''# CASMI26 V1.1 — metric audit and safe tail

Starting point: this account's V1 submission, public MRR **0.425** on 2026-10-08.
This is a conservative candidate-coverage experiment, not a claimed leaderboard improvement.

Changes: pin official RDKit 2026.03.3; preserve the V1 inference/ranking pipeline;
deduplicate only after ranking using the official metric identity; append ranked overflow
and mass-compatible DreaMS candidates into vacant slots; assert every original matchable
identity has equal or better rank. The old DreaMS branch was bypassed by `NameError: Path`.

No hidden-test labels, manual labels, or network access are used. The visible test overlaps
training data, so its score is an operational check and is **not** a validation estimate.

Source attribution: copied V1 ensemble from BobtheBot369 / Ahmed Berat Özer / prvsiyan
and the other authors credited in its code. Scoring reference: Kaggle Competition Metrics,
`https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank`, version 14, Apache-2.0.
'''


    


# %% [code]

'''## 1. Setup

Locates every dataset, installs the competition's RDKit wheel, sets the popularity-prior parameters for the PubChem channel (`POP_LAM = 0.25`, `POP_UNION = 200`) and checks the v4b model files against their manifest.
'''


    


# %% [code]

'''CASMI26 v4n = v4i + ICE_LAM = GL_LAM = 1.0 (v4l, LB 0.380) + engine FP bank fpnet_full1 (v4m, LB 0.378)'''
'''CASMI26 v4i = v4h + PubChem-only channel pass-1 depth N1 1000 -> 5000 (recall; gate/slots unchanged)'''
'''CASMI26 v4h = v4g + GLACIER third re-scoring term: z(ranker) + 0.5 z(ICEBERG) + 0.5 z(GLACIER) (casmi26-glacier)'''
'''casmi26 v28 = v17 + pool popularity re-rank of the base list (mu 0.15, mode A); v17 = Ahmed v4g + popularity prior in the PubChem list (see make_v17.py)'''
'''CASMI26 v4f = v4b (FPNet A engine bank, LB 0.354) + ICEBERG post-ranker isomer re-scoring = the v3.6 model (v1 engine + FPNet A + moderate generation + gated PubChem channel, LB 0.358)'''
'''+ fe_v4 feature families computed on the SAME v1 engine run (derivation evidence / class-3 priors, fragmentation 2.0,'''
'''analog-structure relations, DreamsFP views) + a ranker retrained on the v1-engine simulation sim1x with them.'''
'''Built by work/analysis/fe_v4/build_v4v1.py -- edit there.'''
import os, sys, glob, re, subprocess, time, json, hashlib, pickle
from pathlib import Path
T0 = time.time()
LIB_TAU, REL_TH = 0.9, 600.0
TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 5400, True
SLOTS_AGG, SLOTS_GENTLE = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20]
os.makedirs('/kaggle/working/numba_cache', exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = '/kaggle/working/numba_cache'
def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]
def sha256(path, n=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(n), b''):
            h.update(b)
    return h.hexdigest()
POP_LAM, POP_UNION = 0.15, 200            # v17: popularity prior for the PubChem-only list
os.environ['CASMI_POP_DIR'] = os.path.dirname(find('pc_lsid.npy')); os.environ['CASMI_POP_LAM'] = str(POP_LAM); os.environ['CASMI_POP_UNION'] = str(POP_UNION)
print('v17 popularity prior', POP_LAM, POP_UNION, os.environ['CASMI_POP_DIR'])
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
whl = [w for w in glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True) if tag in w and Path(w).name.startswith('rdkit-2026.3.3-')]
assert whl, 'Official RDKit 2026.3.3 wheel is missing'
print('rdkit wheel:', whl)
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-500:])
r.check_returncode()
import rdkit; print('rdkit', rdkit.__version__)
assert rdkit.__version__ == '2026.03.3'
V4CODE = os.path.dirname(os.path.dirname(find('casmi26-v4b-models/**/casmi/engine.py')))
V4MOD = os.path.dirname(find('casmi26-v4b-models/**/MANIFEST.json'))
V3CODE = os.path.dirname(os.path.dirname(find('casmi26-v3-models/**/casmi/engine.py')))
V3MOD = os.path.dirname(find('casmi26-v3-models/**/ranker_0.pkl'))
POOL_DIR = os.path.dirname(find('pool_meta.parquet'))
PC_DIR = os.path.dirname(find('pc_smiles.npy'))
COMP = os.path.dirname(find('test.parquet'))
MAN = json.load(open(os.path.join(V4MOD, 'MANIFEST.json')))
bad = []
for rel, h in sorted(MAN['files'].items()):
    p = os.path.join(V4CODE, rel[len('code/'):]) if rel.startswith('code/') else os.path.join(V4MOD, rel)
    if not os.path.exists(p) or sha256(p) != h:
        bad.append(rel)
assert not bad, f'dataset files differ from MANIFEST.json: {bad}'
print('V4CODE', V4CODE, '\nV4MOD', V4MOD, '\nV3CODE', V3CODE, '\nPOOL', POOL_DIR, '\nPC', PC_DIR, '\nCOMP', COMP)
print(f"{len(MAN['files'])} files match MANIFEST.json; families {MAN['families']}", f'{time.time()-T0:.0f}s')

'''Compatibility repair: select the simulator RDKit 2025.3.6 wheel by interpreter and platform tags, including bundled .whl.ice files. Fail before inference when no compatible build exists; preserve the main RDKit 2026.03.3 environment.'''

def simulator_rdkit_wheel(pkg):
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    compatible = set(sys_tags())
    bundled = sorted(glob.glob(os.path.join(pkg, 'wheels', 'rdkit-2025.3.6-*.whl*')))
    external = sorted(glob.glob('/kaggle/input/**/rdkit-2025.3.6-*.whl', recursive=True))
    for path in bundled + external:
        name = os.path.basename(path).removesuffix('.ice')
        _, version, _, tags = parse_wheel_filename(name)
        if str(version) == '2025.3.6' and compatible.intersection(tags):
            return path
    raise FileNotFoundError(
        f'RDKit 2025.3.6 wheel compatible with Python {sys.version_info.major}.{sys.version_info.minor} '
        f'and this platform is required for ICEBERG/GLACIER; candidates: {bundled + external}')

_simulator_pkg = os.path.dirname(find('casmi26-iceberg/**/ice_runner.py'))
print('Simulator RDKit wheel:', simulator_rdkit_wheel(_simulator_pkg))


    


# %% [code]

"""Read-only E004 observer, before the source smoke switch."""


def _e004_preflight(scope):
    import json
    import os
    import platform
    import subprocess
    import pandas as pd
    import rdkit

    slug = 'enveda-CASMI26-molecule-id-mass-spectra'
    comp = os.path.realpath(scope['COMP'])
    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}
    assert platform.python_version_tuple()[:2] == ('3', '12')
    assert rdkit.__version__ == '2026.03.3'
    # Do not initialize a parent CUDA context before the source's Engine2 subprocess.
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    assert 'T4' in gpu, gpu
    ids = sorted(set(pd.read_parquet(os.path.join(comp, 'test.parquet'),
                                    columns=['molecule_id']).molecule_id.astype(str)))
    sample = pd.read_csv(os.path.join(comp, 'sample_submission.csv'), dtype=str)
    assert ids and set(sample.molecule_id) == set(ids)
    result = dict(runtime_input_path=comp, input_ids=ids, python=platform.python_version(),
                  rdkit=rdkit.__version__, gpu=gpu)
    print('E004_PREFLIGHT', json.dumps({k: v for k, v in result.items() if k != 'input_ids'}))
    return result


E004_PREFLIGHT = _e004_preflight(globals())
del _e004_preflight


    


# %% [code]

'''## 2. Commit switch

When the visible placeholder test is detected, the notebook runs only 12 molecules with a short ICEBERG budget, so saving a version is quick. The scoring rerun on the hidden test runs everything. Nothing else depends on this switch.
'''


    


# %% [code]

'''SMOKE switch: commits (dummy test) run 12 molecules; the scoring rerun (hidden test) runs everything.'''
import pandas as pd, hashlib
_te = pd.read_parquet(os.path.join(COMP, 'test.parquet'), columns=['molecule_id', 'spectrum_id'])
_sig = hashlib.md5(','.join(sorted(_te.spectrum_id.astype(str))).encode()).hexdigest()
IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or _sig != '654e030e4173780e945252b4a5adf70b'
SMOKE_N = 12
if not IS_RERUN:
    SM = '/kaggle/working/smoke_comp'; os.makedirs(SM, exist_ok=True)
    _full = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
    _keep = sorted(_full.molecule_id.unique())[:SMOKE_N]
    _full[_full.molecule_id.isin(_keep)].to_parquet(os.path.join(SM, 'test.parquet'))
    for _f in ['train.parquet', 'sample_submission.csv']:
        if not os.path.exists(os.path.join(SM, _f)):
            os.symlink(os.path.join(COMP, _f), os.path.join(SM, _f))
    COMP = SM
    ICE_BUDGET = 300
print('RERUN', IS_RERUN, '| sig', _sig, '| COMP', COMP, '| ICE_BUDGET', ICE_BUDGET)


    


# %% [code]

'''## 3. Engine 2: two-ranker engine (BIO + AFIX)

This engine runs first in its own process, while GPU memory is still empty. It writes a top-40 list per molecule with tautomer-canonical InChIKey-14 keys.

- **Candidates** come from COCONUT, ChEBI/LIPID MAPS (BIO) and the training structures.
- **Evidence** comes from library matching (including adduct-shifted matches), analog propagation, prvsiyan's fingerprint models and MetFrag-lite fragment matching.
- **Ranking** blends two gradient-boosted rankers.
'''


    


# %% [code]

'''Our engine (two-ranker + BIO + AFIX) in a subprocess, then weighted reciprocal-rank fusion with the v4g lists.'''
ENG_DIR = '/kaggle/working/eng'; os.makedirs(ENG_DIR, exist_ok=True)
ENG_FILES = {'pv.py': '"""prvsiyan analog-propagation pipeline components (public notebook, version of 2026-09-16), copied\nverbatim where possible so local simulations and the Kaggle notebook share one source of truth.\nAdditions are marked `# ADDED`.\n"""\nimport math\nimport numpy as np\nfrom numba import njit, prange\n\n\nclass CFG:\n    PPM_WIN = 10.0\n    PPM_FALLBACK = 30.0\n    INT_FLOOR = 0.002\n    MAX_PEAKS = 256\n    MZ_TOL = 0.01\n    INT_POWER = 1.0\n    ENT_WEIGHT = True\n    ANALOG_WIN = 200.0\n    N_ANALOG = 80\n    SIM_POWER = 3.0\n    W1_PRIORS = (0.30, 0.60)\n    SEEDS = (0, 1, 2, 3)\n    GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03, min_samples_leaf=80, l2_regularization=1.0)\n    TOPN = 25\n\n\n# ----------------------------------------------------------------------------------------------\n# similarity kernels\n# ----------------------------------------------------------------------------------------------\n@njit(cache=True, fastmath=True)\ndef _clean(mz, it, floor, topk, power, ent_weight):\n    n = len(mz)\n    if n == 0: return np.empty(0, np.float32), np.empty(0, np.float32)\n    mx = 0.0\n    for i in range(n):\n        if it[i] > mx: mx = it[i]\n    if mx <= 0: return np.empty(0, np.float32), np.empty(0, np.float32)\n    thr = floor * mx; c = 0\n    for i in range(n):\n        if it[i] >= thr: c += 1\n    idx = np.empty(c, np.int64); j = 0\n    for i in range(n):\n        if it[i] >= thr: idx[j] = i; j += 1\n    if c > topk:\n        v = np.empty(c, np.float32)\n        for i in range(c): v[i] = it[idx[i]]\n        o = np.argsort(v)[c - topk:]\n        k2 = np.empty(topk, np.int64)\n        for i in range(topk): k2[i] = idx[o[i]]\n        k2.sort(); idx = k2; c = topk\n    om = np.empty(c, np.float32); oi = np.empty(c, np.float32)\n    s = 0.0\n    for i in range(c):\n        om[i] = mz[idx[i]]; v = it[idx[i]] ** power; oi[i] = v; s += v\n    if s > 0:\n        for i in range(c): oi[i] /= s\n    if ent_weight:\n        S = 0.0\n        for i in range(c):\n            if oi[i] > 0: S -= oi[i] * np.log(oi[i])\n        if S < 3.0:\n            w = 0.25 + 0.25 * S; s2 = 0.0\n            for i in range(c): oi[i] = oi[i] ** w; s2 += oi[i]\n            if s2 > 0:\n                for i in range(c): oi[i] /= s2\n    return om, oi\n\n\n@njit(cache=True, fastmath=True)\ndef entropy_sim(qmz, qp, cmz, cp, tol):\n    i = 0; j = 0; n = len(qmz); m = len(cmz)\n    SA = 0.0\n    for x in range(n):\n        if qp[x] > 0: SA -= qp[x] * np.log(qp[x])\n    SB = 0.0\n    for x in range(m):\n        if cp[x] > 0: SB -= cp[x] * np.log(cp[x])\n    SAB = 0.0; tot = 0.0\n    buf = np.empty(n + m, np.float64); b = 0\n    while i < n and j < m:\n        d = qmz[i] - cmz[j]\n        if d < -tol: buf[b] = qp[i]; i += 1; b += 1\n        elif d > tol: buf[b] = cp[j]; j += 1; b += 1\n        else: buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1\n    while i < n: buf[b] = qp[i]; i += 1; b += 1\n    while j < m: buf[b] = cp[j]; j += 1; b += 1\n    for x in range(b): tot += buf[x]\n    if tot <= 0: return 0.0\n    for x in range(b):\n        v = buf[x] / tot\n        if v > 0: SAB -= v * np.log(v)\n    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)\n\n\n@njit(cache=True, fastmath=True, parallel=True)\ndef search(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight):\n    out = np.zeros(len(cand), np.float32)\n    for k in prange(len(cand)):\n        c = cand[k]; a = off[c]; b = off[c + 1]\n        if b <= a: continue\n        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)\n        if len(cm) == 0: continue\n        out[k] = entropy_sim(qmz, qp, cm, cp, tol)\n    return out\n\n\n@njit(cache=True, fastmath=True)\ndef entropy_sim_shift(qmz, qp, cmz, cp, tol, shift):\n    a = entropy_sim(qmz, qp, cmz, cp, tol)\n    if shift > -0.001 and shift < 0.001: return a\n    sm = np.empty(len(cmz), np.float32)\n    for i in range(len(cmz)): sm[i] = cmz[i] + shift\n    b = entropy_sim(qmz, qp, sm, cp, tol)\n    return a if a > b else b\n\n\n# ADDED: search over a pre-cleaned store (reps cleaned once instead of on every call)\n@njit(cache=True, fastmath=True, parallel=True)\ndef search_shift_pre(qmz, qp, lo, hi, roff, rmz, rit, tol, shift):\n    out = np.zeros(hi - lo, np.float32)\n    for k in prange(hi - lo):\n        a = roff[lo + k]; b = roff[lo + k + 1]\n        if b <= a: continue\n        out[k] = entropy_sim_shift(qmz, qp, rmz[a:b], rit[a:b], tol, shift[k])\n    return out\n\n\n# ADDED: library search where each reference is shifted by (query precursor - reference precursor):\n# the same molecule measured as a different adduct keeps its neutral losses (modified-cosine idea)\n@njit(cache=True, fastmath=True, parallel=True)\ndef search_shift_rows(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight, shift):\n    out = np.zeros(len(cand), np.float32)\n    for k in prange(len(cand)):\n        c = cand[k]; a = off[c]; b = off[c + 1]\n        if b <= a: continue\n        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)\n        if len(cm) == 0: continue\n        out[k] = entropy_sim_shift(qmz, qp, cm, cp, tol, shift[k])\n    return out\n\n\n# ADDED: clean a list of spectra into a flat store\n@njit(cache=True)\ndef clean_store(rows, off, allmz, allin, floor, topk, power, ent_weight):\n    n = len(rows)\n    lens = np.zeros(n + 1, np.int64)\n    tmp_m = []\n    tmp_i = []\n    for k in range(n):\n        r = rows[k]\n        cm, cp = _clean(allmz[off[r]:off[r + 1]], allin[off[r]:off[r + 1]], floor, topk, power, ent_weight)\n        tmp_m.append(cm); tmp_i.append(cp)\n        lens[k + 1] = lens[k] + len(cm)\n    om = np.empty(lens[n], np.float32); oi = np.empty(lens[n], np.float32)\n    for k in range(n):\n        om[lens[k]:lens[k + 1]] = tmp_m[k]; oi[lens[k]:lens[k + 1]] = tmp_i[k]\n    return lens, om, oi\n\n\n# ----------------------------------------------------------------------------------------------\n# adducts\n# ----------------------------------------------------------------------------------------------\nMASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163,\n            S=31.97207100, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473,\n            Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213)\nE = 0.00054857990; PROTON = MASS[\'H\'] - E; H2O = 2 * MASS[\'H\'] + MASS[\'O\']\nNH4 = MASS[\'N\'] + 4 * MASS[\'H\']; FORMATE = MASS[\'C\'] + 2 * MASS[\'H\'] + 2 * MASS[\'O\']\nACETATE = 2 * MASS[\'C\'] + 4 * MASS[\'H\'] + 2 * MASS[\'O\']\nADDUCTS = {\n    "[M+H]+": (1, 1, PROTON), "[M+NH4]+": (1, 1, NH4 - E), "[M+Na]+": (1, 1, MASS[\'Na\'] - E),\n    "[M+K]+": (1, 1, MASS[\'K\'] - E), "[M-H2O+H]+": (1, 1, PROTON - H2O), "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),\n    "[M+2H]2+": (1, 2, 2 * PROTON), "[M]+": (1, 1, -E), "[M-H2O]+": (1, 1, -E - H2O),\n    "[M+CH3OH+H]+": (1, 1, PROTON + MASS[\'C\'] + 4 * MASS[\'H\'] + MASS[\'O\']),\n    "[M+CH3CN+H]+": (1, 1, PROTON + 2 * MASS[\'C\'] + 3 * MASS[\'H\'] + MASS[\'N\']),\n    "[M-H]-": (1, 1, -PROTON), "[M-H2O-H]-": (1, 1, -PROTON - H2O), "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),\n    "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON), "[M+Cl]-": (1, 1, MASS[\'Cl\'] + E), "[M]-": (1, 1, E),\n    "[M-2H]-": (1, 2, -2 * PROTON), "[M+Na-2H]-": (1, 1, MASS[\'Na\'] - 2 * PROTON),\n    "[2M+H]+": (2, 1, PROTON), "[2M+Na]+": (2, 1, MASS[\'Na\'] - E), "[2M+NH4]+": (2, 1, NH4 - E),\n    "[2M+K]+": (2, 1, MASS[\'K\'] - E), "[2M-H]-": (2, 1, -PROTON), "[2M+CH2O2-H]-": (2, 1, FORMATE - PROTON),\n    "[2M+C2H4O2-H]-": (2, 1, ACETATE - PROTON), "[2M+Na-2H]-": (2, 1, MASS[\'Na\'] - 2 * PROTON),\n    "[3M+H]+": (3, 1, PROTON), "[3M-H]-": (3, 1, -PROTON),\n}\n\n\ndef neutral_mass(mz, adduct):\n    out = np.full(len(mz), np.nan); ad = np.asarray(adduct, dtype=object)\n    for a, (n, z, d) in ADDUCTS.items():\n        m = (ad == a)\n        if m.any(): out[m] = (mz[m] * z - d) / n\n    return out\n\n\n# ----------------------------------------------------------------------------------------------\n# ranking features (31)\n# ----------------------------------------------------------------------------------------------\nN_ANALOG = 80\nP_SIM = 3.0\nN_FEAT = 31\n\n\ndef _rank_norm(x):\n    o = np.argsort(-x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r / max(1, len(x) - 1)\n\n\ndef _z(x):\n    s = x.std()\n    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)\n\n\ndef rank_features(cand_fp, cand_lib, analog_fp, analog_sim, model_logits=None, frag=None):\n    nc = cand_fp.shape[0]\n    cf = cand_fp.astype(np.float32); cs = cf.sum(1)\n    lv = np.asarray(cand_lib, np.float32)\n    lvmax = float(lv.max()) if nc else 0.0\n    if analog_fp is not None and len(analog_sim):\n        af = analog_fp.astype(np.float32); asum = af.sum(1)\n        inter = cf @ af.T\n        tan = inter / (cs[:, None] + asum[None, :] - inter + 1e-9)\n        w = np.clip(np.asarray(analog_sim, np.float32), 0, None)\n        ap = (tan * (w ** P_SIM)[None, :]).max(1)\n        a1 = (tan * w[None, :]).max(1)\n        best_tan = tan.max(1); top_tan = tan[:, 0]; top_sim = float(w[0])\n        mean_tan = (tan * (w ** P_SIM)[None, :]).sum(1) / ((w ** P_SIM).sum() + 1e-9)\n    else:\n        ap = a1 = best_tan = top_tan = mean_tan = np.zeros(nc, np.float32); top_sim = 0.0\n    apmax = float(ap.max()) if nc else 0.0\n    if model_logits is not None:\n        raw = cf @ np.asarray(model_logits, np.float32)\n        nrm = raw / np.sqrt(np.maximum(cs, 1.0))\n        mfeat = [_z(raw), _rank_norm(raw), raw - raw.max(), _z(nrm), _rank_norm(nrm),\n                 (raw == raw.max()).astype(np.float32)]\n    else:\n        mfeat = [np.zeros(nc, np.float32)] * 6\n    if model_logits is not None and nc:\n        mr = _rank_norm(cf @ np.asarray(model_logits, np.float32))\n        lbest = int(np.argmax(lv)) if lvmax > 0 else -1\n        agree = float(1.0 - mr[lbest]) if lbest >= 0 else 0.0\n        abest = int(np.argmax(ap)) if apmax > 0 else -1\n        agree_a = float(1.0 - mr[abest]) if abest >= 0 else 0.0\n        xfeat = [lv * (1.0 - mr), ap * (1.0 - mr), np.full(nc, agree), np.full(nc, agree_a),\n                 np.full(nc, agree * lvmax), np.full(nc, float(np.corrcoef(lv, -mr)[0, 1]) if lv.std() > 1e-9 else 0.0)]\n    else:\n        xfeat = [np.zeros(nc, np.float32)] * 6\n    if frag is not None:\n        fr = np.asarray(frag, np.float32)\n        ffeat = [fr, _rank_norm(fr), fr - fr.max() if nc else fr, _z(fr)]\n    else:\n        ffeat = [np.zeros(nc, np.float32)] * 4\n    return np.column_stack([\n        lv, _rank_norm(lv), np.full(nc, lvmax), lv - lvmax, (lv > 0).astype(float),\n        ap, _rank_norm(ap), np.full(nc, apmax), ap - apmax,\n        a1, best_tan, top_tan, mean_tan, np.full(nc, top_sim),\n        np.full(nc, np.log(max(nc, 1))),\n        *mfeat, *ffeat, *xfeat,\n    ]).astype(np.float32)\n\n\n# ----------------------------------------------------------------------------------------------\n# MetFrag-lite\n# ----------------------------------------------------------------------------------------------\nAMU = {\'C\': 12.0, \'H\': 1.00782503207, \'N\': 14.0030740048, \'O\': 15.9949146196, \'P\': 30.97376163,\n       \'S\': 31.97207100, \'F\': 18.99840322, \'Cl\': 34.96885268, \'Br\': 78.9183371, \'I\': 126.904473,\n       \'Na\': 22.9897692809, \'K\': 38.96370668, \'Si\': 27.9769265325, \'B\': 11.0093054, \'Se\': 79.9165213}\nH = AMU[\'H\']\n\n\ndef mol_graph(smi):\n    from rdkit import Chem, RDLogger\n    RDLogger.DisableLog(\'rdApp.*\')\n    m = Chem.MolFromSmiles(smi)\n    if m is None: return None\n    n = m.GetNumAtoms()\n    w = np.zeros(n)\n    for a in m.GetAtoms():\n        w[a.GetIdx()] = AMU.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H\n    if (w == 0).any(): return None\n    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]\n    return w, bonds, n\n\n\ndef _components(n, bonds, drop):\n    adj = [[] for _ in range(n)]\n    for i, (a, b) in enumerate(bonds):\n        if i in drop: continue\n        adj[a].append(b); adj[b].append(a)\n    seen = np.zeros(n, bool); comps = []\n    for s in range(n):\n        if seen[s]: continue\n        stack = [s]; seen[s] = True; cur = [s]\n        while stack:\n            u = stack.pop()\n            for v in adj[u]:\n                if not seen[v]: seen[v] = True; stack.append(v); cur.append(v)\n        comps.append(cur)\n    return comps\n\n\ndef fragment_masses(smi, max_breaks=2, max_bonds=34):\n    g = mol_graph(smi)\n    if g is None: return np.zeros(0)\n    w, bonds, n = g\n    nb = len(bonds)\n    if nb == 0 or nb > max_bonds: return np.array([w.sum()])\n    out = {w.sum()}\n    for i in range(nb):\n        for c in _components(n, bonds, {i}):\n            out.add(float(w[c].sum()))\n    if max_breaks >= 2:\n        for i in range(nb):\n            for j in range(i + 1, nb):\n                for c in _components(n, bonds, {i, j}):\n                    out.add(float(w[c].sum()))\n    return np.array(sorted(out))\n\n\ndef frag_masses_safe(smi):\n    try: return fragment_masses(smi)\n    except Exception: return np.zeros(0)\n\n\n# ADDED: charge carriers a fragment may keep, per precursor adduct (prvsiyan only used +-proton)\nCARRIERS = {\n    "[M+H]+": (PROTON,), "[M-H2O+H]+": (PROTON,), "[M-2H2O+H]+": (PROTON,),\n    "[M+NH4]+": (PROTON, NH4 - E), "[M+Na]+": (PROTON, MASS[\'Na\'] - E), "[M+K]+": (PROTON, MASS[\'K\'] - E),\n    "[M-H]-": (-PROTON,), "[M-H2O-H]-": (-PROTON,), "[M+CH2O2-H]-": (-PROTON, FORMATE - PROTON),\n    "[M+Cl]-": (-PROTON, MASS[\'Cl\'] + E),\n}\n\n\ndef explain_score_adduct(frag_mass, peak_mz, peak_int, adduct, tol=0.01, h_shifts=(-2, -1, 0, 1, 2)):\n    if len(frag_mass) == 0 or len(peak_mz) == 0: return 0.0\n    car = CARRIERS.get(adduct, (PROTON,) if "+" in adduct[-2:] else (-PROTON,))\n    ion = np.sort(np.concatenate([frag_mass + dh * H + c for dh in h_shifts for c in car]))\n    w = np.sqrt(np.asarray(peak_int, float)); tot = w.sum()\n    if tot <= 0: return 0.0\n    idx = np.searchsorted(ion, peak_mz)\n    ok = np.zeros(len(peak_mz), bool)\n    for off in (-1, 0):\n        k = np.clip(idx + off, 0, len(ion) - 1)\n        ok |= np.abs(ion[k] - peak_mz) <= tol\n    return float(w[ok].sum() / tot)\n\n\ndef explain_score(frag_mass, peak_mz, peak_int, mode=1.0, tol=0.01, h_shifts=(-2, -1, 0, 1, 2)):\n    if len(frag_mass) == 0 or len(peak_mz) == 0: return 0.0\n    ion = []\n    for dh in h_shifts:\n        ion.append(frag_mass + dh * H + (PROTON if mode > 0 else -PROTON))\n    ion = np.sort(np.concatenate(ion))\n    w = np.sqrt(np.asarray(peak_int, float)); tot = w.sum()\n    if tot <= 0: return 0.0\n    idx = np.searchsorted(ion, peak_mz)\n    ok = np.zeros(len(peak_mz), bool)\n    for off in (-1, 0):\n        k = np.clip(idx + off, 0, len(ion) - 1)\n        ok |= np.abs(ion[k] - peak_mz) <= tol\n    return float(w[ok].sum() / tot)', 'pv_fp.py': '"""Spectrum -> fingerprint model (prvsiyan FPNet), kept apart so CPU worker processes never import torch."""\nimport math\nimport numpy as np\n# ----------------------------------------------------------------------------------------------\n# spectrum -> fingerprint model\n# ----------------------------------------------------------------------------------------------\nimport torch, torch.nn as nn, torch.nn.functional as F\n\nFP_MAX_PEAKS = 128\nADDUCT_LIST = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+",\n               "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",\n               "[M+2H]2+", "[M-2H]-", "[2M+H]+", "[2M+Na]+", "[2M+NH4]+", "[2M-H]-", "[2M+K]+",\n               "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[M+Na-2H]-", "[M-H2O]+", "<unk>"]\nADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}\nINSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]\n\n\ndef instr_family(s):\n    if s is None: return 4\n    t = str(s).lower()\n    if \'timstof\' in t: return 0\n    if \'orbitrap\' in t or \'qft\' in t or \'ftms\' in t or \'hybrid ft\' in t or \'itft\' in t or \'exactive\' in t: return 1\n    if \'tof\' in t: return 2\n    if \'trap\' in t or \'qq\' in t: return 3\n    return 4\n\n\ndef prep_peaks(mz, inten, prec_mz, max_peaks=FP_MAX_PEAKS, floor=1e-3, win=50.0, per_win=8):\n    mz = np.asarray(mz, np.float64); it = np.asarray(inten, np.float64)\n    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)\n    keep = (mz <= prec_mz + 1.5)\n    mz, it = mz[keep], it[keep]\n    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)\n    mx = it.max()\n    if mx <= 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)\n    keep = it >= floor * mx\n    mz, it = mz[keep], it[keep]\n    if len(mz) > max_peaks:\n        order = np.argsort(-it)\n        bucket = (mz // win).astype(np.int64)\n        cnt = {}; sel = []\n        for i in order:\n            b = bucket[i]; c = cnt.get(b, 0)\n            if c < per_win: cnt[b] = c + 1; sel.append(i)\n        sel = np.array(sel)\n        if len(sel) > max_peaks:\n            sel = sel[np.argsort(-it[sel])[:max_peaks]]\n        elif len(sel) < max_peaks:\n            rest = np.array([i for i in order if i not in set(sel.tolist())])\n            need = max_peaks - len(sel)\n            if len(rest): sel = np.concatenate([sel, rest[:need]])\n        mz, it = mz[sel], it[sel]\n    o = np.argsort(mz)\n    mz, it = mz[o], it[o]\n    v = np.sqrt(it / it.max())\n    return mz.astype(np.float32), v.astype(np.float32)\n\n\nclass SinEmb(nn.Module):\n    def __init__(self, dim, lo=-2.0, hi=3.2, power=1.0):\n        super().__init__()\n        n = dim // 2\n        wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)\n        self.register_buffer(\'inv\', (2 * math.pi) / wav)\n\n    def forward(self, x):\n        a = x.unsqueeze(-1) * self.inv\n        return torch.cat([torch.sin(a), torch.cos(a)], -1)\n\n\nclass Block(nn.Module):\n    def __init__(self, d, h, drop):\n        super().__init__(); self.h = h\n        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)\n        self.n2 = nn.LayerNorm(d)\n        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))\n        self.drop = nn.Dropout(drop)\n\n    def forward(self, x, pad):\n        B, N, D = x.shape; y = self.n1(x)\n        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)\n        m = (~pad)[:, None, None, :]\n        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)\n        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))\n        return x + self.drop(self.ff(self.n2(x)))\n\n\nclass FPNet(nn.Module):\n    def __init__(self, nbits, d=512, layers=6, heads=8, drop=0.1):\n        super().__init__()\n        self.d = d\n        self.mz_emb = SinEmb(d)\n        self.nl_emb = SinEmb(d)\n        self.pk = nn.Linear(2 * d + 1, d)\n        self.prec_emb = SinEmb(d)\n        self.ad = nn.Embedding(len(ADDUCT_LIST), d)\n        self.ins = nn.Embedding(len(INSTR_LIST), d)\n        self.gl = nn.Linear(d + 3, d)\n        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])\n        self.norm = nn.LayerNorm(d)\n        self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))\n\n    def forward(self, mz, it, pad, prec, ad, ins, ce, mode):\n        B, N = mz.shape\n        nl = (prec[:, None] - mz).clamp(min=0)\n        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))\n        g = self.gl(torch.cat([self.prec_emb(prec),\n                               (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),\n                               torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)\n        x = torch.cat([g.unsqueeze(1), p], 1)\n        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)\n        for b in self.blocks: x = b(x, pad)\n        x = self.norm(x)\n        cls = x[:, 0]\n        msk = (~pad[:, 1:]).float().unsqueeze(-1)\n        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)\n        return self.head(torch.cat([cls, mean], -1))\n\n\ndef load_fp_models(paths, dev):\n    single, merged = [], []\n    nbits = None\n    for pth in paths:\n        ck = torch.load(pth, map_location=\'cpu\', weights_only=False)\n        net = FPNet(ck[\'nbits\'], d=ck[\'d\'], layers=ck[\'layers\']).to(dev).eval()\n        net.load_state_dict(ck[\'model\'])\n        (merged if \'merged\' in str(pth).replace(\'\\\\\', \'/\').split(\'/\')[-1] else single).append(net)\n        nbits = ck[\'nbits\']\n    return single, merged, nbits\n\n\ndef merge_peaks(specs):\n    """specs: list of (mz, it). All peaks collapsed into one pseudo-spectrum."""\n    specs = [(a, b) for a, b in specs if len(a)]\n    if not specs: return np.zeros(0), np.zeros(0)\n    mz = np.concatenate([np.asarray(a, float) for a, _ in specs])\n    it = np.concatenate([np.asarray(b, float) / max(float(np.asarray(b, float).max()), 1e-9) for _, b in specs])\n    o = np.argsort(mz); mz, it = mz[o], it[o]\n    keep = np.ones(len(mz), bool)\n    for j in range(1, len(mz)):\n        if mz[j] - mz[j - 1] < 0.005:\n            if it[j] >= it[j - 1]: keep[j - 1] = False\n            else: keep[j] = False\n    return mz[keep], it[keep]\n\n\n@torch.no_grad()\ndef logits_batch(P, nets, dev, prec, adduct, instr, ce, mode):\n    """P: list of (mz, it) already prepped; per-row arrays prec/adduct(str)/instr(str)/ce/mode."""\n    keep = [i for i, (a, _) in enumerate(P) if len(a)]\n    if not keep: return None\n    P = [P[i] for i in keep]\n    B = len(P); N = max(len(a) for a, _ in P)\n    mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)\n    for i, (a, b) in enumerate(P):\n        mz[i, :len(a)] = a; it[i, :len(b)] = b; pad[i, :len(a)] = False\n    T = lambda x: torch.as_tensor(x, device=dev)\n    args = (T(mz), T(it), T(pad),\n            T(np.asarray(prec, np.float32)[keep]),\n            T(np.array([ADDUCT_IX.get(a, ADDUCT_IX[\'<unk>\']) for a in np.asarray(adduct, object)[keep]])),\n            T(np.array([instr_family(s) for s in np.asarray(instr, object)[keep]])),\n            T(np.asarray(ce, np.float32)[keep]),\n            T(np.asarray(mode, np.float32)[keep]))\n    return np.mean([n(*args).float().mean(0).cpu().numpy() for n in nets], axis=0)\n\n\ndef molecule_logits(models, specs, prec, adduct, instr, ce, mode):\n    """prvsiyan model_logits(): mean of (single-model per-spectrum mean) and (merged-model on merged list).\n    specs: list of (mz, it); prec/adduct/instr/ce/mode: per-spectrum."""\n    single, merged, dev = models\n    out = []\n    if single:\n        P = [prep_peaks(a, b, p) for (a, b), p in zip(specs, prec)]\n        za = logits_batch(P, single, dev, prec, adduct, instr, ce, mode)\n        if za is not None: out.append(za)\n    if merged:\n        mz, it = merge_peaks(specs)\n        pm = float(np.median(prec))\n        P = [prep_peaks(mz, it, pm)]\n        zb = logits_batch(P, merged, dev, [pm], [adduct[0]], [instr[0]], [25.0], [float(np.mean(mode))])\n        if zb is not None: out.append(zb)\n    return np.mean(out, axis=0) if out else None', 'casmi_engine.py': '"""End-to-end inference (Kaggle notebook body). Mirrors e02_channels.py + e05_rank.py feature blocks exactly,\nbut reads train.parquet / test.parquet directly. Runs locally too (set CASMI_LOCAL=1).\n\nStages: library -> candidate pool -> index -> per-molecule channels -> MetFrag-lite -> features -> ranker -> submission\n"""\nimport os, sys, glob, time, pickle, math\nos.environ.setdefault("OPENBLAS_NUM_THREADS", "1")\nimport numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa\nimport pv\nfrom pv import CFG\nimport re as _re\n_MONO = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, S=31.97207100, P=30.97376163,\n             Cl=34.96885268, Br=78.9183371, F=18.99840322, I=126.904473, Si=27.9769265325, Se=79.9165213,\n             Na=22.9897692809, K=38.96370668, B=11.0093054, As=74.9215965, Fe=55.9349375, D=2.0141017778)\n_TOK = _re.compile(r"([A-Z][a-z]?)(\\d*)")\n\n\ndef formula_mass(f):\n    """ADDED: neutral monoisotopic mass by arithmetic on molecular_formula (prvsiyan fix; immune to\n    riken\'s 0.005 Da precursor precision and gnps adduct mislabels). NaN if an element is untabulated."""\n    if not isinstance(f, str) or not f: return np.nan\n    m = 0.0\n    for el, n in _TOK.findall(f):\n        if el not in _MONO: return np.nan\n        m += _MONO[el] * (int(n) if n else 1)\n    return m\n\nPOOL = None\n\n\nclass RANK:\n    W_A = (0.35, 0.55)      # weight on class-1 simulation rows; two priors averaged\n    SEEDS = (0, 1)\n    GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03, min_samples_leaf=80, l2_regularization=1.0)\n\nT0 = time.time()\nLOCAL = os.environ.get("CASMI_LOCAL") == "1"\nROOTS = [r"C:\\Users\\HW-LEE\\Desktop\\CASMI"] if LOCAL else ["/kaggle/input"]\n\n\ndef find(name):\n    for root in ROOTS:\n        hits = glob.glob(os.path.join(root, "**", name), recursive=True)\n        if hits: return sorted(hits, key=len)[0]\n    raise FileNotFoundError(name)\n\n\ndef log(msg):\n    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)\n\n\n# ---------------------------------------------------------------------------------------------\n# library\n# ---------------------------------------------------------------------------------------------\ndef load_library(path):\n    """Row group by row group, peaks straight to float32: peak RAM stays near the final ~3.5 GB."""\n    f = pq.ParquetFile(path)\n    n_rows = f.metadata.num_rows\n    npk = 0\n    for i in range(f.num_row_groups):\n        c = f.read_row_group(i, columns=["ms2_mzs"]).column(0).combine_chunks()\n        npk += len(c.values)\n    off = np.zeros(n_rows + 1, np.int64)\n    allmz = np.empty(npk, np.float32); allin = np.empty(npk, np.float32)\n    prec, add, pol, ik, smi = [], [], [], [], []\n    fo, tims = [], []\n    r0 = p0 = 0\n    for i in range(f.num_row_groups):\n        t = f.read_row_group(i, columns=["inchikey14", "normalized_smiles", "adduct", "precursor_mz", "ionization_mode",\n                                         "ms2_mzs", "ms2_normalized_intensities", "molecular_formula", "instrument_type"])\n        mzc = t.column("ms2_mzs").combine_chunks(); itc = t.column("ms2_normalized_intensities").combine_chunks()\n        o = mzc.offsets.to_numpy().astype(np.int64)\n        n = len(o) - 1; k = int(o[-1] - o[0])\n        allmz[p0:p0 + k] = mzc.values.to_numpy(zero_copy_only=False)[o[0]:o[-1]]\n        allin[p0:p0 + k] = itc.values.to_numpy(zero_copy_only=False)[o[0]:o[-1]]\n        off[r0 + 1:r0 + n + 1] = p0 + (o[1:] - o[0])\n        prec.append(t.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64))\n        add += t.column("adduct").cast(pa.string()).to_pylist()\n        pol.append(np.array(t.column("ionization_mode").cast(pa.string()).to_pylist(), dtype=object) == "positive")\n        ik += t.column("inchikey14").cast(pa.string()).to_pylist()\n        smi += t.column("normalized_smiles").cast(pa.string()).to_pylist()\n        fo += t.column("molecular_formula").cast(pa.string()).to_pylist()\n        tims.append(np.array([isinstance(x, str) and x.lower().replace("-", "").replace(" ", "") == "timstof"\n                              for x in t.column("instrument_type").cast(pa.string()).to_pylist()]))\n        r0 += n; p0 += k\n        del t, mzc, itc\n    prec = np.concatenate(prec); add = np.asarray(add, dtype=object)\n    pol = np.where(np.concatenate(pol), 1, -1).astype(np.int8)\n    nmp = pv.neutral_mass(prec, add)\n    _fm = {x: formula_mass(x) for x in set(v for v in fo if isinstance(v, str))}\n    nmf = np.array([_fm.get(x, np.nan) if isinstance(x, str) else np.nan for x in fo])\n    nm = np.where(np.isfinite(nmf), nmf, nmp)\n    tims = np.concatenate(tims)\n    log(f"library mass index: formula {np.isfinite(nmf).mean():.1%}, timsTOF rows {tims.mean():.1%}")\n    del fo\n    return dict(off=off, mz=allmz, it=allin, prec=prec, nm=nm, pol=pol, tims=tims,\n                ik=np.asarray(ik, dtype=object), smi=np.asarray(smi, dtype=object))\n\n\n# ---------------------------------------------------------------------------------------------\n# candidate pool = COCONUT (prvsiyan fingerprints) U training structures (fingerprinted here)\n# ---------------------------------------------------------------------------------------------\n_g = {}\n\n\ndef _fp_init():\n    from rdkit import RDLogger\n    from rdkit.Chem import rdFingerprintGenerator\n    RDLogger.DisableLog("rdApp.*")\n    _g["bits"] = np.load(find("fp_bits.npy"))\n    _g["m2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)\n    _g["m3"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)\n    _g["rk"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)\n\n\ndef fp_and_mass(smi):\n    from rdkit import Chem\n    from rdkit.Chem import MACCSkeys\n    from rdkit.Chem.Descriptors import ExactMolWt\n    if not _g: _fp_init()\n    m = Chem.MolFromSmiles(smi)\n    if m is None: return None\n    try:\n        fp = np.concatenate([_g["m2"].GetFingerprintAsNumPy(m).astype(np.uint8),\n                             _g["m3"].GetFingerprintAsNumPy(m).astype(np.uint8),\n                             _g["rk"].GetFingerprintAsNumPy(m).astype(np.uint8),\n                             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])[_g["bits"]]\n        return np.packbits(fp), float(ExactMolWt(m))\n    except Exception:\n        return None\n\n\ndef canon_key(smi):\n    from rdkit import Chem, RDLogger\n    from rdkit.Chem.MolStandardize import rdMolStandardize\n    RDLogger.DisableLog("rdApp.*")\n    if "te" not in _g: _g["te"] = rdMolStandardize.TautomerEnumerator()\n    try:\n        m = Chem.MolFromSmiles(smi)\n        return Chem.MolToInchiKey(_g["te"].Canonicalize(m))[:14] if m is not None else None\n    except Exception:\n        return None\n\n\ndef build_pool(L, workers):\n    from multiprocessing import Pool as MPool\n    d = os.path.dirname(find("coco_fp.npy"))\n    cm = pickle.load(open(os.path.join(d, "coco_meta.pkl"), "rb"))\n    co_fp = np.load(os.path.join(d, "coco_fp.npy")); co_mass = np.load(os.path.join(d, "coco_mass.npy"))\n    co_keys = np.asarray(cm["keys"], dtype=object); co_smi = np.asarray(cm["smiles"], dtype=object)\n    # ADDED: ChEBI + LIPID MAPS (prvsiyan/chebi-lipidmaps-casmi26), same 6,930-bit packed fingerprints\n    bd = os.path.dirname(find("bio_fp.npy")); bm = pickle.load(open(os.path.join(bd, "bio_meta.pkl"), "rb"))\n    bkeys = np.asarray(bm["keys"], dtype=object); bnew = ~pd.Series(bkeys).isin(set(co_keys)).values\n    co_fp = np.vstack([co_fp, np.load(os.path.join(bd, "bio_fp.npy"))[bnew]])\n    co_mass = np.concatenate([co_mass, np.load(os.path.join(bd, "bio_mass.npy"))[bnew]])\n    co_keys = np.concatenate([co_keys, bkeys[bnew]])\n    co_smi = np.concatenate([co_smi, np.asarray(bm["smiles"], dtype=object)[bnew]])\n    log(f"+ ChEBI/LIPID MAPS {int(bnew.sum()):,} new structures")\n    # v14 (from Hongbin Gu, guhongbin/casmi26-v447-pubchemlite): PubChemLite CC0 candidates\n    # (franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints), same 6,930-bit packed fingerprints.\n    # New arrays are built first and swapped in only if every step succeeds; any failure keeps the v13 pool.\n    try:\n        pcd = os.path.dirname(find("pubchemlite_fp.npy"))\n        pcm = pickle.load(open(os.path.join(pcd, "pubchemlite_meta.pkl"), "rb"))\n        pckeys = np.asarray(pcm["keys"], dtype=object)\n        pcnew = ~pd.Series(pckeys).isin(set(co_keys)).values\n        _fp = np.vstack([co_fp, np.load(os.path.join(pcd, "pubchemlite_fp.npy"))[pcnew]])\n        _ms = np.concatenate([co_mass, np.load(os.path.join(pcd, "pubchemlite_mass.npy"))[pcnew]])\n        _ks = np.concatenate([co_keys, pckeys[pcnew]])\n        _sm = np.concatenate([co_smi, np.asarray(pcm["smiles"], dtype=object)[pcnew]])\n        assert len(_fp) == len(_ms) == len(_ks) == len(_sm), (len(_fp), len(_ms), len(_ks), len(_sm))\n        co_fp, co_mass, co_keys, co_smi = _fp, _ms, _ks, _sm\n        log(f"+ PubChemLite CC0 {int(pcnew.sum()):,} new structures")\n    except Exception as _pe:\n        log(f"PubChemLite CC0 skipped: {_pe!r}")\n    tr = pd.DataFrame({"ik": L["ik"], "smi": L["smi"]}).dropna().drop_duplicates("ik")\n    tr = tr[~tr.ik.isin(set(co_keys))]\n    log(f"COCONUT {len(co_keys):,}; training structures to fingerprint {len(tr):,}")\n    with MPool(workers) as mp:\n        res = mp.map(fp_and_mass, list(tr.smi), chunksize=500)\n    ok = np.array([r is not None for r in res])\n    tr_fp = np.stack([r[0] for r in res if r is not None]); tr_mass = np.array([r[1] for r in res if r is not None])\n    fp = np.vstack([co_fp, tr_fp]); mass = np.concatenate([co_mass, tr_mass])\n    keys = np.concatenate([co_keys, tr.ik.values[ok]]); smis = np.concatenate([co_smi, tr.smi.values[ok]])\n    good = np.isfinite(mass)\n    o = np.argsort(np.where(good, mass, 1e18), kind="mergesort")[: good.sum()]\n    P = dict(fp=fp[o], mass=mass[o], keys=keys[o], smiles=smis[o])\n    P["k2i"] = pd.Series(np.arange(len(o)), index=P["keys"]); P["k2i"] = P["k2i"][~P["k2i"].index.duplicated()]\n    log(f"pool {len(o):,} structures")\n    return P\n\n\ndef pool_fps(idx):\n    return np.unpackbits(np.asarray(POOL["fp"][np.asarray(idx, np.int64)]), axis=1)[:, :6930]\n\n\ndef pool_window(t, ppm):\n    a = np.searchsorted(POOL["mass"], t * (1 - ppm / 1e6), "left")\n    b = np.searchsorted(POOL["mass"], t * (1 + ppm / 1e6), "right")\n    return np.arange(a, b)\n\n\n# ---------------------------------------------------------------------------------------------\n# index: neutral-mass sorted library rows + cleaned representatives (all / positive / negative)\n# ---------------------------------------------------------------------------------------------\ndef _reps(L, rows):\n    npk = np.diff(L["off"])[rows]\n    df = pd.DataFrame({"row": rows, "p": L["row_p"][rows], "npk": npk, "tims": L["tims"][rows]})\n    # ADDED: prefer the test instrument (timsTOF) representative, then the richest spectrum\n    df = df.sort_values(["p", "tims", "npk", "row"], ascending=[True, False, False, True]).drop_duplicates("p")\n    rep = df.row.values; rep_nm = L["nm"][rep]\n    o = np.argsort(rep_nm, kind="mergesort"); rep = rep[o]\n    roff, rmz, rit = pv.clean_store(rep, L["off"], L["mz"], L["it"], CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)\n    return dict(rep=rep, rep_p=L["row_p"][rep], rep_nm=L["nm"][rep], roff=roff, rmz=rmz, rit=rit)\n\n\ndef build_index(L):\n    ok = np.isfinite(L["nm"]) & (L["row_p"] >= 0)\n    rows = np.where(ok)[0]\n    o = np.argsort(L["nm"][rows], kind="mergesort")\n    I = dict(lib_rows=rows[o], lib_nm=L["nm"][rows[o]])\n    I["all"] = _reps(L, rows)\n    I["pos"] = _reps(L, rows[L["pol"][rows] == 1])\n    I["neg"] = _reps(L, rows[L["pol"][rows] == -1])\n    log(f"index: {len(rows):,} library rows; representatives {len(I[\'all\'][\'rep\']):,}")\n    return I\n\n\n# ---------------------------------------------------------------------------------------------\n# per-molecule channels (identical to e02_channels.molecule_channels)\n# ---------------------------------------------------------------------------------------------\ndef clean(mz, it):\n    return pv._clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),\n                     CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)\n\n\ndef analog_search(R, Q, target):\n    lo = np.searchsorted(R["rep_nm"], target - CFG.ANALOG_WIN, "left")\n    hi = np.searchsorted(R["rep_nm"], target + CFG.ANALOG_WIN, "right")\n    asim = np.zeros(hi - lo, np.float32)\n    if hi > lo and Q:\n        shift = (target - R["rep_nm"][lo:hi]).astype(np.float32)\n        for qm, qp in Q:\n            asim = np.maximum(asim, pv.search_shift_pre(qm, qp, lo, hi, R["roff"], R["rmz"], R["rit"], CFG.MZ_TOL, shift))\n    return R["rep_p"][lo:hi], asim\n\n\ndef top_analogs(p, s):\n    if len(p) == 0: return []\n    df = pd.DataFrame({"p": p, "s": s}).groupby("p", sort=False).s.max().sort_values(ascending=False, kind="mergesort")\n    return list(zip(df.index[:CFG.N_ANALOG].astype(int), df.values[:CFG.N_ANALOG].astype(float)))\n\n\ndef molecule_channels(L, I, specs, precs, pols, target):\n    Q = [clean(a, b) for a, b in specs]\n    keep = [i for i, (a, _) in enumerate(Q) if len(a)]\n    Q = [Q[i] for i in keep]; precs = np.asarray(precs)[keep]; pols = np.asarray(pols)[keep]\n    tol = target * CFG.PPM_WIN / 1e6\n    lo = np.searchsorted(I["lib_nm"], target - tol, "left"); hi = np.searchsorted(I["lib_nm"], target + tol, "right")\n    crow = I["lib_rows"][lo:hi]\n    u, inv = np.unique(L["row_p"][crow], return_inverse=True)\n    M = np.zeros((max(len(Q), 1), len(u)), np.float32); MS = np.zeros_like(M)\n    for j, (qm, qp) in enumerate(Q):\n        if len(crow) == 0: break\n        s = pv.search(qm, qp, crow, L["off"], L["mz"], L["it"], CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,\n                      CFG.INT_POWER, CFG.ENT_WEIGHT)\n        np.maximum.at(M[j], inv, s)\n        sh = (precs[j] - L["prec"][crow]).astype(np.float32)\n        s2 = pv.search_shift_rows(qm, qp, crow, L["off"], L["mz"], L["it"], CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,\n                                  CFG.INT_POWER, CFG.ENT_WEIGHT, sh)\n        np.maximum.at(MS[j], inv, s2)\n    cnt = np.bincount(inv, minlength=len(u)).astype(np.float32)\n    z = np.zeros(0, np.float32)\n    lib = (u, M.max(0) if len(u) else z, M.mean(0) if len(u) else z, MS.max(0) if len(u) else z, cnt)\n    p, s = analog_search(I["all"], Q, target)\n    ana = top_analogs(p, s)\n    ps, ss = [], []\n    for sign, name in ((1, "pos"), (-1, "neg")):\n        QQ = [q for q, pl in zip(Q, pols) if pl == sign]\n        if QQ:\n            p, s = analog_search(I[name], QQ, target); ps.append(p); ss.append(s)\n    anapol = top_analogs(np.concatenate(ps), np.concatenate(ss)) if ps else []\n    return lib, ana, anapol\n\n\ndef parse_ce(v):\n    try:\n        a = np.abs(np.atleast_1d(np.asarray(v, float)))\n        return float(a.mean()) if len(a) else 25.0\n    except Exception:\n        return 25.0\n\n\n\n\n# ---------------------------------------------------------------------------------------------\n# ranking features: prvsiyan\'s 31 + our blocks (identical to e05_rank.py BLOCKS)\n# ---------------------------------------------------------------------------------------------\ndef _rel(v):\n    v = np.asarray(v, np.float32)\n    return [v, pv._rank_norm(v).astype(np.float32), v - (v.max() if len(v) else 0.0)]\n\n\ndef _analog_feats(cfp, ana):\n    nc = len(cfp)\n    if not ana: return np.zeros((nc, 5), np.float32)\n    afp = pool_fps([p for p, _ in ana]).astype(np.float32); cf = cfp.astype(np.float32)\n    inter = cf @ afp.T\n    tan = inter / (cf.sum(1)[:, None] + afp.sum(1)[None, :] - inter + 1e-9)\n    w = np.clip(np.array([s for _, s in ana], np.float32), 0, None)\n    ap = (tan * (w ** 3)[None, :]).max(1)\n    ap6 = (tan * (w ** 6)[None, :]).max(1)\n    return np.column_stack(_rel(ap) + [ap6, np.full(nc, w[0])]).astype(np.float32)\n\n\ndef blk_base(r, cfp):\n    ana = r["ana"]\n    afp = pool_fps([p for p, _ in ana]) if ana else None\n    sims = np.array([s for _, s in ana], np.float32)\n    return pv.rank_features(cfp, r["lib"], afp, sims, r["zlog"], r["frag"])\n\n\ndef blk_mass(r, cfp):\n    ppm = (POOL["mass"][r["cand"]] - r["target"]) / r["target"] * 1e6\n    return np.column_stack([ppm, np.abs(ppm), np.abs(ppm + 1.5)]).astype(np.float32)\n\n\ndef blk_libx(r, cfp):\n    cnt = np.log1p(r["lib_cnt"])\n    return np.column_stack(_rel(r["lib_mean"]) + _rel(r["libsh"]) + [cnt, cnt - cnt.max()]).astype(np.float32)\n\n\ndef blk_anapol(r, cfp):\n    return _analog_feats(cfp, r["anapol"])\n\n\ndef blk_fragx(r, cfp):\n    return np.column_stack(_rel(r["frag_ad"]) + _rel(r["frag_mean"]) + [r["frag_pol"]]).astype(np.float32)\n\n\nBLOCKS = {"base": blk_base, "mass": blk_mass, "libx": blk_libx, "anapol": blk_anapol, "fragx": blk_fragx}\n\n\n# ---------------------------------------------------------------------------------------------\n# run\n# ---------------------------------------------------------------------------------------------\ndef compute_channels(L, I, models, te):\n    mols = list(te.groupby("molecule_id", sort=True))\n    recs = []\n    for gi, (mid, sub) in enumerate(mols):\n        specs = []\n        for r in sub.itertuples():\n            a_ = np.asarray(r.ms2_mzs, np.float32); b_ = np.asarray(r.ms2_normalized_intensities, np.float32)\n            k_ = a_ <= float(r.precursor_mz) + 2.0\n            a_, b_ = a_[k_], b_[k_]\n            specs.append((a_, b_ / b_.max() if len(b_) and b_.max() > 0 else b_))\n        nms = pv.neutral_mass(sub.precursor_mz.values.astype(np.float64), sub.adduct.astype(str).values)\n        nms = nms[np.isfinite(nms)]\n        rec = dict(mid=mid, cand=np.zeros(0, np.int64))\n        if len(nms):\n            target = float(np.median(nms))\n            modes = np.where(sub.ionization_mode.astype(str).values == "positive", 1.0, -1.0)\n            lib, ana, anapol = molecule_channels(L, I, specs, sub.precursor_mz.values.astype(np.float64), modes, target)\n            cand = pool_window(target, CFG.PPM_WIN)\n            if len(cand) == 0: cand = pool_window(target, CFG.PPM_FALLBACK)\n            ces = np.array([parse_ce(v) for v in sub.collision_energy_ev.values], np.float32)\n            import pv_fp\n            zlog = pv_fp.molecule_logits(models, specs, sub.precursor_mz.values.astype(np.float32),\n                                         list(sub.adduct.astype(str).values), list(sub.instrument_type.astype(str).values),\n                                         ces, modes)\n            u, lmax, lmean, lsh, lcnt = lib\n            pos = {int(x): i for i, x in enumerate(u)}\n            ci = np.array([pos.get(int(c), -1) for c in cand], np.int64); has = ci >= 0\n            rec.update(target=target, cand=cand, zlog=None if zlog is None else zlog.astype(np.float32), specs=specs,\n                       mode=float(np.mean(modes)), modes=modes, adducts=list(sub.adduct.astype(str).values),\n                       ana=ana, anapol=anapol)\n            for nm_, arr in (("lib", lmax), ("lib_mean", lmean), ("libsh", lsh), ("lib_cnt", lcnt)):\n                v = np.zeros(len(cand), np.float32); v[has] = arr[ci[has]]; rec[nm_] = v\n        recs.append(rec)\n        if gi % 50 == 0: log(f"  channels {gi}/{len(mols)}")\n    return mols, recs\n\n\ndef compute_frag(recs, workers):\n    from multiprocessing import Pool as MPool\n    uc = np.unique(np.concatenate([r["cand"] for r in recs]))\n    log(f"MetFrag-lite on {len(uc):,} candidates")\n    with MPool(workers) as mp:\n        fr = mp.map(pv.frag_masses_safe, list(POOL["smiles"][uc]), chunksize=16)\n    fmap = dict(zip(uc.tolist(), fr))\n    for r in recs:\n        if not len(r["cand"]): continue\n        peaks = []\n        for a_, b_ in r.pop("specs"):\n            m2, i2 = pv._clean(a_, b_, CFG.INT_FLOOR, CFG.MAX_PEAKS, 1.0, False)\n            peaks.append((np.asarray(m2, float), np.asarray(i2, float)))\n        F1 = np.zeros((len(r["cand"]), max(len(peaks), 1)), np.float32); F2 = np.zeros_like(F1); F3 = np.zeros_like(F1)\n        for j, c in enumerate(r["cand"]):\n            fm = fmap[int(c)]\n            for k, (x, y) in enumerate(peaks):\n                F1[j, k] = pv.explain_score(fm, x, y, mode=r["mode"], tol=CFG.MZ_TOL)\n                F2[j, k] = pv.explain_score_adduct(fm, x, y, r["adducts"][k], tol=CFG.MZ_TOL)\n                F3[j, k] = pv.explain_score(fm, x, y, mode=r["modes"][k], tol=CFG.MZ_TOL)\n        r["frag"] = F1.max(1); r["frag_ad"] = F2.max(1); r["frag_mean"] = F3.mean(1); r["frag_pol"] = F3.max(1)\n\n\ndef _hgb(X, Y, W, seed, gbm):\n    from sklearn.ensemble import HistGradientBoostingClassifier\n    m = HistGradientBoostingClassifier(random_state=seed, **gbm); m.fit(X, Y, sample_weight=W)\n    return m\n\n\ndef fit_rankers(rank_train_path):\n    """Our ranker: simulation rows from 7 query sets (S = 0 class-1 sim, 1 = class-2 sim)."""\n    z = np.load(rank_train_path, allow_pickle=True)\n    X, Y, S = z["X"], z["Y"], z["S"]\n    rankers = [_hgb(X, Y, np.where(S == 0, wA, 1.0 - wA), sd, RANK.GBM) for wA in RANK.W_A for sd in RANK.SEEDS]\n    log(f"our ranker: {len(rankers)} GBMs on {X.shape[0]:,} rows x {X.shape[1]} features")\n    return rankers, [str(b) for b in z["blocks"]]\n\n\ndef fit_pv_rankers(pv_rank_path, priors=(0.30, 0.60), seeds=(0, 1, 2, 3)):\n    """prvsiyan\'s ranker, exactly as in his notebook: shipped rank_train.npz (M = 0 class-1 sim)."""\n    z = np.load(pv_rank_path)\n    rankers = [_hgb(z["X"], z["Y"], np.where(z["M"] == 0, w1, 1.0 - w1), sd, CFG.GBM) for w1 in priors for sd in seeds]\n    log(f"prvsiyan ranker: {len(rankers)} GBMs on {z[\'X\'].shape[0]:,} rows x {z[\'X\'].shape[1]} features")\n    return rankers, z["X"].shape[1]\n\n\ndef score_molecules(recs, rankers, blocks, use_fp):\n    out = {}\n    for r in recs:\n        if not len(r["cand"]): continue\n        rr = r if use_fp else dict(r, zlog=None)\n        cfp = pool_fps(r["cand"])\n        Xm = np.hstack([BLOCKS[b](rr, cfp) for b in blocks]).astype(np.float32)\n        out[r["mid"]] = np.mean([m.predict_proba(Xm)[:, 1] for m in rankers], axis=0)\n    return out\n\n\ndef blend_scores(a, b, wa):\n    """Within-molecule rank blend (both inputs are per-candidate probabilities)."""\n    out = {}\n    for mid in a:\n        ra = 1.0 - pv._rank_norm(a[mid]); rb = 1.0 - pv._rank_norm(b[mid])\n        out[mid] = wa * ra + (1.0 - wa) * rb + 1e-6 * a[mid]\n    return out\n\n\ndef make_submissions(mols, recs, score_sets, sample_path, workers, topn=25):\n    """score_sets: {name: {mid: scores}} -> {name: submission DataFrame}; one canonicalisation pass for all."""\n    from multiprocessing import Pool as MPool\n    cand = {r["mid"]: r["cand"] for r in recs if len(r["cand"])}\n    ordered = {n: {mid: cand[mid][np.argsort(-sc, kind="mergesort")][:topn + 15] for mid, sc in S.items()}\n               for n, S in score_sets.items()}\n    allc = [v for o in ordered.values() for v in o.values()]\n    short = np.unique(np.concatenate(allc)) if allc else np.zeros(0, np.int64)\n    with MPool(workers) as mp:\n        ck = mp.map(canon_key, list(POOL["smiles"][short]), chunksize=64)\n    ckey = dict(zip(short.tolist(), ck))\n    samp = pd.read_csv(sample_path)\n    subs = {}\n    for n, o in ordered.items():\n        rows = []\n        for mid, _ in mols:\n            out, seen = [], set()\n            for c in o.get(mid, []):\n                k = ckey.get(int(c)) or POOL["keys"][c]\n                if k in seen: continue\n                seen.add(k); out.append(POOL["smiles"][c])\n                if len(out) == topn: break\n            while len(out) < topn:\n                out.append("CCO")\n            rows.append((mid, ";".join(out[:topn])))\n        sub = samp[["molecule_id"]].merge(pd.DataFrame(rows, columns=["molecule_id", "smiles"]), on="molecule_id", how="left")\n        sub["smiles"] = sub["smiles"].fillna("CCO")\n        assert len(sub) == len(samp) and sub.molecule_id.duplicated().sum() == 0 and sub.smiles.isnull().sum() == 0\n        assert (sub.smiles.str.split(";").map(len) <= topn).all()\n        subs[n] = sub\n    return subs\n\n\ndef main(test_path, train_path, sample_path, our_rank_path, pv_rank_path, fp_model_paths, workers=4, limit=0, w_pv=0.88):\n    global POOL\n    import torch\n    import pv_fp\n    L = load_library(train_path); log(f"library {len(L[\'prec\']):,} spectra")\n    POOL = build_pool(L, workers)\n    L["row_p"] = POOL["k2i"].reindex(L["ik"]).fillna(-1).values.astype(np.int64)\n    del L["ik"], L["smi"]\n    I = build_index(L)\n    dev = "cuda" if torch.cuda.is_available() else "cpu"\n    single, merged, _ = pv_fp.load_fp_models(fp_model_paths, dev)\n    log(f"fp models: {len(single)} single + {len(merged)} merged on {dev}")\n    te = pq.read_table(test_path).to_pandas()\n    if limit:\n        keep = sorted(te.molecule_id.unique())[:limit]; te = te[te.molecule_id.isin(keep)]\n    log(f"test: {len(te):,} spectra / {te.molecule_id.nunique():,} molecules")\n    mols, recs = compute_channels(L, I, (single, merged, dev), te)\n    del L, I\n    compute_frag(recs, workers)\n    ours, blocks = fit_rankers(our_rank_path)\n    s_ours = score_molecules(recs, ours, blocks, use_fp=False)\n    del ours\n    pvr, nfeat = fit_pv_rankers(pv_rank_path)\n    s_pv = score_molecules(recs, pvr, ["base"], use_fp=True)\n    del pvr\n    subs = make_submissions(mols, recs, {"blend": blend_scores(s_pv, s_ours, w_pv), "pv": s_pv, "ours": s_ours},\n                            sample_path, workers)\n    log("submissions ready")\n    return subs, recs, {"pv": s_pv, "ours": s_ours}\n'}
for _n, _src in ENG_FILES.items():
    open(os.path.join(ENG_DIR, _n), 'w').write(_src)
open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w').write('"""Our engine (two-ranker, BIO + AFIX) -> {molecule_id: {smiles: [...], keys: [...]}} (blend order, top 40)."""\nimport os, sys, json, time\nimport numpy as np\nif __name__ == \'__main__\':\n    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)\n    test, train, sample, out = sys.argv[1:5]\n    import casmi_engine as E\n    _orig_find = E.find\n    def _find(n):                                  # v4g ships its own fp_bits.npy (10,226 bits): take prvsiyan\'s 6,930\n        if n == \'fp_bits.npy\':\n            return os.path.join(os.path.dirname(_orig_find(\'coco_fp.npy\')), \'fp_bits.npy\')\n        return _orig_find(n)\n    E.find = _find\n    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200\n    import glob\n    fp_models = sorted(p for p in glob.glob(\'/kaggle/input/**/fp_*.pt\', recursive=True) if \'casmi26-fp-models-v2\' in p)\n    print(\'engine fp models\', fp_models, flush=True)\n    T0 = time.time()\n    subs, recs, S = E.main(test, train, sample, E.find(\'sim_rank_rows_nofp.npz\'), E.find(\'rank_train.npz\'), fp_models,\n                           workers=os.cpu_count(), w_pv=0.88)\n    bl = E.blend_scores(S[\'pv\'], S[\'ours\'], 0.88)\n    res = {}\n    for r in recs:\n        mid = r[\'mid\']\n        if mid not in bl:\n            continue\n        order = np.argsort(-bl[mid], kind=\'mergesort\')[:80]\n        smis, keys, seen = [], [], set()\n        for c in r[\'cand\'][order]:\n            s = E.POOL[\'smiles\'][c]; k = E.canon_key(s)\n            if k is None or k in seen:\n                continue\n            seen.add(k); smis.append(s); keys.append(k)\n            if len(smis) >= 40:\n                break\n        res[str(mid)] = dict(smiles=smis, keys=keys)\n    json.dump(res, open(out, \'w\'))\n    print(\'engine lists\', len(res), f\'{time.time()-T0:.0f}s\', flush=True)\n')
ENG = {}
try:
    _t0 = time.time()
    rr = subprocess.run([sys.executable, os.path.join(ENG_DIR, 'eng_runner.py'), os.path.join(COMP, 'test.parquet'),
                         os.path.join(COMP, 'train.parquet'), os.path.join(COMP, 'sample_submission.csv'),
                         '/kaggle/working/eng_lists.json'], capture_output=True, text=True, timeout=4 * 3600, cwd=ENG_DIR)
    print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
    ENG = json.load(open('/kaggle/working/eng_lists.json'))
    print('engine lists', len(ENG), f'{time.time()-_t0:.0f}s')
except Exception as e:
    print('ENGINE FAILED -> v4g lists kept:', repr(e))


    


# %% [code]

'''## 4. PubChem-only channel with a popularity prior

This channel searches about 105 million PubChem structures that are not in the main candidate pool. Its score combines the spectrum fit with how well documented a structure is.

$$\\text{score} = z(f\\cdot z) + 0.25\\,\\big(\\log(1+\\#\\text{PubChem substances}) + \\log(1+\\#\\text{PubMed papers})\\big)$$

The 200 most documented structures in the mass window are always added to the shortlist. A gate later decides how many PubChem candidates enter the final list. This prior added +0.004 on the leaderboard.
'''


    


# %% [code]

V17_PATCH = "\n\n# ---- v17 patch: popularity prior (log1p SID + log1p PMID from PubChem, mapped onto this store by stereo-stripped SMILES) ----\n_POP_DIR = os.environ.get('CASMI_POP_DIR'); _POP_LAM = float(os.environ.get('CASMI_POP_LAM', '0')); _POP_UNION = int(os.environ.get('CASMI_POP_UNION', '0'))\n_init_worker0 = init_worker\n\n\ndef init_worker(pc_dir, bits_path, pool_meta_path, code_dir=None):\n    _init_worker0(pc_dir, bits_path, pool_meta_path, code_dir)\n    if _POP_DIR:\n        _W['ls'] = np.load(os.path.join(_POP_DIR, 'pc_lsid.npy'), mmap_mode='r')\n        _W['lp'] = np.load(os.path.join(_POP_DIR, 'pc_lpmid.npy'), mmap_mode='r')\n\n\ndef probe_one(task):\n    mid, target, z = task\n    t0 = time.time()\n    chem = _W['chem']; Chem = _W['Chem']; gen = _W['ecfp4']\n    tol = float(target) * PPM * 1e-6\n    a0 = int(np.searchsorted(_W['mass'], float(target) - tol, 'left'))\n    smis = _window(float(target))\n    n = len(smis)\n    diag = dict(molecule_id=mid, n_window=n, n_pass1=0, n_pass2=0, n_keyed=0, n_in_pool=0, n_listed=0, fz_top=None)\n    if n == 0:\n        diag['secs'] = time.time() - t0\n        return mid, [], [], [], diag\n    raw_e = _W['raw_e']; z_e = z[_W['sel_e']].astype(np.float32)\n    E = np.zeros((n, len(raw_e)), np.float32); ok = np.zeros(n, bool)\n    for i, s in enumerate(smis):\n        m = Chem.MolFromSmiles(s)\n        if m is None:\n            continue\n        E[i] = gen.GetFingerprintAsNumPy(m)[raw_e]; ok[i] = True\n    sc1 = E @ z_e\n    sc1[~ok] = -np.inf\n    n1 = min(PC_N1, int(ok.sum()))\n    diag['n_pass1'] = n1\n    if n1 == 0:\n        diag['secs'] = time.time() - t0\n        return mid, [], [], [], diag\n    top1 = np.argpartition(-sc1, n1 - 1)[:n1] if n1 < n else np.where(ok)[0]\n    use_pop = ('ls' in _W) and _POP_LAM > 0\n    pop = (np.asarray(_W['ls'][a0:a0 + n], np.float32) + np.asarray(_W['lp'][a0:a0 + n], np.float32)) if use_pop else np.zeros(n, np.float32)\n    if use_pop and _POP_UNION > 0:\n        pp = np.where(ok, pop, -np.inf); ku = min(_POP_UNION, int(ok.sum()))\n        top1 = np.union1d(top1, np.argpartition(-pp, ku - 1)[:ku])\n    bits = _W['bits']\n    fz, idx = [], []\n    for i in top1:\n        fp = chem.raw_fingerprint(smis[i])\n        if fp is None:\n            continue\n        fz.append(float(fp[bits].astype(np.float32) @ z)); idx.append(int(i))\n    diag['n_pass2'] = len(idx)\n    if not idx:\n        diag['secs'] = time.time() - t0\n        return mid, [], [], [], diag\n    fz = np.asarray(fz, np.float64); idx = np.asarray(idx)\n    keyc = {}\n    def key(j):\n        if j not in keyc:\n            keyc[j] = chem.score_key(smis[idx[j]]); diag['n_keyed'] += 1\n        return keyc[j]\n    for j in np.argsort(-fz, kind='stable'):        # gate quantity = v16's pc_fz[0] (best non-pool structure by raw f.z)\n        k = key(j)\n        if k is not None and k not in _W['pool_keys']:\n            diag['fz_top'] = float(fz[j]); break\n    zf = (fz - fz.mean()) / (fz.std() + 1e-9)\n    sc = zf + _POP_LAM * pop[idx]\n    order = np.argsort(-sc, kind='stable')\n    out, out_s, out_k, out_p, out_sc, seen = [], [], [], [], [], set()\n    for j in order:\n        k = key(j)\n        if k is None or k in seen:\n            continue\n        seen.add(k)\n        if k in _W['pool_keys']:\n            diag['n_in_pool'] += 1\n            continue\n        out.append(smis[idx[j]]); out_s.append(float(sc[j])); out_k.append(k); out_p.append(float(pop[idx[j]])); out_sc.append(float(sc[j]))\n        if len(out) >= K:\n            break\n    diag['n_listed'] = len(out)\n    diag['S'] = float(out_sc[0]) if out_sc else 0.0\n    diag['top_pop'] = float(out_p[0]) if out_p else 0.0\n    diag['secs'] = time.time() - t0\n    return mid, out, out_s, out_k, diag\n"
'''PubChem-only channel in its own process (S1 code snapshot + FPNet A+B of casmi26-v3-models) -- identical to v3.6'''
CORE = '"""P_pubchem probe core (lever 3). Embedded verbatim into the Kaggle notebook (written to /kaggle/working and imported,\nso worker processes can import it) and imported directly by the local smoke test.\n\nPer molecule:\n  1. f.z logits of the A+B FPNet ensemble, exactly like Engine.model_logits: z = 0.5 * (mean per-spectrum logits +\n     mean per-polarity merged-spectrum logits); models averaged inside ModelBank.\n  2. PubChem window: +-10 ppm around the target neutral mass (median over the molecule\'s spectra), no subsampling.\n  3. pass 1 (src/casmi/pubchem.py tier): partial f.z on the ECFP4 block of the selected bits, keep top PC_N1 = 1000.\n  4. pass 2: full selected-bit fingerprints, full f.z, sort descending.\n  5. walk down the f.z order, compute the metric score key (tautomer-canonical InChIKey14); keep a structure only if\n     its key is NOT in the candidate pool (train U COCONUT) and not already listed; stop at 25.\n  An empty list becomes the single \'CCO\' fallback (never padded).\nReading: sum of reciprocal ranks of PubChem-only truths / ~130 public molecules.\n"""\nfrom __future__ import annotations\n\nimport os\nimport time\n\nimport numpy as np\n\nPC_N1 = 5000\nPPM = 10.0\nK = 25\n\n_W = {}\n\n\n# ------------------------------------------------------------------------------------------ model part (main process)\ndef molecule_logits(bank, spectra):\n    """spectra: list of dict(mz, it, mode, adduct, prec, ce, ce_n). Returns z_avg (nbits,) float32 or None."""\n    from casmi import chem, fpnet\n    from casmi.spectra import merge_spectra\n    items_s = []\n    for s in spectra:\n        pm, pi = fpnet.prep_peaks(s[\'mz\'], s[\'it\'], s[\'prec\'])\n        ce_ok = s[\'ce\'] is not None and not np.isnan(s[\'ce\'])\n        items_s.append(dict(mz=pm, it=pi, prec=float(s[\'prec\']), adduct_ix=chem.adduct_index(s[\'adduct\']),\n                            ce=float(s[\'ce\']) if ce_ok else 0.0, ce_known=1.0 if ce_ok else 0.0,\n                            n_merged=min(int(s.get(\'ce_n\', 1)), 8), mode=float(s[\'mode\'])))\n    items_m = []\n    for md in (1, -1):\n        grp = [s for s in spectra if s[\'mode\'] == md]\n        if not grp:\n            continue\n        mm, ii = merge_spectra([(s[\'mz\'], s[\'it\']) for s in grp])\n        prec = float(np.median([s[\'prec\'] for s in grp]))\n        pm, pi = fpnet.prep_peaks(mm, ii, prec)\n        ces = [s[\'ce\'] for s in grp if s[\'ce\'] is not None and not np.isnan(s[\'ce\'])]\n        adducts = [s[\'adduct\'] for s in grp]\n        ad = max(adducts, key=adducts.count)          # deterministic tie-break (first occurrence)\n        items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad),\n                            ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,\n                            n_merged=min(sum(max(1, int(s.get(\'ce_n\', 1))) for s in grp), 8), mode=float(md)))\n    if not items_s:\n        return None\n    zs, _ = bank.logits_el(items_s)\n    zm, _ = bank.logits_el(items_m)\n    return (0.5 * (zs.mean(0) + zm.mean(0))).astype(np.float32)\n\n\n# ------------------------------------------------------------------------------------------ PubChem part (workers)\ndef init_worker(pc_dir, bits_path, pool_meta_path, code_dir=None):\n    import sys\n    if code_dir and code_dir not in sys.path:\n        sys.path.insert(0, code_dir)\n    import pandas as pd\n    from casmi import chem\n    _W[\'chem\'] = chem\n    _W[\'mass\'] = np.load(os.path.join(pc_dir, \'pc_mass.npy\'), mmap_mode=\'r\')\n    _W[\'off\'] = np.load(os.path.join(pc_dir, \'pc_off.npy\'), mmap_mode=\'r\')\n    _W[\'buf\'] = np.load(os.path.join(pc_dir, \'pc_smiles.npy\'), mmap_mode=\'r\')\n    bits = np.load(bits_path)\n    _W[\'bits\'] = bits\n    _W[\'sel_e\'] = np.where(bits < 4096)[0]\n    _W[\'raw_e\'] = bits[_W[\'sel_e\']]\n    _W[\'pool_keys\'] = set(pd.read_parquet(pool_meta_path, columns=[\'key\']).key.values.tolist())\n    r = chem._rdkit()\n    _W[\'ecfp4\'] = r[\'gen\'].GetMorganGenerator(radius=2, fpSize=4096)\n    _W[\'Chem\'] = r[\'Chem\']\n\n\ndef _window(target):\n    tol = target * PPM * 1e-6\n    a = int(np.searchsorted(_W[\'mass\'], target - tol, \'left\')); b = int(np.searchsorted(_W[\'mass\'], target + tol, \'right\'))\n    off = np.asarray(_W[\'off\'][a:b + 1]).astype(np.int64)\n    if b <= a:\n        return []\n    raw = bytes(_W[\'buf\'][off[0]:off[-1]])\n    base = off[0]\n    return [raw[off[i] - base:off[i + 1] - base].decode(\'ascii\') for i in range(b - a)]\n\n\ndef probe_one(task):\n    """task = (molecule_id, target_mass, z (nbits,) float32). Returns (molecule_id, [smiles], [f.z], [score keys], diag dict)."""\n    mid, target, z = task\n    t0 = time.time()\n    chem = _W[\'chem\']; Chem = _W[\'Chem\']; gen = _W[\'ecfp4\']\n    smis = _window(float(target))\n    n = len(smis)\n    diag = dict(molecule_id=mid, n_window=n, n_pass1=0, n_pass2=0, n_keyed=0, n_in_pool=0, n_listed=0)\n    if n == 0:\n        diag[\'secs\'] = time.time() - t0\n        return mid, [], [], [], diag\n    raw_e = _W[\'raw_e\']; z_e = z[_W[\'sel_e\']].astype(np.float32)\n    E = np.zeros((n, len(raw_e)), np.float32); ok = np.zeros(n, bool)\n    for i, s in enumerate(smis):\n        m = Chem.MolFromSmiles(s)\n        if m is None:\n            continue\n        E[i] = gen.GetFingerprintAsNumPy(m)[raw_e]; ok[i] = True\n    sc1 = E @ z_e\n    sc1[~ok] = -np.inf\n    n1 = min(PC_N1, int(ok.sum()))\n    diag[\'n_pass1\'] = n1\n    if n1 == 0:\n        diag[\'secs\'] = time.time() - t0\n        return mid, [], [], [], diag\n    top1 = np.argpartition(-sc1, n1 - 1)[:n1] if n1 < n else np.where(ok)[0]\n    bits = _W[\'bits\']\n    fz, idx = [], []\n    for i in top1:\n        fp = chem.raw_fingerprint(smis[i])\n        if fp is None:\n            continue\n        fz.append(float(fp[bits].astype(np.float32) @ z)); idx.append(int(i))\n    diag[\'n_pass2\'] = len(idx)\n    order = np.argsort(-np.asarray(fz), kind=\'stable\')\n    out, out_fz, out_k, seen = [], [], [], set()\n    for o in order:\n        s = smis[idx[o]]\n        k = chem.score_key(s)\n        diag[\'n_keyed\'] += 1\n        if k is None or k in seen:\n            continue\n        seen.add(k)\n        if k in _W[\'pool_keys\']:\n            diag[\'n_in_pool\'] += 1\n            continue\n        out.append(s); out_fz.append(float(fz[o])); out_k.append(k)\n        if len(out) >= K:\n            break\n    diag[\'n_listed\'] = len(out)\n    diag[\'secs\'] = time.time() - t0\n    return mid, out, out_fz, out_k, diag\n'
RUNNER = '"""PubChem-only channel for the final submission (runs in its own process so its casmi package\n(S1 snapshot, FPNet A+B) never collides with the v1 engine\'s casmi package).\n\nPer molecule: z = A+B logits (probe_core2.molecule_logits, identical to the probe that scored 0.037);\nbest_pool_fz = max f.z over the pool window (+-10 ppm, fallback 30 ppm) with the same z (as P1 logits.py);\nPubChem-only list = probe_core2.probe_one (top 25 by f.z, keys not in the pool) with f.z values and score keys.\nWrites a json {molecule_id: {pc, pc_fz, pc_keys, best_pool_fz, target}}.\nusage: python pc_runner.py <v3_code_dir> <v3_models_dir> <pool_dir> <pubchem_dir> <test_parquet> <out_json> <n_workers> [max_mols]\n"""\nimport os, sys, json, time\nimport numpy as np, pandas as pd\n\nif __name__ == \'__main__\':\n    V3CODE, V3MOD, POOL_DIR, PC_DIR, TEST, OUT, NW = sys.argv[1:8]\n    MAXM = int(sys.argv[8]) if len(sys.argv) > 8 else 0\n    NW = int(NW)\n    sys.path.insert(0, V3CODE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n    import torch, glob\n    import probe_core2 as pc\n    from casmi import fpnet, chem\n    T0 = time.time()\n    models = sorted(glob.glob(os.path.join(V3MOD, \'fpnet_*.pt\')))\n    assert len(models) == 2, models\n    dev = \'cuda\' if torch.cuda.is_available() else \'cpu\'\n    bank = fpnet.ModelBank(models, device=dev)\n    bits = np.load(os.path.join(POOL_DIR, \'fp_bits.npy\')); nbits = len(bits)\n    pmass = pd.read_parquet(os.path.join(POOL_DIR, \'pool_meta.parquet\'), columns=[\'mass\']).mass.values.astype(np.float64)\n    assert np.all(np.diff(pmass) >= 0), \'pool not mass-sorted\'\n    pfp = np.load(os.path.join(POOL_DIR, \'pool_fp.npy\'), mmap_mode=\'r\')\n    te = pd.read_parquet(TEST)\n    te[\'nm\'] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)\n    tasks, extra = [], {}\n    groups = list(te.groupby(\'molecule_id\', sort=False))\n    if MAXM:\n        groups = groups[:MAXM]\n    for mid, sub in groups:\n        nms = sub.nm.values[np.isfinite(sub.nm.values)]\n        if not len(nms):\n            continue\n        target = float(np.median(nms))\n        spectra = []\n        for r in sub.itertuples():\n            ce = r.collision_energy_ev\n            ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)\n            spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),\n                                mode=1 if r.ionization_mode == \'positive\' else -1, adduct=r.adduct,\n                                prec=float(r.precursor_mz), ce=float(ce_arr.mean()) if len(ce_arr) else np.nan,\n                                ce_n=int(len(ce_arr)) if len(ce_arr) else 1))\n        spectra = spectra[:16]\n        try:\n            z = pc.molecule_logits(bank, spectra)\n        except Exception as e:\n            print(\'ERROR logits\', mid, repr(e), flush=True); z = None\n        if z is None:\n            continue\n        tol = target * 10e-6\n        a = np.searchsorted(pmass, target - tol, \'left\'); b = np.searchsorted(pmass, target + tol, \'right\')\n        if b <= a:\n            tol = target * 30e-6\n            a = np.searchsorted(pmass, target - tol, \'left\'); b = np.searchsorted(pmass, target + tol, \'right\')\n        if b > a:\n            F = np.unpackbits(np.asarray(pfp[a:b]), axis=1)[:, :nbits].astype(np.float64)\n            best = float((F @ z.astype(np.float64)).max())\n        else:\n            best = float(\'nan\')\n        extra[mid] = dict(best_pool_fz=best, target=target)\n        tasks.append((mid, target, z))\n    del bank; torch.cuda.empty_cache()\n    print(f\'logits for {len(tasks)} molecules {time.time()-T0:.0f}s\', flush=True)\n    res = {}\n    import multiprocessing as mp\n    ctx = mp.get_context(\'fork\' if sys.platform != \'win32\' else \'spawn\')\n    with ctx.Pool(NW, initializer=pc.init_worker, initargs=(PC_DIR, os.path.join(POOL_DIR, \'fp_bits.npy\'),\n                                                            os.path.join(POOL_DIR, \'pool_meta.parquet\'), V3CODE)) as P:\n        for k, (mid, smis, fzs, keys, d) in enumerate(P.imap_unordered(pc.probe_one, tasks, chunksize=1)):\n            res[mid] = dict(pc=smis, pc_fz=fzs, pc_keys=keys, fz_top=d.get(\'fz_top\'), S=d.get(\'S\'), top_pop=d.get(\'top_pop\', 0.0), **extra[mid])\n            if (k + 1) % 25 == 0 or k + 1 == len(tasks):\n                print(f\'  pubchem {k+1}/{len(tasks)} {time.time()-T0:.0f}s\', flush=True)\n    json.dump(res, open(OUT, \'w\'))\n    print(\'pc_runner done\', len(res), f\'{time.time()-T0:.0f}s\', flush=True)\n'
open('/kaggle/working/probe_core2.py', 'w').write(CORE + V17_PATCH)
open('/kaggle/working/pc_runner.py', 'w').write(RUNNER)
print('v17 patch appended to probe_core2:', 'def probe_one' in V17_PATCH)
PC = {}
try:
    rr = subprocess.run([sys.executable, '/kaggle/working/pc_runner.py', V3CODE, V3MOD, POOL_DIR, PC_DIR,
                         os.path.join(COMP, 'test.parquet'), '/kaggle/working/pc_lists.json', '4'],
                        capture_output=True, text=True, timeout=4 * 3600)
    print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
    PC = json.load(open('/kaggle/working/pc_lists.json'))
except Exception as e:
    print('PUBCHEM CHANNEL FAILED -> base lists only:', repr(e))
print('pubchem lists', len(PC), f'{time.time()-T0:.0f}s')


    


# %% [code]

'''## 5. Engine 1: Ahmed Berat Özer's v4n

This step builds the spectral library from `train.parquet` and links the precomputed candidate pool: training structures plus COCONUT, about 711k in total. The pool comes with FPNet fingerprints and fragment tables.
'''


    


# %% [code]

'''v1 engine code (the 0.358 model's engine) + fe_v4; pool symlinks; spectrum cache from train.parquet'''
sys.path.insert(0, V4CODE)
os.makedirs('work', exist_ok=True)
for f in ['pool_meta.parquet', 'pool_fp.npy', 'pool_frag_off.npy', 'pool_frag_mass.npy', 'fp_bits.npy', 'train_structs.parquet', 'train_fp_sel.npy']:
    src = os.path.join(POOL_DIR, f)
    dst = f'work/{f}'
    if not os.path.exists(dst):
        os.symlink(src, dst)
from casmi.build import build_spec_cache
build_spec_cache(os.path.join(COMP, 'train.parquet'), 'work')
import casmi; print('casmi from', casmi.__file__)
sys.path.insert(1, os.path.join(V4CODE, 'fe_v4'))
print(f'cache built {time.time()-T0:.0f}s')


    


# %% [code]

'''v1 engine exactly as in v3.6 (FPNet A, EngineCfg(generate=True)) wrapped by V1FE (recording hooks + families)'''
import numpy as np, pandas as pd, torch, lightgbm as lgb
from casmi.library import Library
from casmi.pool import Pool
from casmi.engine import Engine, EngineCfg, FEATURES
from casmi import fpnet, chem
import v1engine
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
L = Library('work'); P = Pool('work')
tfp = np.load('work/train_fp_sel.npy')
_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)   # v4m
bank = fpnet.ModelBank([_FULL[0] if _FULL else os.path.join(V4MOD, 'fpnet_0.pt')], device=dev)
print('engine FP bank:', _FULL[0] if _FULL else 'fpnet_0 (A) FALLBACK', flush=True)
E = Engine(L, P, tfp, EngineCfg(generate=True), bank)
V = v1engine.V1FE(E, [os.path.join(V4MOD, m['file']) for m in MAN['fe_models']], MAN['families'], device=dev)
if 'derivation' in MAN['families']:
    n_bad = V.mods['derivation'].self_test()
    print('derivation prior self-test mismatches:', n_bad)
    assert n_bad == 0, 'RDKit build differs from the one the derivation priors were built with'
RK = pickle.load(open(os.path.join(V4MOD, MAN['ranker']['file']), 'rb'))
RK_FEATS = list(RK['features']); RK_B = [lgb.Booster(model_str=s) for s in RK['boosters']]
def rank_score(X, names):
    pos = {n: i for i, n in enumerate(names)}
    miss = [f for f in RK_FEATS if f not in pos]
    assert not miss, f'ranker features not produced (would be zero-filled): {miss}'
    Xs = np.ascontiguousarray(X[:, [pos[f] for f in RK_FEATS]], dtype=np.float32)
    return np.mean([b.predict(Xs, num_threads=4) for b in RK_B], axis=0)
print('engine ready; ranker features', len(RK_FEATS), 'boosters', len(RK_B), f'{time.time()-T0:.0f}s')


    


# %% [code]

'''## 6. Main candidate lists + pool popularity re-rank

For each molecule, v4n scores candidates with FPNet fingerprints, its feature families and a 4-booster LightGBM ranker, and keeps the top 60. The list is then re-ranked by

$$z(\\text{ranker score}) + 0.15\\,\\big(\\log(1+\\#\\text{substances}) + \\log(1+\\#\\text{papers})\\big)$$

Only pool structures are re-ranked. Generated (novel) candidates keep their positions and form their own simulator groups. This step added +0.010 on the leaderboard.
'''


    


# %% [code]

'''all molecules -> base lists (v3.6 spectra construction and ordering)'''
te = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
te['nm'] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
mols = list(te.groupby('molecule_id', sort=False))
BASE, n_err = {}, 0
POOLPOP_MU = 0.15
POOL_POP = np.load(find('pool_lsid.npy')).astype(np.float64) + np.load(find('pool_lpmid.npy')).astype(np.float64)
assert len(POOL_POP) == len(P.key), (len(POOL_POP), len(P.key))
n_pp = 0
print('v19 pool popularity re-rank, mu', POOLPOP_MU, 'pool', len(POOL_POP))
for gi, (mid, sub) in enumerate(mols):
    nms = sub.nm.values[np.isfinite(sub.nm.values)]
    smis, keys, lib_max, scs, forms = [], [], 0.0, [], []
    pids_ = []
    if len(nms):
        target = float(np.median(nms))
        spectra = []
        for r in sub.itertuples():
            ce = r.collision_energy_ev
            ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
            spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                                mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct,
                                prec=float(r.precursor_mz), ce=float(ce_arr.mean()) if len(ce_arr) else np.nan,
                                ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
        try:
            C, X, F, names, info = V.run(spectra, target)
            lib_max = float(info.get('lib_max', 0.0))
            if len(C.pid):
                score = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))
                order = np.argsort(-score, kind='stable')
                seen = set()
                for i in order:
                    k = C.key[i]
                    if k in seen: continue
                    seen.add(k); smis.append(C.smiles[i]); keys.append(k); scs.append(float(score[i])); forms.append(C.formula[i]); pids_.append(int(C.pid[i]))
                    if len(smis) >= TOPN: break
        except Exception as e:
            print('ERROR', mid, repr(e)); n_err += 1
    if POOLPOP_MU > 0 and len(scs) > 1:            # v19: popularity re-rank of the top-60 base list
        try:
            s_ = np.asarray(scs, np.float64); z_ = (s_ - s_.mean()) / (s_.std() + 1e-9)
            pid_ = np.asarray(pids_); isg_ = pid_ < 0
            f_ = z_ + POOLPOP_MU * np.where(isg_, 0.0, POOL_POP[np.maximum(pid_, 0)])
            pool_i = np.where(~isg_)[0]; o_ = np.empty(len(pid_), np.int64)
            o_[isg_] = np.where(isg_)[0]; o_[~isg_] = pool_i[np.argsort(-f_[pool_i], kind='stable')]   # generated keep their slots
            smis = [smis[i] for i in o_]; keys = [keys[i] for i in o_]; scs = [float(f_[i]) for i in o_]
            forms = [forms[i] + ('|gen' if isg_[i] else '') for i in o_]      # generated form their own ICEBERG groups
            n_pp += 1
        except Exception as e:
            print('poolpop failed', mid, repr(e))
    BASE[mid] = (smis, keys, lib_max, scs, forms)
    if (gi + 1) % 25 == 0 or gi == len(mols) - 1:
        print(f'  {gi+1}/{len(mols)} molecules {time.time()-T0:.0f}s', flush=True)
print('errors', n_err, '| popularity re-ranked lists', n_pp)
'''Keep ranked overflow for append-only final coverage.'''
BASE_TAIL_CANDIDATES = {str(k): list(v[0]) for k, v in BASE.items()}


    


# %% [code]

'''## 7. ICEBERG + GLACIER spectrum simulators

Both models predict each candidate's spectrum and compare it with the measured one. Candidates with the same molecular formula are then re-ranked by

$$z(\\text{score}) + 1.0\\,z(\\text{ICEBERG}) + 1.0\\,z(\\text{GLACIER})$$

Most wrong answers share the right answer's formula, so this is where isomers get separated.

- Second-engine candidates join the simulator input, so they are scored too.
- **GLACIER uses [M+H]+ spectra only.** Restricting it this way added +0.006 on the leaderboard.
'''


    


# %% [code]

'''--- CLAW FIX (b418/b419): stage health is now RECORDED, never silently swallowed ---'''
'''`glob('/kaggle/input/**/x')` WITHOUT recursive=True matches one directory level only, so it'''
'''misses /kaggle/input/datasets/<owner>/<slug>/. That silently disabled ICEBERG+GLACIER:'''
'''ICE FAILED -> ranker order kept: FileNotFoundError('cp313 rdkit wheel not found')  [b417 log]'''
'''rdkit-2025.3.6-cp312-...whl is not a supported wheel on this platform               [b417 log]'''
'''recursive=True finds the wheel under BOTH mount layouts (same 2025.3.6 cp313 file either way).'''
CLAW_ICE_HEALTH = 'ICE NOT RUN'
CLAW_GL_HEALTH = 'GL NOT RUN'
BASE_ICE = False         # b421: base-list ICEBERG fusion OFF (b420 showed it costs 0.009); see header
'''ICEBERG post-ranker isomer re-scoring (casmi26-iceberg). Optional: any failure keeps the ranker order.'''
import gc
ICE_SCORES, ice_stats = {}, dict(molecules=0, changed_top25=0, changed_top1=0)


'''--- CLAW FIX (b415c): make ICEBERG/GLACIER runnable on a cp313 Kaggle image ---'''
'''The ICEBERG package bundles rdkit-2025.3.6 built for **cp312**; this image is cp313,'''
'''so pip rejects that wheel and install_site() raises before anything is scored'''
'''(n_mols_scored == 0 for ICEBERG *and* GLACIER). The MANIFEST pins 2025.3.6 because'''
'''"FragmentEngine Kekulize differs in 2026.03", so swapping in the runtime's 2026.03'''
'''rdkit would change the scoring. Instead we install the SAME 2025.3.6 release built'''
'''for cp313, so the pinned version is honoured exactly and RDKIT_REQ passes natively.'''
''''''
'''install_site() is idempotent via <site>/.ice_site_ok, which it compares against the'''
'''name list of `pkg/wheels`. We pre-build the site and write that exact list, so both'''
'''runners short-circuit and never invoke the failing pip.'''
def _claw_preseed_site(pkg, site):
    import glob as _g, os as _o, shutil as _s, subprocess as _sp, sys as _sys
    srcs = sorted(_g.glob(_o.path.join(pkg, 'wheels', '*.whl.ice')) +
                  _g.glob(_o.path.join(pkg, 'wheels', '*.whl')))
    # install everything EXCEPT the cp-tag-mismatched rdkit
    keep = [w for w in srcs if not _o.path.basename(w).lower().startswith('rdkit')]
    dropped = [_o.path.basename(w) for w in srcs if w not in keep]
    new = [simulator_rdkit_wheel(pkg)]
    keep = keep + new[:1]
    _o.makedirs(site, exist_ok=True)
    wdir = _o.path.join(site, '.wheels')
    _o.makedirs(wdir, exist_ok=True)
    whl = []
    for w in keep:
        dst = _o.path.join(wdir, _o.path.basename(w)[:-4] if w.endswith('.ice') else _o.path.basename(w))
        _s.copyfile(w, dst)
        whl.append(dst)
    r = _sp.run([_sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps',
                 '--no-compile', '--disable-pip-version-check', '--upgrade', '--target', site] + whl,
                stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
    print('[claw] site preseed rc', r.returncode, (r.stdout or '')[-400:], flush=True)
    if r.returncode != 0:
        raise RuntimeError('claw preseed pip failed: ' + (r.stdout or '')[-600:])
    _s.rmtree(wdir, ignore_errors=True)
    # the marker MUST equal exactly what install_site() derives from the PACKAGE list
    want = '\n'.join(_o.path.basename(w) for w in srcs)
    with open(_o.path.join(site, '.ice_site_ok'), 'w') as f:
        f.write(want)
    print('[claw] site ready; substituted', dropped, '->', [_o.path.basename(w) for w in new[:1]], flush=True)
    return want


try:
    ICE_PKG = os.path.dirname(find('casmi26-iceberg/**/ice_runner.py'))
    sys.path.append(ICE_PKG)
    import fuse as ice_fuse
    del V, E, bank; gc.collect(); torch.cuda.empty_cache()
    # priority: molecules without a library-level match first (class 1 rarely changes), then the rest
    mids = sorted(BASE, key=lambda m: BASE[m][2])
    cands = {m: (ice_fuse.ice_candidates(BASE[m][0], BASE[m][4], TOPN) if BASE_ICE else []) for m in mids}
    if ENG:                              # ADDED (ICE_UNION): also score our engine's candidates (same-formula groups
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _CMF   # formed over the v4g top-N + engine top-40)
        from rdkit import Chem as _Ch
        def _f2(s):
            try:
                return _CMF(_Ch.MolFromSmiles(s))
            except Exception:
                return s
        n_add = 0
        for m in mids:
            e = ENG.get(str(m))
            if not e or not e.get('smiles'):
                continue
            bs, bk, bf = list(BASE[m][0][:TOPN]), set(BASE[m][1][:TOPN]), list(BASE[m][4][:TOPN])
            es = [s for s, k in zip(e['smiles'][:40], e['keys'][:40]) if k not in bk]
            ef = [_f2(s) for s in es]
            extra = ice_fuse.ice_candidates(bs + es, bf + ef, len(bs) + len(es))
            before = len(cands.get(m, []))
            cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + extra))
            n_add += len(cands[m]) - before
        print('ICE_UNION: engine candidates added to ICE input:', n_add, flush=True)
    if ICE_PC:                          # also the PubChem-only list of molecules the v3.6 gate may use (lib_max < LIB_TAU)
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        from rdkit import Chem as _Chem
        def _form(s):
            try:
                return CalcMolFormula(_Chem.MolFromSmiles(s))
            except Exception:
                return s
        for m in mids:
            p = PC.get(m) if isinstance(PC, dict) else None
            if p and p.get('pc') and BASE[m][2] < LIB_TAU:
                p['pc_form'] = [_form(x) for x in p['pc']]
                extra = ice_fuse.ice_candidates(p['pc'], p['pc_form'], 25)
                cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + extra))
    cands = {m: c for m, c in cands.items() if c}
    items = ice_fuse.build_ice_input(te, cands)
    print('ICE input molecules', len(items), 'candidates', sum(len(it['cands']) for it in items), f'{time.time()-T0:.0f}s', flush=True)
    _claw_preseed_site(ICE_PKG, '/kaggle/working/ice_site')
    ICE_SCORES = ice_fuse.run_ice(ICE_PKG, items, workdir='/kaggle/working/ice_work', device='cuda', budget_s=ICE_BUDGET,
                                  site='/kaggle/working/ice_site', extra_args=['--any-rdkit'])
    meta_f = '/kaggle/working/ice_work/ice_out.json.meta.json'
    if os.path.exists(meta_f):
        print('ICE meta', open(meta_f).read()[:1500])
except Exception as e:
    CLAW_ICE_HEALTH = f'ICE FAILED: {e!r}'
    print('ICE FAILED -> ranker order kept:', repr(e), flush=True)
'''v4h: GLACIER third re-scoring term (casmi26-glacier) on the SAME runner input; any GLACIER failure -> v4g (ICE only)'''
GL_SCORES, gl_stats, GL_LAM, GL_BUDGET = {}, dict(molecules=0, changed_vs_ice_top25=0, changed_vs_ice_top1=0), 1.0, 4000
try:
    GL_PKG = os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))
    sys.path.append(GL_PKG)
    import gl_fuse
    gc.collect(); torch.cuda.empty_cache()   # ICE ran in its own (finished) process; free the notebook's cached GPU memory
    gl_items = [dict(it, spectra=[s for s in it['spectra'] if s.get('adduct') == '[M+H]+']) for it in items]   # v29: GLACIER on [M+H]+ only (v27: +0.006)
    gl_items = [it for it in gl_items if it['spectra']]
    print('v29 GLACIER [M+H]+ only: molecules', len(gl_items), 'of', len(items))
    GL_SCORES = gl_fuse.run_gl(GL_PKG, gl_items, workdir='/kaggle/working/gl_work', device='cuda', budget_s=GL_BUDGET,
                               site='/kaggle/working/ice_site', wheels=os.path.join(ICE_PKG, 'wheels'), extra_args=['--any-rdkit'])
    meta_f = '/kaggle/working/gl_work/gl_out.json.meta.json'
    print('GL meta', open(meta_f).read()[:1500] if os.path.exists(meta_f) else 'missing')
except Exception as e:
    GL_SCORES = {}
    CLAW_GL_HEALTH = f'GL FAILED: {e!r}'
    print('GL FAILED -> v4g behaviour (ICE only):', repr(e), flush=True)
# ---- N1A_MCE (gold roadmap 2026-10-09 section 3.2: N1 arm A + I14b merged CE on the proxies only) ---------------
# Molecules WITHOUT a real [M+H]+ spectrum are GLACIER-blind in H1. N1 arm A scores their [M+NH4]+ / [M-H2O+H]+
# spectra as [M+H]+ proxies (NH4: M = prec - 18.033823, peaks >= M + 1.007276 - 1.5 dropped; water loss:
# M = prec + 18.010565 - 1.007276, peaks < prec - 1.5 kept; precursor -> M + 1.007276, relabelled [M+H]+;
# [M-2H2O+H]+ / Na / K never), and a merged-CE proxy (>= 2 distinct CE buckets) is predicted at EVERY bucket and the
# predictions merge()d like the SUM_CE path (I14b). = bench kdev_glbench_n1 arm N1A_D on the blind molecules (521
# bench molecules: 21 blind molecules with a proxy, lift over random +0.270 [0.117, 0.443]; vs H1 on all np molecules
# +0.027 [0.010, 0.047]; merged vs mean CE on those 21 +0.0026). Arm B (proxies also where [M+H]+ exists, -0.047) and
# merged CE for real [M+H]+ spectra (no gain) are NOT done: molecules WITH a real [M+H]+ keep the H1 run above
# untouched (stock gl_runner, mean CE, byte-identical input). The bench's own code runs here, embedded verbatim
# (sha256-checked): glb_core.py (n1_proxy / build_variants / mol_flags / arm_tags / arm_scores) in this process,
# glb_driver.py (driver around the attached gl_runner.py: same env, model, Predictor, canonicalisation, CE bucketing,
# merge, entropy similarity) in its own process like gl_runner. N1 scores are added to GL_SCORES (-> the PubChem-list
# re-rank below and the fused re-rank of cell 19, whose `if ice:` gate is opened ONLY for N1 molecules with scores).
# Any failure -> no N1 scores = H1 behaviour (the commit raises instead). The 12 dummy commit molecules never
# trigger N1, so the commit adds a forced exercise: synthetic [M+H]+-less copies 'n1x:<mid>' ([M+H]+ -> [M+NH4]+ with
# an injected NH4 precursor peak, the first one also -> [M-H2O+H]+) run through the same code + asserts.
N1_ARM = 'N1A_D'                 # bench arm: blind molecule -> spectrum tags (NH4p, WLp), CE mode 'merged'
N1_BUDGET = 1200.0 if IS_RERUN else 600.0
N1_WORK, N1_SITE, N1_DEVICE, N1_DRIVER_EXTRA = '/kaggle/working/n1_work', '/kaggle/working/ice_site', 'cuda', []
N1_MIDS, N1_GL_MIDS, N1_STATS, N1_DRV, N1_HEALTH = set(), set(), {}, {}, 'N1 NOT RUN'
N1X_MIDS, N1X_GL, N1X_STATS = set(), {}, {}
# k2 (E006 port), variant A = N1-complement. E006 (cell 31) re-ranks a GLACIER-blind molecule with ICEBERG when its ICE
# dict has finite scores (in practice [M+Na]+ without [M+H]+: `elif ice_fallback and ice and not gl`). N1 GL scores
# would move such a molecule to the GLACIER branch, so they are kept ONLY for N1 molecules WITHOUT any finite ICE
# score (E006 has no forward-model evidence there: its ICE fallback on an all-None dict is a no-op). False = the plain
# port (N1 GL scores for every N1 molecule, E2 behaviour). Molecules with a real [M+H]+ are never touched either way.
N1_COMPLEMENT = True
N1_ICE_DROPPED = set()           # N1 molecules left to E006's ICE fallback by the complement filter
N1_CORE_SHA256 = 'f7a3b8cd8802a7ba3b69a297603241784a63c069044cf7bd6efda7d68cc8fdfa'      # = subs/out/kdev_glbench_n1_full/glb_core.py
N1_DRIVER_SHA256 = 'e919a154b57bda5a52573481348e38d8b4e7c97883532ea5b385bc3cb6f98502'    # = subs/out/kdev_glbench_n1_full/glb_driver.py
N1_GL_RUNNER_SHA256 = '5c4f74313ac54263d05d987c39d012d81ec57c5f35e1095ecd49a5bcb6440fca'   # casmi26-glacier gl_runner.py (B1 MANIFEST, bench)
# kdev_glbench_n1 glb_core.py, verbatim (proxy construction, variants, arm resolution, aggregation)
N1_CORE_SRC = ''.join([
    '"""casmi26 kdev_glbench_n1 -- pure-python core of the roadmap 3.2 GLACIER bench (no torch / rdkit import).\n',
    '\n',
    'Query handling (N1, roadmap 3.2), on fuse.build_ice_input(..., covered_only=False) spectrum dicts:\n',
    '  [M+NH4]+    M = prec - 18.033823; prec -> M + 1.007276; drop peaks mz >= M + 1.007276 - 1.5; relabel [M+H]+\n',
    '  [M-H2O+H]+  M = prec + 18.010565 - 1.007276; prec -> M + 1.007276; keep peaks mz < original prec - 1.5; relabel\n',
    '  (a proxy left without peaks is dropped; [M-2H2O+H]+, [M+Na]+, [M+K]+ are never proxied)\n',
    '  Hcut (diagnostic): a real [M+H]+ spectrum with the same precursor-region cut (peaks mz >= prec - 1.5 dropped), the\n',
    '  like-for-like reference for the proxies (separates "different adduct" from "precursor region removed").\n',
    '\n',
    'Scoring: the driver returns per-spectrum GLACIER similarities for the frozen bench candidates in two CE modes\n',
    "('mean' = production gl_runner: one prediction at the bucket of mean(ce_ev); 'merged' = I14b: one prediction per\n",
    "distinct CE bucket of a merged-CE spectrum, merged like the SUM_CE path). A molecule's arm score per candidate =\n",
    "mean over the arm's spectra with a similarity = gl_ex_mean restricted to those spectra.\n",
    '\n',
    'Ranking metric: expected reciprocal rank of the truth (cands[mid][0]) under a UNIFORM RANDOM tie-break (no truth-first\n',
    'bias: a truth tied with k decoys gets mean(1/(r..r+k))); unscored candidates rank below every scored one (tied among\n',
    'themselves). Baseline = a random order of all frozen candidates, mean(1/1..1/n). An arm "covers" a molecule when the\n',
    'truth and >= 2 candidates are scored; elsewhere the arm has no evidence (= the random baseline in fill-in totals).\n',
    '"""\n',
    'import math\n',
    'from collections import Counter, OrderedDict\n',
    '\n',
    'import numpy as np\n',
    '\n',
    'PROTON = 1.007276\n',
    'NH4_ADD = 18.033823           # [M+NH4]+ precursor m/z - M\n',
    'H2O = 18.010565\n',
    'CUT_DA = 1.5\n',
    'MIN_PEAKS = 1\n',
    "PROXY_TAG = OrderedDict([('[M+NH4]+', 'NH4p'), ('[M-H2O+H]+', 'WLp')])\n",
    'EPS_TIE = 1e-12\n',
    'N_BOOT = 2000\n',
    'BOOT_SEED = 12345\n',
    '\n',
    "# arm -> (spectrum tags, CE mode, description); 'N1A' is resolved per molecule (see arm_tags)\n",
    'ARMS = OrderedDict([\n',
    "    ('A_H', (('H',), 'mean', '(a) [M+H]+ only, mean CE = B1 production GL')),\n",
    "    ('B_NH4', (('NH4p',), 'mean', '(b) [M+NH4]+ proxies relabelled [M+H]+')),\n",
    "    ('C_WL', (('WLp',), 'mean', '(c) [M-H2O+H]+ proxies relabelled [M+H]+')),\n",
    "    ('D_Hmerged', (('H',), 'merged', '(d) [M+H]+ only, merged-CE per-bucket prediction (I14b)')),\n",
    "    ('A_Hcut', (('Hcut',), 'mean', 'diag: [M+H]+ with the proxy-style precursor cut')),\n",
    "    ('B_NH4m', (('NH4p',), 'merged', 'diag: NH4 proxies, merged CE')),\n",
    "    ('C_WLm', (('WLp',), 'merged', 'diag: water-loss proxies, merged CE')),\n",
    "    ('HNa', (('H', 'Na'), 'mean', 'diag: [M+H]+ and [M+Na]+ (old fwdbench COV; P4 analogue)')),\n",
    "    ('AB', (('H', 'NH4p', 'WLp'), 'mean', 'diag: N1 arm B = [M+H]+ plus all proxies')),\n",
    "    ('N1A', (None, 'mean', 'N1 arm A: [M+H]+ if the molecule has one, else its proxies')),\n",
    "    ('N1A_D', (None, 'merged', 'diag: N1 arm A with merged CE (N1A + I14b)')),\n",
    '])\n',
    '\n',
    '\n',
    '# ------------------------------------------------------------------------------------------------ query handling --\n',
    'def _keep_below(sp, cut, min_peaks):\n',
    "    mz, it = sp.get('mz') or [], sp.get('it') or []\n",
    '    keep = [(float(m), float(i)) for m, i in zip(mz, it) if float(m) < cut]\n',
    '    if len(keep) < min_peaks or not any(i > 0 for _, i in keep):\n',
    '        return None\n',
    '    return [m for m, _ in keep], [i for _, i in keep]\n',
    '\n',
    '\n',
    'def n1_proxy(sp, min_peaks=MIN_PEAKS):\n',
    '    """[M+NH4]+ / [M-H2O+H]+ positive spectrum dict -> relabelled [M+H]+ proxy dict, or None."""\n',
    "    if not isinstance(sp, dict) or sp.get('mode') != 'positive':\n",
    '        return None\n',
    "    ad = sp.get('adduct')\n",
    '    try:\n',
    "        prec = float(sp['prec'])\n",
    '    except Exception:\n',
    '        return None\n',
    '    if not math.isfinite(prec):\n',
    '        return None\n',
    "    if ad == '[M+NH4]+':\n",
    '        M = prec - NH4_ADD\n',
    '        cut = M + PROTON - CUT_DA\n',
    "    elif ad == '[M-H2O+H]+':\n",
    '        M = prec + H2O - PROTON\n',
    '        cut = prec - CUT_DA\n',
    '    else:\n',
    '        return None\n',
    '    k = _keep_below(sp, cut, min_peaks)\n',
    '    if k is None:\n',
    '        return None\n',
    '    out = dict(sp)\n',
    "    out.update(mz=k[0], it=k[1], prec=M + PROTON, adduct='[M+H]+', mode='positive', n1_src=ad)\n",
    '    return out\n',
    '\n',
    '\n',
    'def h_cut(sp, min_peaks=MIN_PEAKS):\n',
    '    """Real [M+H]+ spectrum with the proxy-style cut (peaks mz >= prec - 1.5 dropped); None if nothing is left."""\n',
    "    if not isinstance(sp, dict) or sp.get('mode') != 'positive' or sp.get('adduct') != '[M+H]+':\n",
    '        return None\n',
    "    k = _keep_below(sp, float(sp['prec']) - CUT_DA, min_peaks)\n",
    '    if k is None:\n',
    '        return None\n',
    '    return dict(sp, mz=k[0], it=k[1])\n',
    '\n',
    '\n',
    'def build_variants(items_all):\n',
    '    """fuse.build_ice_input(Q, cands, covered_only=False) items -> driver items whose spectra carry sid / tag\n',
    '    (tags H, Hcut, Na, NH4p, WLp), plus counts. sid = \'<index of the spectrum in the molecule>:<tag>\'."""\n',
    '    out, st = [], Counter()\n',
    '    for it in items_all:\n',
    '        sps = []\n',
    "        for j, sp in enumerate(it.get('spectra') or []):\n",
    "            if sp.get('mode') != 'positive':\n",
    '                continue\n',
    "            ad = sp.get('adduct')\n",
    "            if ad == '[M+H]+':\n",
    "                sps.append(dict(sp, sid=f'{j}:H', tag='H')); st['H'] += 1\n",
    '                c = h_cut(sp)\n',
    '                if c is None:\n',
    "                    st['Hcut_empty'] += 1\n",
    '                else:\n',
    "                    sps.append(dict(c, sid=f'{j}:Hcut', tag='Hcut')); st['Hcut'] += 1\n",
    "            elif ad == '[M+Na]+':\n",
    "                sps.append(dict(sp, sid=f'{j}:Na', tag='Na')); st['Na'] += 1\n",
    '            elif ad in PROXY_TAG:\n',
    '                tag = PROXY_TAG[ad]\n',
    '                p = n1_proxy(sp)\n',
    '                if p is None:\n',
    "                    st[tag + '_empty'] += 1\n",
    '                else:\n',
    "                    sps.append(dict(p, sid=f'{j}:{tag}', tag=tag)); st[tag] += 1\n",
    "        out.append(dict(mid=str(it['mid']), cands=list(it['cands']), spectra=sps))\n",
    '    return out, dict(st)\n',
    '\n',
    '\n',
    'def proxy_prec_check(items):\n',
    '    """Real-data check of the adduct arithmetic: proxy precursor (M + 1.007276) vs the molecule\'s real [M+H]+\n',
    '    precursor m/z (median over its [M+H]+ spectra), in mDa, per proxy tag."""\n',
    '    out = {}\n',
    '    for tag in PROXY_TAG.values():\n',
    '        d = []\n',
    '        for it in items:\n',
    "            H = [float(sp['prec']) for sp in it['spectra'] if sp.get('tag') == 'H']\n",
    '            if H:\n',
    "                d += [1000.0 * (float(sp['prec']) - float(np.median(H))) for sp in it['spectra'] if sp.get('tag') == tag]\n",
    '        out[tag] = dict(n=len(d), median_mda=float(np.median(d)) if d else None,\n',
    '                        max_abs_mda=float(np.max(np.abs(d))) if d else None)\n',
    '    return out\n',
    '\n',
    '\n',
    '# ---------------------------------------------------------------------------------------------------- ranking ----\n',
    'def rr_random_tb(vals, ti=0, eps=EPS_TIE):\n',
    '    """Expected reciprocal rank of vals[ti] under a uniform random tie-break; None entries rank below all scored\n',
    '    ones. None if vals[ti] is None."""\n',
    '    t = vals[ti]\n',
    '    if t is None:\n',
    '        return None\n',
    '    gt = eq = 0\n',
    '    for j, v in enumerate(vals):\n',
    '        if j == ti or v is None:\n',
    '            continue\n',
    '        if v > t + eps:\n',
    '            gt += 1\n',
    '        elif abs(v - t) <= eps:\n',
    '            eq += 1\n',
    '    return sum(1.0 / (gt + 1 + k) for k in range(eq + 1)) / (eq + 1)\n',
    '\n',
    '\n',
    'def rr_truth_first(vals, ti=0, eps=EPS_TIE):\n',
    '    """Old-bench convention with the truth winning every tie (the bias this bench removes)."""\n',
    '    t = vals[ti]\n',
    '    if t is None:\n',
    '        return None\n',
    '    return 1.0 / (1 + sum(1 for j, v in enumerate(vals) if j != ti and v is not None and v > t + eps))\n',
    '\n',
    '\n',
    'def rr_split(vals, ti=0, eps=EPS_TIE):\n',
    '    """fwdbench bench_core.mrr_table convention (ties count half)."""\n',
    '    t = vals[ti]\n',
    '    if t is None:\n',
    '        return None\n',
    '    gt = sum(1 for j, v in enumerate(vals) if j != ti and v is not None and v > t + eps)\n',
    '    eq = sum(1 for j, v in enumerate(vals) if j != ti and v is not None and abs(v - t) <= eps)\n',
    '    return 1.0 / (1 + gt + 0.5 * eq)\n',
    '\n',
    '\n',
    'def rr_rand(n):\n',
    "    return sum(1.0 / k for k in range(1, n + 1)) / n if n > 0 else float('nan')\n",
    '\n',
    '\n',
    'def covered(vals, ti=0):\n',
    '    return vals[ti] is not None and sum(v is not None for v in vals) >= 2\n',
    '\n',
    '\n',
    '# ------------------------------------------------------------------------------------------------ aggregation ----\n',
    'def mol_flags(item):\n',
    "    tags = Counter(sp['tag'] for sp in item['spectra'])\n",
    "    return dict(has_H=tags['H'] > 0, has_NH4p=tags['NH4p'] > 0, has_WLp=tags['WLp'] > 0, has_Na=tags['Na'] > 0)\n",
    '\n',
    '\n',
    'def arm_tags(arm, flags):\n',
    '    tags, mode, _ = ARMS[arm]\n',
    '    if tags is None:                                        # N1 arm A: proxies only where no real [M+H]+ exists\n',
    "        tags = ('H',) if flags['has_H'] else ('NH4p', 'WLp')\n",
    '    return tags, mode\n',
    '\n',
    '\n',
    'def arm_scores(mol_out, tags, mode):\n',
    '    """mol_out = driver output of one molecule {\'cands\': [...], \'spec\': {sid: {\'tag\',\'nb\',\'mean\',\'merged\'}}} ->\n',
    '    per-candidate list (aligned with mol_out[\'cands\']) of the mean similarity over the arm\'s spectra (None if none)."""\n',
    "    cands = mol_out.get('cands') or []\n",
    "    cols = [s[mode] for sid, s in sorted((mol_out.get('spec') or {}).items(), key=lambda kv: _sid_key(kv[0]))\n",
    "            if s.get('tag') in tags and s.get(mode) is not None]\n",
    '    out = []\n',
    '    for i in range(len(cands)):\n',
    '        xs = [c[i] for c in cols if i < len(c) and c[i] is not None]\n',
    '        out.append(float(np.mean(xs)) if xs else None)\n',
    '    return out\n',
    '\n',
    '\n',
    'def _sid_key(sid):\n',
    "    a, _, b = str(sid).partition(':')\n",
    '    try:\n',
    '        return (int(a), b)\n',
    '    except ValueError:\n',
    '        return (10 ** 9, str(sid))\n',
    '\n',
    '\n',
    'def molecule_table(items, out, done):\n',
    '    """-> list of row dicts, one per molecule the driver finished: rr per arm (None = arm does not cover it)."""\n',
    '    rows = []\n',
    '    for it in items:\n',
    "        mid = it['mid']\n",
    '        if mid not in done:\n',
    '            continue\n',
    "        mo = out.get(mid) or {'cands': it['cands'], 'spec': {}}\n",
    "        assert list(mo.get('cands') or it['cands']) == list(it['cands']), ('candidate order changed', mid)\n",
    '        fl = mol_flags(it)\n',
    "        n = len(it['cands'])\n",
    "        nb_h = [s.get('nb', 1) for s in (mo.get('spec') or {}).values() if s.get('tag') == 'H']\n",
    "        r = dict(mid=mid, set=mid.split(':')[0], n=n, rand=rr_rand(n), has_merged_H=any(b >= 2 for b in nb_h), **fl)\n",
    '        for arm in ARMS:\n',
    '            tags, mode = arm_tags(arm, fl)\n',
    '            v = arm_scores(mo, tags, mode)\n',
    "            r['rr_' + arm] = rr_random_tb(v) if covered(v) else None\n",
    "            if arm == 'A_H':\n",
    "                r['rrtf_A_H'] = rr_truth_first(v) if covered(v) else None\n",
    "                r['rrsp_A_H'] = rr_split(v) if covered(v) else None\n",
    "                r['top1_A_H'] = (None if not covered(v) else float(rr_random_tb(v) == 1.0))\n",
    '        rows.append(r)\n',
    '    return rows\n',
    '\n',
    '\n',
    '# ----------------------------------------------------------------------------------------------------- stats -----\n',
    'def _boot_idx(n, seed=BOOT_SEED, nb=N_BOOT):\n',
    '    return np.random.default_rng(seed).integers(0, n, (nb, n))\n',
    '\n',
    '\n',
    "def paired(rows, a, b, sel=None, label=''):\n",
    '    """Paired comparison of arms a and b over the rows both cover (and sel(row) if given)."""\n',
    "    R = [r for r in rows if r['rr_' + a] is not None and r['rr_' + b] is not None and (sel is None or sel(r))]\n",
    '    n = len(R)\n',
    '    res = dict(label=label, a=a, b=b, n=n)\n',
    '    if n == 0:\n',
    '        return res\n',
    "    ra = np.array([r['rr_' + a] for r in R]); rb = np.array([r['rr_' + b] for r in R])\n",
    "    rn = np.array([r['rand'] for r in R])\n",
    '    d = rb - ra\n',
    '    idx = _boot_idx(n)\n',
    '    la, lb = ra - rn, rb - rn\n',
    "    with np.errstate(divide='ignore', invalid='ignore'):\n",
    '        bo_ratio = lb[idx].mean(1) / la[idx].mean(1)\n',
    "    ratio = float(lb.mean() / la.mean()) if la.mean() > 0 else float('nan')\n",
    '    res.update(mrr_a=float(ra.mean()), mrr_b=float(rb.mean()), rand=float(rn.mean()), lift_a=float(la.mean()),\n',
    '               lift_b=float(lb.mean()), delta=float(d.mean()),\n',
    '               delta_ci=[float(x) for x in np.percentile(d[idx].mean(1), [2.5, 97.5])],\n',
    '               ratio=ratio, ratio_ci=[float(x) for x in np.nanpercentile(bo_ratio, [2.5, 97.5])] if n > 1 else [ratio, ratio],\n',
    '               wins=int((d > 1e-9).sum()), ties=int((np.abs(d) <= 1e-9).sum()), losses=int((d < -1e-9).sum()),\n',
    "               by_set={s: int(sum(r['set'] == s for r in R)) for s in sorted({r['set'] for r in R})})\n",
    '    return res\n',
    '\n',
    '\n',
    "def arm_table(rows, arm, sets=('np', 'e180', 'all')):\n",
    '    out = {}\n',
    '    for s in sets:\n',
    "        R = [r for r in rows if r['rr_' + arm] is not None and (s == 'all' or r['set'] == s)]\n",
    '        if not R:\n',
    '            out[s] = dict(n=0)\n',
    '            continue\n',
    "        rr = np.array([r['rr_' + arm] for r in R]); rn = np.array([r['rand'] for r in R])\n",
    '        out[s] = dict(n=len(R), mrr=float(rr.mean()), rand=float(rn.mean()), lift=float((rr - rn).mean()),\n',
    '                      top1=float(np.mean(rr >= 1.0 - 1e-12)))\n',
    '    return out\n',
    '\n',
    '\n',
    "def filled_delta(rows, a, b, sel=None, label=''):\n",
    '    """Mean over ALL selected molecules with a missing arm filled by the random baseline (no evidence)."""\n',
    '    R = [r for r in rows if sel is None or sel(r)]\n',
    '    n = len(R)\n',
    '    res = dict(label=label, a=a, b=b, n=n)\n',
    '    if n == 0:\n',
    '        return res\n',
    "    fa = np.array([r['rr_' + a] if r['rr_' + a] is not None else r['rand'] for r in R])\n",
    "    fb = np.array([r['rr_' + b] if r['rr_' + b] is not None else r['rand'] for r in R])\n",
    '    d = fb - fa\n',
    '    idx = _boot_idx(n)\n',
    '    res.update(mrr_a=float(fa.mean()), mrr_b=float(fb.mean()), delta=float(d.mean()),\n',
    '               delta_ci=[float(x) for x in np.percentile(d[idx].mean(1), [2.5, 97.5])],\n',
    '               n_changed=int((np.abs(d) > 1e-9).sum()))\n',
    '    return res\n',
    '\n',
    '\n',
    'def fmt_p(p):\n',
    "    if not p.get('n'):\n",
    '        return f"{p.get(\'label\', \'\')}: {p[\'a\']} vs {p[\'b\']} n=0"\n',
    '    s = (f"{p.get(\'label\', \'\')}: n={p[\'n\']} {p[\'a\']} {p[\'mrr_a\']:.4f} vs {p[\'b\']} {p[\'mrr_b\']:.4f} (rand {p[\'rand\']:.4f}) "\n',
    '         f"delta {p[\'delta\']:+.4f} CI [{p[\'delta_ci\'][0]:+.4f}, {p[\'delta_ci\'][1]:+.4f}] "\n',
    '         f"lift {p[\'lift_a\']:.4f} -> {p[\'lift_b\']:.4f} ratio {p[\'ratio\']:.3f} CI [{p[\'ratio_ci\'][0]:.3f}, {p[\'ratio_ci\'][1]:.3f}] "\n',
    '         f"W/T/L {p[\'wins\']}/{p[\'ties\']}/{p[\'losses\']} sets {p[\'by_set\']}")\n',
    '    return s\n',
    '\n',
    '\n',
    'def gates(rows):\n',
    '    """Roadmap 3.2 gates. N1-NH4 / N1-WL: kill if the proxy lift < 50% of the paired [M+H]+ lift (same molecules).\n',
    '    I14b: adopt only if delta >= +0.01 on merged-CE e180 molecules and >= 0 on np merged-CE molecules."""\n',
    '    g = OrderedDict()\n',
    "    for key, arm in (('N1_NH4', 'B_NH4'), ('N1_WL', 'C_WL')):\n",
    "        p = paired(rows, 'A_H', arm, label=f'{key} paired [M+H]+ vs proxy')\n",
    "        pc = paired(rows, 'A_Hcut', arm, label=f'{key} paired [M+H]+cut vs proxy (diag)')\n",
    "        if not p.get('n'):\n",
    "            v = 'NO DATA'\n",
    "        elif not math.isfinite(p['ratio']):\n",
    "            v = 'UNDEFINED ([M+H]+ lift <= 0)'\n",
    '        else:\n',
    "            v = 'PASS' if p['ratio'] >= 0.5 else 'KILL'\n",
    "        g[key] = dict(verdict=v, rule='proxy lift >= 0.5 x paired [M+H]+ lift', paired=p, paired_cut=pc)\n",
    "    pe = paired(rows, 'A_H', 'D_Hmerged', sel=lambda r: r['set'] == 'e180' and r['has_merged_H'], label='I14b e180 merged-CE')\n",
    "    pn = paired(rows, 'A_H', 'D_Hmerged', sel=lambda r: r['set'] == 'np' and r['has_merged_H'], label='I14b np merged-CE')\n",
    "    if not pe.get('n') or not pn.get('n'):\n",
    "        v = 'NO DATA'\n",
    '    else:\n',
    "        v = 'ADOPT' if (pe['delta'] >= 0.01 and pn['delta'] >= 0.0) else 'REJECT'\n",
    "    g['I14b'] = dict(verdict=v, rule='delta >= +0.01 on e180 merged-CE and >= 0 on np merged-CE', e180=pe, np=pn,\n",
    "                     np_all=paired(rows, 'A_H', 'D_Hmerged', sel=lambda r: r['set'] == 'np', label='I14b np all'),\n",
    "                     e180_all=paired(rows, 'A_H', 'D_Hmerged', sel=lambda r: r['set'] == 'e180', label='I14b e180 all'))\n",
    '    return g\n',
    '\n',
    '\n',
    'def blind_report(rows):\n',
    '    """Molecules with NO real [M+H]+ spectrum (GLACIER-blind in B1): what N1 arm A recovers there."""\n',
    "    B = [r for r in rows if not r['has_H']]\n",
    "    res = OrderedDict(n_blind=len(B), n_blind_np=sum(r['set'] == 'np' for r in B),\n",
    "                      n_blind_with_proxy=sum(r['has_NH4p'] or r['has_WLp'] for r in B),\n",
    "                      n_blind_proxy_covered=sum(r['rr_N1A'] is not None for r in B))\n",
    "    C = [r for r in B if r['rr_N1A'] is not None]\n",
    '    if C:\n',
    "        rr = np.array([r['rr_N1A'] for r in C]); rn = np.array([r['rand'] for r in C])\n",
    '        idx = _boot_idx(len(C))\n',
    '        res.update(mrr_proxy=float(rr.mean()), rand=float(rn.mean()), lift=float((rr - rn).mean()),\n',
    '                   lift_ci=[float(x) for x in np.percentile((rr - rn)[idx].mean(1), [2.5, 97.5])],\n',
    '                   top1=float(np.mean(rr >= 1.0 - 1e-12)),\n',
    "                   by_src={k: int(sum(1 for r in C if r[k])) for k in ('has_NH4p', 'has_WLp')},\n",
    "                   mids=[r['mid'] for r in C])\n",
    "    res['overall_np'] = filled_delta(rows, 'A_H', 'N1A', sel=lambda r: r['set'] == 'np', label='N1A vs A, all np (blind filled with rand)')\n",
    "    res['overall_all'] = filled_delta(rows, 'A_H', 'N1A', label='N1A vs A, all molecules')\n",
    "    res['overall_np_D'] = filled_delta(rows, 'A_H', 'N1A_D', sel=lambda r: r['set'] == 'np', label='N1A+I14b vs A, all np')\n",
    '    return res\n',
    '\n',
    '\n',
    'def score_corr(new, old):\n',
    '    """new / old: {mid: {smiles: score|None}} -> pooled Pearson + mean per-molecule Spearman over common scored."""\n',
    '    xs, ys, sp = [], [], []\n',
    '    for mid, d in new.items():\n',
    '        o = old.get(mid) or {}\n',
    '        pairs = [(v, o.get(s)) for s, v in d.items() if v is not None and o.get(s) is not None]\n',
    '        if len(pairs) < 3:\n',
    '            continue\n',
    '        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])\n',
    '        xs.extend(a); ys.extend(b)\n',
    '        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))\n',
    '        if ra.std() > 0 and rb.std() > 0:\n',
    '            sp.append(float(np.corrcoef(ra, rb)[0, 1]))\n',
    '    if len(xs) < 3:\n',
    '        return dict(n_mols=0)\n',
    '    return dict(n_mols=len(sp), n_pairs=len(xs), pearson=float(np.corrcoef(xs, ys)[0, 1]),\n',
    "                spearman_mean=float(np.mean(sp)) if sp else float('nan'))\n",
    '\n',
    '\n',
    '# -------------------------------------------------------------------------------------------------- self-test ----\n',
    'def selftest():\n',
    '    """Forced unit tests of the pure functions (run on Kaggle before any GPU work and locally by the build)."""\n',
    '    M = 300.0\n',
    '    nh4 = dict(mz=[100.0, 250.0, 299.5, 300.0, 301.0, 318.03], it=[0.1, 1.0, 0.2, 0.3, 0.4, 0.5], prec=M + NH4_ADD,\n',
    "               adduct='[M+NH4]+', mode='positive', instrument='timsTOF', ce_ev=[20.0], ce_orig='20', ce_units='eV')\n",
    '    p = n1_proxy(nh4)\n',
    "    assert p is not None and p['mz'] == [100.0, 250.0, 299.5] and p['it'] == [0.1, 1.0, 0.2], p\n",
    "    assert abs(p['prec'] - (M + PROTON)) < 1e-9 and p['adduct'] == '[M+H]+' and p['n1_src'] == '[M+NH4]+'\n",
    "    assert p['ce_ev'] == [20.0] and p['instrument'] == 'timsTOF' and nh4['adduct'] == '[M+NH4]+'   # input untouched\n",
    "    wl = dict(nh4, mz=[100.0, 281.4, 281.5, 283.0], it=[1.0, 0.5, 0.5, 0.9], prec=M + PROTON - H2O, adduct='[M-H2O+H]+')\n",
    '    p = n1_proxy(wl)\n',
    "    assert p is not None and p['mz'] == [100.0, 281.4] and abs(p['prec'] - (M + PROTON)) < 1e-9, p\n",
    "    for ad in ('[M+Na]+', '[M+K]+', '[M-2H2O+H]+', '[M+H]+', '[M-H]-'):\n",
    '        assert n1_proxy(dict(nh4, adduct=ad)) is None, ad\n',
    "    assert n1_proxy(dict(nh4, mode='negative')) is None\n",
    '    assert n1_proxy(dict(nh4, mz=[310.0, 315.0], it=[1.0, 1.0])) is None             # nothing below the cut\n',
    '    assert n1_proxy(dict(nh4, mz=[100.0], it=[0.0])) is None                           # no positive intensity\n',
    "    h = dict(nh4, adduct='[M+H]+', prec=M + PROTON, mz=[100.0, 299.5, 299.6, 301.0], it=[1.0, 1.0, 1.0, 1.0])\n",
    '    c = h_cut(h)\n',
    "    assert c is not None and c['mz'] == [100.0, 299.5] and c['prec'] == h['prec'] and c['adduct'] == '[M+H]+', c\n",
    '    assert h_cut(nh4) is None\n',
    "    items = [dict(mid='np:X', cands=['A', 'B'], spectra=[h, nh4, wl, dict(h, mode='negative', adduct='[M-H]-'),\n",
    "                                                        dict(h, adduct='[M+Na]+'), dict(h, adduct='[M+K]+')])]\n",
    '    v, st = build_variants(items)\n',
    "    assert [s['sid'] for s in v[0]['spectra']] == ['0:H', '0:Hcut', '1:NH4p', '2:WLp', '4:Na'], v\n",
    '    assert st == dict(H=1, Hcut=1, NH4p=1, WLp=1, Na=1), st\n',
    '    # random tie-break expected RR\n',
    '    assert abs(rr_random_tb([0.5, 0.9, 0.5, 0.1]) - (1 / 2 + 1 / 3) / 2) < 1e-12\n',
    '    assert rr_random_tb([None, 0.2, 0.1]) is None and rr_random_tb([0.2, None, 0.1]) == 1.0\n',
    '    assert abs(rr_random_tb([0.0, 0.0, 0.0, 0.0]) - rr_rand(4)) < 1e-12\n',
    '    assert rr_truth_first([0.0, 0.0, 0.0, 0.0]) == 1.0 and abs(rr_split([0.0, 0.0, 0.0]) - 0.5) < 1e-12\n',
    '    assert abs(rr_rand(3) - (1 + 0.5 + 1 / 3) / 3) < 1e-12\n',
    '    assert covered([0.1, None, 0.2]) and not covered([0.1, None]) and not covered([None, 0.1, 0.2])\n',
    "    # aggregation: mean over the arm's spectra, None-aware; N1A tag resolution\n",
    "    mo = {'cands': ['A', 'B', 'C'], 'spec': {\n",
    "        '0:H': dict(tag='H', nb=3, mean=[0.2, 0.4, None], merged=[0.3, 0.1, None]),\n",
    "        '1:H': dict(tag='H', nb=1, mean=[0.4, None, None], merged=[0.4, None, None]),\n",
    "        '2:NH4p': dict(tag='NH4p', nb=1, mean=[0.9, 0.1, 0.5], merged=[0.9, 0.1, 0.5])}}\n",
    "    am = arm_scores(mo, ('H',), 'mean')\n",
    '    assert len(am) == 3 and np.allclose(am[:2], [0.3, 0.4]) and am[2] is None, am\n',
    "    assert np.allclose(arm_scores(mo, ('H',), 'merged')[:2], [0.35, 0.1])\n",
    "    assert arm_tags('N1A', dict(has_H=True)) == (('H',), 'mean')\n",
    "    assert arm_tags('N1A', dict(has_H=False)) == (('NH4p', 'WLp'), 'mean')\n",
    "    assert arm_tags('N1A_D', dict(has_H=False)) == (('NH4p', 'WLp'), 'merged')\n",
    '    # end-to-end on two synthetic molecules\n',
    "    its = [dict(mid='np:X', cands=['A', 'B', 'C'], spectra=[dict(tag='H'), dict(tag='NH4p')]),\n",
    "           dict(mid='np:Y', cands=['A', 'B', 'C'], spectra=[dict(tag='NH4p')]),\n",
    "           dict(mid='e180:Z', cands=['A', 'B'], spectra=[dict(tag='H')])]\n",
    "    out = {'np:X': mo, 'np:Y': {'cands': ['A', 'B', 'C'], 'spec': {'0:NH4p': dict(tag='NH4p', nb=1, mean=[0.1, 0.5, 0.0], merged=[0.1, 0.5, 0.0])}}}\n",
    "    rows = molecule_table(its, out, done={'np:X', 'np:Y'})\n",
    "    assert [r['mid'] for r in rows] == ['np:X', 'np:Y']\n",
    '    x, y = rows\n',
    "    assert abs(x['rr_A_H'] - 0.5) < 1e-12 and x['rr_B_NH4'] == 1.0 and x['rr_N1A'] == x['rr_A_H'] and x['has_merged_H']\n",
    "    assert y['rr_A_H'] is None and abs(y['rr_N1A'] - 0.5) < 1e-12 and not y['has_H']\n",
    "    p = paired(rows, 'A_H', 'B_NH4')\n",
    "    assert p['n'] == 1 and abs(p['delta'] - 0.5) < 1e-12\n",
    "    f = filled_delta(rows, 'A_H', 'N1A')\n",
    "    assert f['n'] == 2 and abs(f['delta'] - (0.5 - rr_rand(3)) / 2) < 1e-12\n",
    '    g = gates(rows)                                          # A lift < 0 here (0.5 vs rand 0.611)\n',
    "    assert g['N1_NH4']['verdict'].startswith('UNDEFINED') and g['N1_WL']['verdict'] == 'NO DATA', g['N1_NH4']['verdict']\n",
    "    assert g['I14b']['verdict'] == 'NO DATA'\n",
    '\n',
    '    def row(s, a, b, d, mh=True):\n',
    "        return dict(mid=f'{s}:{a}{b}{d}', set=s, rand=0.2, rr_A_H=a, rr_B_NH4=b, rr_A_Hcut=None, rr_C_WL=None,\n",
    '                    rr_D_Hmerged=d, has_merged_H=mh)\n',
    "    gk = gates([row('np', 1.0, 0.5, 1.0), row('np', 0.6, 0.4, 0.6), row('e180', 1.0, None, 1.0)])   # ratio 0.25/0.6\n",
    "    assert gk['N1_NH4']['verdict'] == 'KILL' and abs(gk['N1_NH4']['paired']['ratio'] - 0.25 / 0.6) < 1e-12\n",
    "    assert gk['I14b']['verdict'] == 'REJECT' and gk['I14b']['e180']['delta'] == 0.0      # +0.01 needed on e180\n",
    "    gp = gates([row('np', 1.0, 0.9, 1.0), row('np', 0.6, 0.6, 0.5), row('e180', 0.5, None, 1.0)])   # ratio 1.1/1.2\n",
    "    assert gp['N1_NH4']['verdict'] == 'PASS' and gp['I14b']['verdict'] == 'REJECT'          # np merged delta < 0\n",
    "    ga = gates([row('np', 0.5, 0.5, 0.5), row('e180', 0.5, None, 0.52)])\n",
    "    assert ga['I14b']['verdict'] == 'ADOPT' and abs(ga['I14b']['e180']['delta'] - 0.02) < 1e-12\n",
    '    b = blind_report(rows)\n',
    "    assert b['n_blind'] == 1 and b['n_blind_proxy_covered'] == 1\n",
    "    sc = score_corr({'m': {'a': 1.0, 'b': 2.0, 'c': 3.0}}, {'m': {'a': 2.0, 'b': 4.0, 'c': 6.5}})\n",
    "    assert sc['n_mols'] == 1 and sc['pearson'] > 0.99 and abs(sc['spearman_mean'] - 1.0) < 1e-12\n",
    "    return 'glb_core selftest ok (proxy NH4/WL/Hcut, variants, random tie-break RR, aggregation, paired, gates, blind)'\n",
])
# kdev_glbench_n1 glb_driver.py, verbatim (GLACIER driver: mean + merged-CE per spectrum, own process)
N1_DRIVER_SRC = ''.join([
    '#!/usr/bin/env python\n',
    '"""casmi26 kdev_glbench_n1 -- GLACIER bench driver (runs in its OWN process, like gl_runner.py).\n',
    '\n',
    'Imports the attached casmi26-glacier gl_runner.py as a module and reuses ITS environment setup (offline wheel site,\n',
    'shims, RDKit 2025.03 check), model loading, batched Predictor (canonical-SMILES dedupe, OOM split, fail-fast), query\n',
    "cleaning, CE bucketing, summed-CE merge() and entropy similarity. The 'mean' mode calls gl_runner.score_molecule on\n",
    "one spectrum at a time, so a molecule's mean over its [M+H]+ spectra is exactly the production gl_ex_mean.\n",
    "Added: per-spectrum output, and the I14b 'merged' mode (a merged-CE eV spectrum = >= 2 distinct CE buckets is\n",
    'predicted at each bucket and the predictions are merge()d like the SUM_CE path; single-CE / NCE / no-CE spectra:\n',
    'merged = mean).\n',
    '\n',
    'usage: python glb_driver.py <gl_pkg> <in.json> <out.json> [--device cuda] [--budget S] [--site DIR] [--wheels DIR]\n',
    '                            [--bs 32] [--chunk-jobs 3000] [--fake]\n',
    'in.json : [{"mid", "cands": [smiles], "spectra": [{"sid", "tag", "mz", "it", "prec", "adduct", "mode", "instrument",\n',
    '            "ce_ev", "ce_orig", "ce_units"}]}]\n',
    'out.json: {mid: {"cands": [smiles], "spec": {sid: {"tag", "nb", "mean": [sim|null], "merged": [sim|null]}}}}\n',
    '          (lists aligned with "cands"; null = rejected SMILES / failed prediction)\n',
    'out.json.meta.json: status, counts, timings, versions, errors, "done" = molecules fully scored (input order).\n',
    '--fake: no torch / model / wheel site (local dry run): deterministic pseudo-random predictions, SMILES accepted as is.\n',
    '"""\n',
    'import os, sys, json, time, math, argparse, traceback, hashlib, importlib.util\n',
    '\n',
    'sys.dont_write_bytecode = True\n',
    "NEED = ('setup_env', 'load_model', 'Predictor', 'smiles_ok', '_CANON', 'molecule_jobs', 'score_molecule',\n",
    "        'prep_spectrum', 'prep_query', 'entropy_sim', 'merge', 'post', 'ce_bucket', 'spectrum_ce', 'is_covered',\n",
    "        'instr_token', 'SUM_CE', 'FLOOR', 'TOPK', 'POWER', 'ENTW', 'TOL', 'PPM', 'write_json', 'Log')\n",
    '\n',
    '\n',
    'def sha256(path):\n',
    '    h = hashlib.sha256()\n',
    "    with open(path, 'rb') as f:\n",
    "        for b in iter(lambda: f.read(1 << 20), b''):\n",
    '            h.update(b)\n',
    '    return h.hexdigest()\n',
    '\n',
    '\n',
    'def load_gl(pkg):\n',
    '    """gl_runner.py of the package as a module (no side effects beyond its imports: numpy, numba)."""\n',
    "    spec = importlib.util.spec_from_file_location('gl_runner', os.path.join(pkg, 'gl_runner.py'))\n",
    '    G = importlib.util.module_from_spec(spec)\n',
    "    sys.modules['gl_runner'] = G\n",
    '    spec.loader.exec_module(G)\n',
    '    miss = [n for n in NEED if not hasattr(G, n)]\n',
    '    if miss:\n',
    "        raise RuntimeError(f'gl_runner API differs from the one this driver was written for: missing {miss}')\n",
    '    return G\n',
    '\n',
    '\n',
    '# ------------------------------------------------------------------------------------------- per-spectrum plan ----\n',
    'def merged_buckets(G, sp):\n',
    '    """I14b: sorted distinct CE buckets of a merged-CE eV spectrum when there are >= 2, else None."""\n',
    "    if sp.get('ce_units') == 'NCE':\n",
    '        return None\n',
    '    b = set()\n',
    "    for x in sp.get('ce_ev') or []:\n",
    '        try:\n',
    '            x = float(x)\n',
    '        except (TypeError, ValueError):\n',
    '            continue\n',
    '        if math.isfinite(x):\n',
    '            cb = G.ce_bucket(x)\n',
    '            if math.isfinite(cb):\n',
    '                b.add(float(cb))\n',
    '    return sorted(b) if len(b) >= 2 else None\n',
    '\n',
    '\n',
    'def spec_plan(G, sp):\n',
    '    """None if gl_runner would not score this spectrum, else {\'entry\': molecule_plan entry, \'mb\': buckets|None}."""\n',
    '    if not G.is_covered(sp):\n',
    '        return None\n',
    '    qm, qp = G.prep_spectrum(sp)\n',
    "    return dict(entry=(qm, qp, sp['adduct'], G.instr_token(sp), G.ce_bucket(G.spectrum_ce(sp))), mb=merged_buckets(G, sp))\n",
    '\n',
    '\n',
    'def spec_jobs(G, plan, cands):\n',
    "    J = set(G.molecule_jobs([plan['entry']], cands))\n",
    "    if plan['mb']:\n",
    "        _, _, ad, ins, _ = plan['entry']\n",
    '        for c in cands:\n',
    "            for b in plan['mb']:\n",
    '                J.add((c, ad, ins, b))\n',
    '    return J\n',
    '\n',
    '\n',
    'def spec_scores(G, plan, cands, cache):\n',
    '    """-> (mean {c: sim|None}, merged {c: sim|None}, n_buckets)."""\n',
    "    mean = G.score_molecule([plan['entry']], cands, cache)\n",
    "    if not plan['mb']:\n",
    '        return mean, dict(mean), 1\n',
    "    qm, qp, ad, ins, _ = plan['entry']\n",
    '    merged = {}\n',
    '    for c in cands:\n',
    "        parts = [cache.get((c, ad, ins, b)) for b in plan['mb']]\n",
    '        if any(p is None for p in parts):            # = the SUM_CE path: every component must be predicted\n',
    '            merged[c] = None\n',
    '            continue\n',
    '        pr = G.merge(parts)\n',
    '        pm, pp = G.prep_query(pr[0], pr[1], G.FLOOR, G.TOPK, G.POWER, G.ENTW)\n',
    '        s = float(G.entropy_sim(qm, qp, pm, pp, G.TOL, G.PPM))\n',
    '        merged[c] = s if math.isfinite(s) else None\n',
    "    return mean, merged, len(plan['mb'])\n",
    '\n',
    '\n',
    '# ------------------------------------------------------------------------------------------------ fake mode ------\n',
    'class FakePredictor:\n',
    '    """Same interface as gl_runner.Predictor (predict(jobs, deadline) -> bool, .cache); deterministic random peaks."""\n',
    '\n',
    '    def __init__(self, G):\n',
    '        self.G, self.cache = G, {}\n',
    '        self.n_pred = self.n_fail = self.n_feat_fail = 0\n',
    '        self.t_pred, self.bs = 0.0, 0\n',
    '\n',
    '    def predict(self, jobs, deadline):\n',
    '        import numpy as np\n',
    '        t0 = time.time()\n',
    '        for j in sorted(jobs):\n',
    '            if j in self.cache:\n',
    '                continue\n',
    '            if time.time() > deadline:\n',
    '                self.t_pred += time.time() - t0\n',
    '                return False\n',
    '            c = self.G._CANON.get(j[0], (None,))[0]\n',
    '            if c is None:\n',
    '                self.cache[j] = None\n',
    '                continue\n',
    '            r = np.random.default_rng(int(hashlib.md5(repr((c, j[1], j[2], float(j[3]))).encode()).hexdigest()[:8], 16))\n',
    '            mz = np.round(r.uniform(40.0, 400.0, 25), 2)\n',
    '            o = self.G.post(np.stack([mz, r.uniform(0.05, 1.0, 25)], 1).astype(np.float32))\n',
    '            self.cache[j] = (o[0].astype(np.float64), o[1].astype(np.float64))\n',
    '            self.n_pred += 1\n',
    '        self.t_pred += time.time() - t0\n',
    '        return True\n',
    '\n',
    '\n',
    'def fake_smiles_ok(G):\n',
    '    def ok(smi):\n',
    '        good = isinstance(smi, str) and bool(smi)\n',
    "        G._CANON[smi] = (smi if good else None, max(1, len(smi) // 2) if good else 0, '' if good else 'empty')\n",
    "        return good, ('' if good else 'empty')\n",
    '    return ok\n',
    '\n',
    '\n',
    '# ------------------------------------------------------------------------------------------------- self-test -----\n',
    'def selftest(G):\n',
    '    """Forced unit test of the merged-CE logic on a synthetic cache (no model); run by the notebook before the GPU\n',
    '    job and by the local build."""\n',
    '    import numpy as np\n',
    "    sp = dict(mz=[50.0, 80.0, 120.0, 150.5], it=[0.2, 1.0, 0.5, 0.3], prec=151.0, adduct='[M+H]+', mode='positive',\n",
    "              instrument='timsTOF', ce_ev=[20.0, 40.0, 60.0], ce_orig='20,40,60', ce_units='eV')\n",
    '    p = spec_plan(G, sp)\n',
    "    assert p is not None and p['mb'] == [20.0, 40.0, 60.0], p\n",
    "    assert p['entry'][2:] == ('[M+H]+', 'QTOF', 40.0), p['entry'][2:]\n",
    "    cands = ['CCO', 'CCC']\n",
    '    J = spec_jobs(G, p, cands)\n',
    "    assert J == {(c, '[M+H]+', 'QTOF', b) for c in cands for b in (20.0, 40.0, 60.0)}, J\n",
    '    rng = np.random.default_rng(0)\n',
    '    cache = {}\n',
    '    for j in sorted(J):\n',
    '        mz = np.sort(rng.choice([50.0, 80.0, 120.0, 150.5, 60.0, 99.0], 4, replace=False))\n',
    '        cache[j] = (mz.astype(np.float64), rng.uniform(0.1, 1.0, 4).astype(np.float64))\n',
    "    cache[('CCC', '[M+H]+', 'QTOF', 60.0)] = None\n",
    '    mean, merged, nb = spec_scores(G, p, cands, cache)\n',
    "    assert nb == 3 and merged['CCC'] is None and mean['CCC'] is not None and mean['CCO'] is not None, (mean, merged)\n",
    "    pr = G.merge([cache[('CCO', '[M+H]+', 'QTOF', b)] for b in (20.0, 40.0, 60.0)])\n",
    '    qm, qp = G.prep_spectrum(sp)\n',
    '    pm, pp = G.prep_query(pr[0], pr[1], G.FLOOR, G.TOPK, G.POWER, G.ENTW)\n',
    "    assert abs(merged['CCO'] - float(G.entropy_sim(qm, qp, pm, pp, G.TOL, G.PPM))) < 1e-12\n",
    "    pm, pp = G.prep_query(*cache[('CCO', '[M+H]+', 'QTOF', 40.0)], G.FLOOR, G.TOPK, G.POWER, G.ENTW)\n",
    "    assert abs(mean['CCO'] - float(G.entropy_sim(qm, qp, pm, pp, G.TOL, G.PPM))) < 1e-12\n",
    '    p1 = spec_plan(G, dict(sp, ce_ev=[40.0]))                         # single CE: merged == mean\n',
    '    m1, g1, n1 = spec_scores(G, p1, cands, cache)\n',
    "    assert p1['mb'] is None and n1 == 1 and m1 == g1 and abs(m1['CCO'] - mean['CCO']) < 1e-12\n",
    '    assert merged_buckets(G, dict(sp, ce_ev=[20.0, 21.0])) is None    # same bucket twice\n',
    '    assert merged_buckets(G, dict(sp, ce_ev=[0.0, 20.0])) == [5.0, 20.0]\n',
    "    assert merged_buckets(G, dict(sp, ce_units='NCE', ce_orig='20,40')) is None\n",
    '    assert merged_buckets(G, dict(sp, ce_ev=[])) is None\n',
    "    assert spec_plan(G, dict(sp, adduct='[M+NH4]+')) is None and spec_plan(G, dict(sp, mode='negative')) is None\n",
    '    pn = spec_plan(G, dict(sp, ce_ev=[]))                              # no CE: SUM_CE jobs, merged == mean\n',
    "    assert pn['mb'] is None and spec_jobs(G, pn, ['CCO']) == {('CCO', '[M+H]+', 'QTOF', c) for c in G.SUM_CE}\n",
    "    return 'glb_driver selftest ok (merged-CE buckets, jobs, merge == SUM_CE path, single-CE identity, coverage)'\n",
    '\n',
    '\n',
    '# ------------------------------------------------------------------------------------------------------ main -----\n',
    'def parse(argv=None):\n',
    '    ap = argparse.ArgumentParser()\n',
    "    ap.add_argument('pkg'); ap.add_argument('inp'); ap.add_argument('out')\n",
    "    ap.add_argument('--device', default='cuda')\n",
    "    ap.add_argument('--budget', type=float, default=0.0)\n",
    "    ap.add_argument('--site', default='/tmp/glb_site')\n",
    "    ap.add_argument('--wheels', default=None)\n",
    "    ap.add_argument('--bs', type=int, default=32)\n",
    "    ap.add_argument('--chunk-jobs', type=int, default=3000)\n",
    "    ap.add_argument('--fake', action='store_true')\n",
    '    return ap.parse_args(argv)\n',
    '\n',
    '\n',
    'def main(argv=None):\n',
    '    a = parse(argv)\n',
    '    if not a.fake:\n',
    "        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # = gl_runner.main (deterministic torch)\n",
    '    t_start = time.time()\n',
    "    deadline = t_start + a.budget if a.budget > 0 else float('inf')\n",
    "    meta = dict(model='glacier-bench', status='init', fake=a.fake, n_mols=0, n_mols_done=0, n_spec=0, n_spec_covered=0,\n",
    '                n_spec_merged=0, n_jobs=0, errors=[], done=[])\n',
    '    G = None\n',
    '    result = {}\n',
    '\n',
    '    def write(status=None):\n',
    '        if status:\n',
    "            meta['status'] = status\n",
    "        meta['seconds'] = round(time.time() - t_start, 2)\n",
    "        tmp = a.out + '.tmp'\n",
    "        with open(tmp, 'w') as f:\n",
    '            json.dump(result, f, allow_nan=False)\n',
    '        os.replace(tmp, a.out)\n',
    "        with open(a.out + '.meta.json.tmp', 'w') as f:          # atomic too: a kill mid-write never leaves bad JSON\n",
    '            json.dump(meta, f, allow_nan=False)\n',
    "        os.replace(a.out + '.meta.json.tmp', a.out + '.meta.json')\n",
    '\n',
    '    def log(*x):\n',
    "        print(f'[glb {time.time() - t_start:7.1f}s]', *x, file=sys.stderr, flush=True)\n",
    '\n',
    '    try:\n',
    '        mols = json.load(open(a.inp))\n',
    "        meta['n_mols'] = len(mols)\n",
    '        write()\n',
    '        G = load_gl(a.pkg)\n',
    "        meta['gl_runner_sha256'] = sha256(os.path.join(a.pkg, 'gl_runner.py'))\n",
    '        if a.fake:\n',
    "            meta['rdkit'] = 'fake'\n",
    '            G.smiles_ok = fake_smiles_ok(G)\n',
    '            pred = FakePredictor(G)\n',
    "            log('FAKE mode: pseudo-random predictions, no model')\n",
    '        else:\n',
    '            ns = argparse.Namespace(pkg=a.pkg, site=a.site, wheels=a.wheels, any_rdkit=False)\n',
    "            meta['rdkit'] = G.setup_env(ns, log)\n",
    '            import torch\n',
    "            meta['torch'] = torch.__version__\n",
    '            dev = a.device\n',
    "            if dev.startswith('cuda') and not torch.cuda.is_available():\n",
    "                log('cuda unavailable -> cpu'); dev = 'cpu'\n",
    "            meta['device'] = dev\n",
    '            torch.use_deterministic_algorithms(True, warn_only=True)\n',
    '            torch.backends.cudnn.benchmark = False\n',
    '            model = G.load_model(os.path.abspath(a.pkg), dev, log)\n',
    '            pred = G.Predictor(model, dev, a.bs, log)\n',
    '            log(f\'model loaded on {dev} (rdkit {meta["rdkit"]}, torch {torch.__version__}, numba {G.HAVE_NUMBA})\')\n',
    "        meta['t_load'] = round(time.time() - t_start, 2)\n",
    '        okc = {}\n',
    '        chunk, chunk_jobs = [], set()\n',
    '\n',
    '        def flush():\n',
    '            nonlocal chunk, chunk_jobs\n',
    '            if not chunk:\n',
    '                return True\n',
    '            t0 = time.time()\n',
    '            if not pred.predict(chunk_jobs, deadline):\n',
    '                return False\n',
    '            for mid, cands, plans in chunk:\n',
    '                spec = {}\n',
    '                for sid, tag, pl in plans:\n',
    '                    try:\n',
    '                        mean, merged, nb = spec_scores(G, pl, cands, pred.cache)\n',
    '                    except Exception as e:\n',
    "                        meta['errors'].append(f'{mid} {sid}: score {repr(e)[:200]}')\n",
    '                        continue\n',
    "                    spec[sid] = dict(tag=tag, nb=nb, mean=[mean.get(c) for c in result[mid]['cands']],\n",
    "                                     merged=[merged.get(c) for c in result[mid]['cands']])\n",
    "                result[mid]['spec'] = spec\n",
    "                meta['done'].append(mid)\n",
    "            meta['n_mols_done'] = len(meta['done'])\n",
    '            write()\n',
    "            log(f'chunk: {len(chunk)} mols {len(chunk_jobs)} jobs {time.time() - t0:.1f}s '\n",
    '                f\'({meta["n_mols_done"]}/{meta["n_mols"]} mols done, {pred.n_pred} predictions)\')\n',
    '            chunk, chunk_jobs = [], set()\n',
    '            return True\n',
    '\n',
    '        stopped = False\n',
    '        for mol in mols:\n',
    "            mid = str(mol['mid'])\n",
    "            allc = list(dict.fromkeys(mol.get('cands') or []))\n",
    '            result[mid] = dict(cands=allc, spec={})\n',
    '            for c in allc:\n',
    '                if c not in okc:\n',
    '                    okc[c], why = G.smiles_ok(c)\n',
    '                    if not okc[c] and sum(1 for v in okc.values() if not v) <= 5:\n',
    "                        meta['errors'].append(f'smiles rejected ({why}): {str(c)[:100]}')\n",
    '            cands = [c for c in allc if okc[c]]\n',
    '            plans = []\n',
    "            for sp in mol.get('spectra') or []:\n",
    "                meta['n_spec'] += 1\n",
    '                try:\n',
    '                    pl = spec_plan(G, sp)\n',
    '                except Exception as e:\n',
    '                    meta[\'errors\'].append(f\'{mid} {sp.get("sid")}: plan {repr(e)[:200]}\')\n',
    '                    continue\n',
    '                if pl is None:\n',
    '                    continue\n',
    "                meta['n_spec_covered'] += 1\n",
    "                meta['n_spec_merged'] += int(bool(pl['mb']))\n",
    "                plans.append((sp['sid'], sp.get('tag'), pl))\n",
    '            if not cands or not plans:\n',
    "                meta['done'].append(mid)                         # nothing GLACIER could score: done, empty\n",
    '                continue\n',
    '            J = set()\n',
    '            for _, _, pl in plans:\n',
    '                J |= spec_jobs(G, pl, cands)\n',
    "            meta['n_jobs'] += len(J)\n",
    '            chunk.append((mid, cands, plans))\n',
    '            chunk_jobs |= J\n',
    '            if len(chunk_jobs) >= a.chunk_jobs and not flush():\n',
    '                stopped = True\n',
    '                break\n',
    '        if not stopped and not flush():\n',
    '            stopped = True\n',
    '        meta.update(n_pred=pred.n_pred, n_pred_fail=pred.n_fail, n_feat_fail=pred.n_feat_fail, t_pred=round(pred.t_pred, 2),\n',
    '                    pred_per_s=round(pred.n_pred / max(pred.t_pred, 1e-9), 2), bs_final=pred.bs,\n',
    "                    n_bad_smiles=sum(1 for v in okc.values() if not v), n_mols_done=len(meta['done']))\n",
    "        if not a.fake and meta.get('device', '').startswith('cuda'):\n",
    '            import torch\n',
    "            meta['max_mem_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)\n",
    "        write('budget' if stopped else 'ok')\n",
    "        log('status', meta['status'], {k: meta[k] for k in ('n_mols', 'n_mols_done', 'n_spec_covered', 'n_spec_merged',\n",
    "                                                            'n_pred', 'n_pred_fail', 'pred_per_s')})\n",
    '        return 0\n',
    '    except BaseException as e:                                   # never crash the notebook: status error + partial out\n',
    "        meta['errors'].append(traceback.format_exc()[-3000:])\n",
    "        log('ERROR', repr(e)[:300])\n",
    '        try:\n',
    "            write('error')\n",
    '        except Exception:\n',
    '            return 2\n',
    '        return 0\n',
    '\n',
    '\n',
    "if __name__ == '__main__':\n",
    '    sys.exit(main())\n',
])


def _n1_mod(name, path):
    import importlib.util as _iu
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def n1_select(items_all, K):
    """N1 arm A (bench glb_core) on fuse.build_ice_input(..., covered_only=False) items -> (driver items, counts):
    the molecules WITHOUT a real positive [M+H]+ spectrum that have >= 1 [M+NH4]+ / [M-H2O+H]+ proxy, carrying only
    those proxies (bench tags NH4p / WLp, sid '<spectrum index>:<tag>'), in input (= GLACIER priority) order."""
    var, _ = K.build_variants(items_all)
    raw = {str(it['mid']): sum(1 for s in it.get('spectra') or [] if s.get('mode') == 'positive'
                               and s.get('adduct') in K.PROXY_TAG) for it in items_all}
    out = []
    st = dict(molecules=0, with_H=0, blind=0, blind_nh4_wl=0, n1=0, proxies=0, NH4p=0, WLp=0, proxy_empty=0)
    for it in var:
        fl = K.mol_flags(it)
        st['molecules'] += 1
        if fl['has_H']:                  # arm A: a real [M+H]+ exists -> the H1 run above scores this molecule
            st['with_H'] += 1
            continue
        st['blind'] += 1
        tags, _mode = K.arm_tags(N1_ARM, fl)
        sps = [s for s in it['spectra'] if s.get('tag') in tags]
        st['blind_nh4_wl'] += int(raw.get(it['mid'], 0) > 0)
        st['proxy_empty'] += raw.get(it['mid'], 0) - len(sps)
        if not sps or not it['cands']:
            continue
        st['n1'] += 1
        st['proxies'] += len(sps)
        for s in sps:
            st[s['tag']] += 1
        out.append(dict(mid=it['mid'], cands=list(it['cands']), spectra=sps))
    return out, st


def n1_run(items, budget):
    """The bench glb_driver.py in its own process (= gl_fuse.run_gl pattern) -> (out, meta, done). Never raises for a
    driver failure: the driver checkpoints after every chunk, so a budget stop / crash / kill keeps the molecules it
    finished; unreadable output -> nothing done."""
    fin, fout = os.path.join(N1_WORK, 'n1_in.json'), os.path.join(N1_WORK, 'n1_out.json')
    for f in (fout, fout + '.meta.json'):
        if os.path.exists(f):
            os.remove(f)
    with open(fin, 'w') as f:
        json.dump(items, f)
    cmd = [sys.executable, os.path.join(N1_WORK, 'glb_driver.py'), GL_PKG, fin, fout, '--device', N1_DEVICE,
           '--budget', str(float(budget)), '--site', N1_SITE, '--wheels', os.path.join(ICE_PKG, 'wheels')] + list(N1_DRIVER_EXTRA)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace',
                           timeout=float(budget) + 600)
        tail = (r.stdout or '')[-3000:]
    except subprocess.TimeoutExpired as e:
        tail = f'TIMEOUT after {time.time() - t0:.0f}s ' + (e.stdout.decode(errors='replace') if isinstance(e.stdout, bytes)
                                                            else str(e.stdout or ''))[-2000:]
    print(tail, flush=True)
    meta, out = {}, {}
    try:
        with open(fout + '.meta.json') as f:
            meta = json.load(f)
    except Exception as e:
        print('[N1] driver meta unreadable:', repr(e), flush=True)
    try:
        with open(fout) as f:
            out = json.load(f)
    except Exception as e:
        print('[N1] driver output unreadable:', repr(e), flush=True)
    done = set(meta.get('done') or []) if meta else {m for m, v in out.items() if isinstance(v, dict) and v.get('spec')}
    return out, meta, done


def n1_scores(out, done, items, K, mode=None):
    """Driver output -> {mid: {smiles: score | None}} = bench arm_scores of N1_ARM (mean over the molecule's proxy
    spectra of the per-spectrum similarity in the arm's CE mode, or `mode`); finished molecules only, candidate order
    as sent; non-finite / outside [0, 1] -> None; molecules without any score are left out."""
    tags, amode = K.arm_tags(N1_ARM, dict(has_H=False))
    res = {}
    for it in items:
        mid = str(it['mid'])
        mo = out.get(mid)
        if mid not in done or not isinstance(mo, dict):
            continue
        cands = [str(c) for c in mo.get('cands') or []]
        if cands != [str(c) for c in dict.fromkeys(it['cands'])]:
            print('[N1] candidate order changed -> molecule dropped', mid, flush=True)
            continue
        d = {}
        for c, x in zip(cands, K.arm_scores(mo, tags, mode or amode)):
            d[c] = float(x) if x is not None and math.isfinite(x) and -1e-9 <= x <= 1.0 + 1e-9 else None
        if any(x is not None for x in d.values()):
            res[mid] = d
    return res


def n1_synthetic_te(te_df, mids, K):
    """Commit-only forced exercise input: for every molecule in `mids` with a positive [M+H]+ spectrum a copy
    'n1x:<mid>' WITHOUT any [M+H]+: each positive [M+H]+ row -> an [M+NH4]+ row (precursor + 18.033823 - 1.007276,
    plus an injected peak AT the NH4 precursor that the proxy cut must drop); the first one also -> an [M-H2O+H]+ row
    (precursor - 18.010565, peaks unchanged); original NH4 / water-loss rows are left out; every other row (negative,
    Na, K, 2M, ...) is copied unchanged and must never be proxied.
    -> (DataFrame, {xmid: [(kind, original [M+H]+ precursor, mz list, intensity list) per row]})"""
    rows, ref = [], {}
    want = {str(m) for m in mids}
    for mid, sub in te_df.groupby(te_df.molecule_id.astype(str), sort=False):
        if mid not in want:
            continue
        recs = sub.to_dict('records')
        if not any(r.get('ionization_mode') == 'positive' and r.get('adduct') == '[M+H]+' for r in recs):
            continue
        xm, first, rf = f'n1x:{mid}', True, []
        for r in recs:
            r = dict(r, molecule_id=xm)
            if r.get('ionization_mode') == 'positive' and r.get('adduct') == '[M+H]+':
                prec = float(r['precursor_mz'])
                mz = [float(x) for x in r['ms2_mzs']]
                it = [float(x) for x in r['ms2_normalized_intensities']]
                if first:
                    rows.append(dict(r, adduct='[M-H2O+H]+', precursor_mz=prec - K.H2O, ms2_mzs=np.asarray(mz),
                                     ms2_normalized_intensities=np.asarray(it)))
                    rf.append(('WL', prec, mz, it))
                    first = False
                p2 = prec - K.PROTON + K.NH4_ADD
                rows.append(dict(r, adduct='[M+NH4]+', precursor_mz=p2, ms2_mzs=np.asarray(mz + [p2]),
                                 ms2_normalized_intensities=np.asarray(it + [1.0])))
                rf.append(('NH4', prec, mz, it))
            elif r.get('ionization_mode') == 'positive' and r.get('adduct') in K.PROXY_TAG:
                continue                 # an original NH4 / water-loss row: left out, so every proxy traces to a converted row
            else:
                rows.append(r)
                rf.append(('keep', None, None, None))
        ref[xm] = rf
    return pd.DataFrame(rows), ref


def n1_check_proxies(x_items, ref, K):
    """Commit asserts on the synthetic proxies: relabel, precursor back at the original [M+H]+ m/z (< 1e-6 Da), cut =
    exactly the original peaks below the bench cut (injected NH4 precursor peak gone), one proxy per converted row."""
    n = 0
    for it in x_items:
        rf = ref[it['mid']]
        for s in it['spectra']:
            j, tag = s['sid'].split(':')
            kind, prec0, mz0, it0 = rf[int(j)]
            assert (tag, kind) in (('NH4p', 'NH4'), ('WLp', 'WL')), (it['mid'], s['sid'], kind)
            assert s['adduct'] == '[M+H]+' and s['mode'] == 'positive' and s['n1_src'] == (
                '[M+NH4]+' if kind == 'NH4' else '[M-H2O+H]+'), (it['mid'], s['sid'])
            assert abs(float(s['prec']) - prec0) < 1e-6, (it['mid'], s['sid'], s['prec'], prec0)
            if kind == 'NH4':
                p2 = prec0 - K.PROTON + K.NH4_ADD
                cut = (p2 - K.NH4_ADD) + K.PROTON - K.CUT_DA
            else:
                cut = (prec0 - K.H2O) - K.CUT_DA
            keep = [(m, i) for m, i in zip(mz0, it0) if m < cut]
            assert s['mz'] == [m for m, _ in keep] and s['it'] == [i for _, i in keep], (it['mid'], s['sid'])
            assert max(s['mz']) < float(s['prec']) - K.CUT_DA + 1e-6, (it['mid'], s['sid'])
            n += 1
        assert {s['tag'] for s in it['spectra']} <= {'NH4p', 'WLp'}, it['mid']
    return n


def _n1_spearman(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float('nan')


def n1_ice_finite(ice):
    """Number of finite ICEBERG scores in one molecule's ICE dict (None / NaN / inf / non-numeric = no evidence; the
    per-value test of fuse.rerank and of E006's _finite_coverage)."""
    import math as _m
    n = 0
    for v in (ice.values() if isinstance(ice, dict) else ()):
        try:
            n += v is not None and _m.isfinite(float(v))
        except (TypeError, ValueError):
            pass
    return n


def n1_complement(scores, ice_scores):
    """k2 variant A: drop the N1 GL scores of the molecules whose ICE dict (ice_scores[str(mid)]) holds a finite
    score -> (kept {mid: same score dict}, dropped {mid}). Kept dicts are passed through unchanged."""
    ice_scores = ice_scores if isinstance(ice_scores, dict) else {}
    drop = {m for m in scores if n1_ice_finite(ice_scores.get(str(m))) > 0}
    return {m: v for m, v in scores.items() if m not in drop}, drop


try:
    import math
    _n1_h1_gl = isinstance(GL_SCORES, dict) and any(isinstance(g, dict) and any(v is not None for v in g.values())
                                                  for g in GL_SCORES.values())
    if not _n1_h1_gl:
        raise RuntimeError('H1 GLACIER produced no scores (GL failed / not run) -> N1 not attempted')
    os.makedirs(N1_WORK, exist_ok=True)
    for _n1_fn, _n1_s, _n1_h in (('glb_core.py', N1_CORE_SRC, N1_CORE_SHA256), ('glb_driver.py', N1_DRIVER_SRC, N1_DRIVER_SHA256)):
        assert hashlib.sha256(_n1_s.encode('utf-8')).hexdigest() == _n1_h, ('embedded bench source changed', _n1_fn)
        with open(os.path.join(N1_WORK, _n1_fn), 'w', encoding='utf-8') as _n1_f:
            _n1_f.write(_n1_s)
    N1K = _n1_mod('glb_core', os.path.join(N1_WORK, 'glb_core.py'))
    N1DR = _n1_mod('glb_driver', os.path.join(N1_WORK, 'glb_driver.py'))
    print('[N1]', N1K.selftest(), flush=True)
    _n1_G = N1DR.load_gl(GL_PKG)            # attached gl_runner.py as a module (numpy / numba only; no torch here)
    print('[N1]', N1DR.selftest(_n1_G), flush=True)
    _n1_sg = sha256(os.path.join(GL_PKG, 'gl_runner.py'))
    print(f'[N1] code: glb_core {N1_CORE_SHA256[:12]} glb_driver {N1_DRIVER_SHA256[:12]} (= kdev_glbench_n1) | '
          f'gl_runner.py {_n1_sg[:12]} matches the B1/bench MANIFEST: {_n1_sg == N1_GL_RUNNER_SHA256} | arm {N1_ARM} '
          f'(tags, mode) {N1K.arm_tags(N1_ARM, dict(has_H=False))} | budget {N1_BUDGET:.0f}s', flush=True)
    _n1_items_all = ice_fuse.build_ice_input(te, cands, covered_only=False)
    _n1_items, N1_STATS = n1_select(_n1_items_all, N1K)
    print(f"[N1] molecules: GL/ICE input {N1_STATS['molecules']} | with real [M+H]+ {N1_STATS['with_H']} (H1 run, "
          f"unchanged) | blind {N1_STATS['blind']} (with NH4/WL spectra {N1_STATS['blind_nh4_wl']}) | N1 molecules "
          f"{N1_STATS['n1']} | proxies built {N1_STATS['proxies']} (NH4p {N1_STATS['NH4p']}, WLp {N1_STATS['WLp']}; "
          f"empty after the cut {N1_STATS['proxy_empty']})", flush=True)
    _n1_x_items, _n1_x_ref = [], {}
    if not IS_RERUN:                     # forced commit exercise: synthetic blind copies through the same code
        _n1_tx, _n1_x_ref = n1_synthetic_te(te, list(cands), N1K)
        _n1_x_all = ice_fuse.build_ice_input(_n1_tx, {f'n1x:{m}': c for m, c in cands.items() if f'n1x:{m}' in _n1_x_ref},
                                          covered_only=False)
        _n1_x_items, N1X_STATS = n1_select(_n1_x_all, N1K)
        N1X_STATS['proxy_checked'] = n1_check_proxies(_n1_x_items, _n1_x_ref, N1K)
        assert N1X_STATS['blind'] == N1X_STATS['molecules'] == len(_n1_x_ref) >= 1 and N1X_STATS['with_H'] == 0, N1X_STATS
        assert N1X_STATS['n1'] >= 1 and N1X_STATS['proxy_checked'] == N1X_STATS['proxies'] >= N1X_STATS['n1'], N1X_STATS
        assert N1X_STATS['proxies'] + N1X_STATS['proxy_empty'] == sum(k != 'keep' for v in _n1_x_ref.values() for k, *_ in v)
        # arm-B guard: a molecule WITH a real [M+H]+ plus NH4 / water-loss spectra must NOT become an N1 molecule
        _n1_src = sorted(_n1_x_ref)[0].split(':', 1)[1]
        _n1_gdf = pd.concat([te[te.molecule_id.astype(str) == _n1_src], _n1_tx[_n1_tx.molecule_id == f'n1x:{_n1_src}']],
                            ignore_index=True).assign(molecule_id=f'n1b:{_n1_src}')
        _n1_gi, _n1_gs = n1_select(ice_fuse.build_ice_input(_n1_gdf, {f'n1b:{_n1_src}': {str(m): c for m, c in cands.items()}[_n1_src]},
                                                           covered_only=False), N1K)
        assert _n1_gi == [] and _n1_gs['with_H'] == 1 and _n1_gs['n1'] == 0, ('arm-B guard: [M+H]+ molecule selected', _n1_gs)
        N1X_STATS['armB_guard'] = 'ok'
        N1X_MIDS = {it['mid'] for it in _n1_x_items}
    _n1_run = _n1_items + _n1_x_items
    _n1_exp_merged = sum(bool(N1DR.merged_buckets(_n1_G, s)) for it in _n1_run for s in it['spectra'])
    _n1_out, N1_DRV, _n1_done = n1_run(_n1_run, N1_BUDGET) if _n1_run else ({}, {'status': 'nothing to score'}, set())
    print(f"[N1] driver: status {N1_DRV.get('status')} | molecules sent {len(_n1_run)} (real {len(_n1_items)}, synthetic "
          f"{len(_n1_x_items)}) done {len(_n1_done)} | spectra covered {N1_DRV.get('n_spec_covered')} (merged-CE "
          f"{N1_DRV.get('n_spec_merged')}, expected {_n1_exp_merged}) | predictions {N1_DRV.get('n_pred')} (fail "
          f"{N1_DRV.get('n_pred_fail')}) | {N1_DRV.get('seconds')}s | errors {(N1_DRV.get('errors') or [])[:2]}", flush=True)
    _n1_sc = n1_scores(_n1_out, _n1_done, _n1_run, N1K)
    _n1_real = {m: v for m, v in _n1_sc.items() if not m.startswith('n1x:')}
    _n1_clash = sorted(m for m in _n1_real if any(x is not None for x in (GL_SCORES.get(m) or {}).values()))
    if not IS_RERUN:
        assert not _n1_clash, ('N1 molecule already scored by the H1 run', _n1_clash[:5])
        N1X_GL = {m: v for m, v in _n1_sc.items() if m.startswith('n1x:')}
        assert N1_DRV.get('status') == 'ok' and N1X_MIDS <= _n1_done, (N1_DRV.get('status'), sorted(N1X_MIDS - _n1_done)[:5])
        assert len(N1X_GL) >= 1 and N1_DRV.get('n_spec_merged') == _n1_exp_merged, (len(N1X_GL), N1_DRV.get('n_spec_merged'))
        _n1_nb1 = _n1_nbm = 0
        for _n1_m in N1X_MIDS:              # single-CE spectra: merged-CE mode == mean (production gl_runner) exactly;
            for _n1_sp in (_n1_out[_n1_m].get('spec') or {}).values():     # merged-CE ones: the I14b path acts
                if _n1_sp.get('nb') == 1:
                    assert _n1_sp['mean'] == _n1_sp['merged'], _n1_m
                    _n1_nb1 += 1
                else:
                    _n1_nbm += int(_n1_sp['mean'] != _n1_sp['merged'])
        # the dummy commit molecules carry 20,40,60 eV [M+H]+ spectra -> merged-CE proxies (8 in the local dry run): the
        # per-bucket prediction + merge() must run on them with the real model and change their scores vs mean CE
        assert _n1_exp_merged >= 1 and _n1_nbm >= 1, ('merged-CE path not exercised on the commit', _n1_exp_merged, _n1_nbm)
        _n1_rho = []                        # proxies (= the [M+H]+ peaks, cut) vs the real H1 GL scores of the source
        _n1_mean = n1_scores(_n1_out, _n1_done, _n1_x_items, N1K, mode='mean')
        for _n1_m, _n1_d in _n1_mean.items():
            _n1_h = GL_SCORES.get(_n1_m.split(':', 1)[1]) or {}
            _n1_p = [(v, _n1_h.get(c)) for c, v in _n1_d.items() if v is not None and _n1_h.get(c) is not None]
            if len(_n1_p) >= 4:
                _n1_r = _n1_spearman([a for a, _ in _n1_p], [b for _, b in _n1_p])
                if math.isfinite(_n1_r):
                    _n1_rho.append(_n1_r)
        N1X_STATS.update(gl_scored=len(N1X_GL), single_ce_identical=_n1_nb1, merged_changed=_n1_nbm, rho_n=len(_n1_rho),
                         rho_median=float(np.median(_n1_rho)) if _n1_rho else float('nan'))
        print(f"[N1X] synthetic exercise: molecules {N1X_STATS['molecules']} | N1 molecules {N1X_STATS['n1']} | proxies "
              f"built {N1X_STATS['proxies']} (NH4p {N1X_STATS['NH4p']}, WLp {N1X_STATS['WLp']}; empty "
              f"{N1X_STATS['proxy_empty']}) | proxy checks OK {N1X_STATS['proxy_checked']} | arm-B guard "
              f"{N1X_STATS['armB_guard']} | merged-CE spectra "
              f"{N1_DRV.get('n_spec_merged')} (expected {_n1_exp_merged}; merged != mean CE {_n1_nbm}) | single-CE "
              f"merged==mean {_n1_nb1} | GL-scored "
              f"synthetic molecules {len(N1X_GL)} | proxy (mean CE) vs H1 GL Spearman median "
              f"{N1X_STATS['rho_median']:.3f} over {len(_n1_rho)}", flush=True)
        assert not _n1_rho or N1X_STATS['rho_median'] >= 0.3, ('proxy scores do not track the H1 GL scores', _n1_rho)
    else:
        _n1_real = {m: v for m, v in _n1_real.items() if m not in _n1_clash}
    if N1_COMPLEMENT:                    # k2 variant A: E006's ICE fallback keeps the blind molecules ICEBERG covers
        _n1_n0 = len(_n1_real)
        _n1_real, N1_ICE_DROPPED = n1_complement(_n1_real, ICE_SCORES)
        if not IS_RERUN:                 # forced commit exercise of the filter on THIS commit's real ICE output: a
            #                              synthetic copy 'n1x:<mid>' stands for a blind molecule with <mid>'s ICE dict
            _n1_xi = {m: (ICE_SCORES.get(m.split(':', 1)[1]) if isinstance(ICE_SCORES, dict) else None) for m in N1X_GL}
            _n1_xk, _n1_xd = n1_complement(N1X_GL, _n1_xi)
            assert _n1_xd == {m for m, v in _n1_xi.items() if n1_ice_finite(v) > 0}, _n1_xd
            assert set(_n1_xk) == set(N1X_GL) - _n1_xd and all(_n1_xk[m] is N1X_GL[m] for m in _n1_xk)
            assert _n1_xd, ('complement filter not exercised: no synthetic N1 molecule whose source has a finite ICE '
                            'score', len(N1X_GL), sum(n1_ice_finite(v) for v in _n1_xi.values()))
            _n1_t = n1_complement({'f': {}, 'n': {}, 'x': {}, 'b': {}, 'm': {}, 'e': {}},
                                  {'f': {'p': 0.5, 'q': None}, 'n': {'p': None}, 'x': {'p': float('nan'), 'q': float('inf')},
                                   'b': {'p': 'bad', 'q': [1]}, 'e': {}})
            assert set(_n1_t[0]) == {'n', 'x', 'b', 'm', 'e'} and _n1_t[1] == {'f'}, _n1_t
            assert n1_complement({}, ICE_SCORES) == ({}, set()) and n1_complement({'f': {}}, None) == ({'f': {}}, set())
            N1X_STATS['complement'] = dict(synthetic=len(N1X_GL), dropped=len(_n1_xd), kept=len(_n1_xk))
            print(f"[N1X] complement filter exercise: synthetic N1 molecules with GL {len(N1X_GL)} | source ICE dict has a "
                  f"finite score -> dropped {len(_n1_xd)} | kept {len(_n1_xk)} | unit cases OK", flush=True)
        print(f"[N1] complement (k2 variant A): N1 molecules with GL scores {_n1_n0} | dropped (finite ICE score, E006 ICE "
              f"fallback kept) {len(N1_ICE_DROPPED)} | kept {len(_n1_real)}", flush=True)
    _n1_health = (f"N1 OK: {len(_n1_real)} of {len(_n1_items)} N1 molecules GL-scored "
                  f"({sum(sum(v is not None for v in d.values()) for d in _n1_real.values())} candidates), driver "
                  f"{N1_DRV.get('status')}, clashes dropped {len(_n1_clash)}")
    N1_MIDS, N1_GL_MIDS = {str(it['mid']) for it in _n1_items}, set(_n1_real)
    GL_SCORES.update(_n1_real)           # last statement: N1 scores exist only if everything above succeeded
    N1_HEALTH = _n1_health
except Exception as e:
    for _n1_m in list(N1_GL_MIDS):
        GL_SCORES.pop(_n1_m, None)
    N1_MIDS, N1_GL_MIDS = set(), set()
    N1_HEALTH = f'N1 FAILED: {e!r}'
    print('[N1] FAILED -> H1 behaviour (no N1 GL scores):', repr(e)[:600], flush=True)
    if not IS_RERUN:
        raise
print(f'[N1] GL-scored N1 molecules: {len(N1_GL_MIDS)} of {len(N1_MIDS)} | health: {N1_HEALTH} | '
      f'{time.time()-T0:.0f}s', flush=True)
# ---- /N1A_MCE ---------------------------------------------------------------------------------------------------
def gl_of(m):
    g = GL_SCORES.get(str(m), {}) if isinstance(GL_SCORES, dict) else {}
    return g if isinstance(g, dict) and any(v is not None for v in g.values()) else {}
def gl_rerank(order0, smis, keys, scs, forms, ice, gl, top_n, tag=''):
    # z(scs) + ICE_LAM z(ice) + GL_LAM z(gl) in the same slots; no GL scores / any failure -> order0 (= v4g order)
    if not gl:
        return order0
    try:
        o = gl_fuse.rerank_multi(smis, keys, scs, forms, [ice, gl], [ICE_LAM, GL_LAM], top_n=top_n)
    except Exception as e:
        print('gl rerank failed', tag, repr(e)); return order0
    for k, v in (('molecules', 1), ('changed_vs_ice_top25', int(o[:25] != order0[:25])), ('changed_vs_ice_top1', int(o[:1] != order0[:1]))):
        gl_stats[tag + k] = gl_stats.get(tag + k, 0) + v
    return o
for m in list(BASE):
    smis, keys, lib_max, scs, forms = BASE[m]
    ice = ICE_SCORES.get(str(m), {}) if ICE_SCORES else {}
    gl = gl_of(m)
    if BASE_ICE and ice and any(v is not None for v in ice.values()):
        try:
            order = ice_fuse.rerank(smis, keys, scs, forms, ice, lam=ICE_LAM, top_n=TOPN)
            order = gl_rerank(order, smis, keys, scs, forms, ice, gl, TOPN)
            s2 = [smis[i] for i in order]; k2 = [keys[i] for i in order]
            ice_stats['molecules'] += 1
            ice_stats['changed_top25'] += int(s2[:25] != smis[:25]); ice_stats['changed_top1'] += int(bool(s2) and s2[0] != smis[0])
            smis, keys = s2, k2
        except Exception as e:
            print('rerank failed', m, repr(e))
    BASE[m] = (smis[:25], keys[:25], lib_max)
    p = PC.get(m) if (ICE_PC and isinstance(PC, dict)) else None
    if p and p.get('pc_form') and ice:
        try:                             # re-order the PubChem list by z(f.z) + lam z(ice) inside same-formula groups;
            o = ice_fuse.rerank(p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, lam=ICE_LAM, top_n=25)
            o = gl_rerank(o, p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, gl, 25, 'pc_')
            fz0 = p['pc_fz'][0]          # the gate keeps using the ORIGINAL top f.z (rel), so gate decisions are unchanged
            p['pc'] = [p['pc'][i] for i in o]; p['pc_keys'] = [p['pc_keys'][i] for i in o]
            p['pc_fz'] = [fz0] + [p['pc_fz'][i] for i in o][1:]
            ice_stats['pc_changed'] = ice_stats.get('pc_changed', 0) + int(o[:1] != [0])
        except Exception as e:
            print('pc rerank failed', m, repr(e))
print('ICE rerank stats', ice_stats, f'{time.time()-T0:.0f}s')
try:                                     # N1A_MCE: the N1 GL scores also reach the PubChem-list re-rank above (log only)
    _n1_base = {str(m): m for m in BASE}
    _n1_pc = sum(1 for s in N1_GL_MIDS if ICE_PC and isinstance(PC, dict) and (PC.get(_n1_base.get(s, s)) or {}).get('pc_form')
                 and isinstance(ICE_SCORES, dict) and ICE_SCORES.get(s))
    print(f'[N1] PubChem-list GL re-rank: N1 molecules with GL scores {len(N1_GL_MIDS)} | with a gated PubChem list '
          f'(lib_max < LIB_TAU) and an ICE dict = re-ranked with their N1 GL scores {_n1_pc}', flush=True)
except Exception as e:
    print('[N1] PubChem-list count failed (log only):', repr(e), flush=True)
print('GL rerank stats', gl_stats, f'{time.time()-T0:.0f}s')


    


# %% [code]

'''## 8. Gated PubChem merge

PubChem-only candidates enter the main list only when there is no strong library match (`lib_max < 0.9`). When the PubChem fingerprint fit beats the best pool candidate by a wide margin, they take slots 2, 4, 6, 8 and 10. Otherwise they take slots 4, 8, 12, 16 and 20.
'''


    


# %% [code]

'''gated PubChem merge (v3.6, unchanged) -> submission.csv'''
def merge(base, base_keys, pc, pc_keys, slots, n=25):
    bk = set(base_keys)
    pcs = [s for s, k in zip(pc, pc_keys) if k not in bk]
    out, bi, pj = [], 0, 0
    for pos in range(1, n + 1):
        if pos in slots and pj < len(pcs):
            out.append(pcs[pj]); pj += 1
        elif bi < len(base):
            out.append(base[bi]); bi += 1
        elif pj < len(pcs):
            out.append(pcs[pj]); pj += 1
    return out
'''---- CLAW PROMOTION RULE (b417) --------------------------------------------------'''
'''The shipped gate only INTERLEAVES PubChem-only structures into fixed slots, so the'''
'''main list always keeps rank 1. When the best PubChem proposal is a strong outlier --'''
'''S = z(f.z over the pass-2 set) + 0.25*pop, carried out of the worker -- AND is well'''
'''documented (pop >= 5), let it lead; everything else keeps its usual slot.'''
'''NOTE: S is the Z-SCORED quantity (median ~6). The raw p['pc_fz'][0] (median -1437)'''
'''must never be compared to S_TAU; that would fire on ~71% of molecules.'''
'''EVIDENCE: b416 (promotion, ICEBERG broken) = 0.417 vs base415 (ICE broken, no'''
'''promotion) 0.374 and b415c (ICE working, no promotion) 0.393. The promotion overcame'''
'''losing ICE, so its marginal value is >= +0.024 (est. ~+0.037 using the notebook's own'''
'''ICE +0.013 / GL +0.006 measurements). b417 isolates it on the WORKING-ICE base.'''
S_TAU, POP_TAU = 6.0, 5.0


def promote(base, base_keys, pc, pc_keys, slots, n=25):
    """Confident PubChem proposal leads the usual list; every other entry keeps its slot."""
    usual = merge(base, base_keys, pc, pc_keys, slots, n=n)
    if not pc:
        return usual
    top = pc[0]
    return ([top] + [x for x in usual if x != top])[:n]


'''self-test of both pure functions (this also runs in the cheap smoke pass)'''
_b, _bk = ['A', 'B', 'C'], ['ka', 'kb', 'kc']
_p, _pk = ['X', 'Y'], ['kx', 'ky']
assert merge(_b, _bk, _p, _pk, [2, 4]) == ['A', 'X', 'B', 'Y', 'C'], merge(_b, _bk, _p, _pk, [2, 4])
assert promote(_b, _bk, _p, _pk, [2, 4]) == ['X', 'A', 'B', 'Y', 'C'], promote(_b, _bk, _p, _pk, [2, 4])
assert merge(_b, _bk, _p, _pk, [2, 4])[0] == 'A', 'shipped gate must keep rank 1'
assert merge(_b, _bk, [], [], [2, 4]) == ['A', 'B', 'C']
assert promote(_b, _bk, [], [], [2, 4]) == ['A', 'B', 'C']
assert promote(_b, _bk, _p, _pk, [2, 4])[:1] == ['X'] and len(promote(_b, _bk, _p, _pk, [2, 4])) == 5
print('merge/promote self-test OK')

rows, stats = [], dict(untouched=0, gentle=0, aggressive=0, no_pc=0, promoted=0)
for mid in te.molecule_id.unique():
    smis, keys, lib_max = BASE.get(mid, ([], [], 0.0))
    p = PC.get(mid)
    if p is None or not p['pc']:
        stats['no_pc'] += 1; final = smis
    elif lib_max >= LIB_TAU:
        stats['untouched'] += 1; final = smis
    else:
        rel = (p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]) - p['best_pool_fz'] if np.isfinite(p['best_pool_fz']) else 1e9
        slots = SLOTS_AGG if rel > REL_TH else SLOTS_GENTLE
        stats['aggressive' if rel > REL_TH else 'gentle'] += 1
        S = p.get('S'); pop_top = float(p.get('top_pop') or 0.0)
        # Skomuro LB 0.403 Rule: Promote on Popularity OR Structure-Evidence Margin!
        # rel > 250 captures confident PubChem outliers that popularity alone misses
        # v11: rel promotion threshold 150.0 (public evidence: seyitkaangunes E006 sweep; antriksharya e2rank2 base)
        if (S is not None and float(S) > S_TAU and pop_top >= POP_TAU) or (rel > 150.0):
            stats['promoted'] += 1
            final = promote(smis, keys, p['pc'], p['pc_keys'], slots)
        else:
            final = merge(smis, keys, p['pc'], p['pc_keys'], slots)
    if not final: final = ['CCO']
    rows.append((mid, ';'.join(final[:25])))
print('merge stats', stats)
sub = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
samp = pd.read_csv(os.path.join(COMP, 'sample_submission.csv'))
sub = samp[['molecule_id']].merge(sub, on='molecule_id', how='left')
sub['smiles'] = sub['smiles'].fillna('CCO')
assert len(sub) == len(samp) and sub.molecule_id.is_unique and sub.smiles.notna().all()
assert sub.smiles.map(lambda s: len(s.split(';'))).max() <= 25
sub.to_csv('submission.csv', index=False)
print('CLAW STAGE HEALTH |', CLAW_ICE_HEALTH, '|', CLAW_GL_HEALTH, flush=True)
try:
    _ice_n = json.load(open('/kaggle/working/ice_work/ice_out.json.meta.json')).get('n_mols_scored', 0)
    _gl_n = json.load(open('/kaggle/working/gl_work/gl_out.json.meta.json')).get('n_mols_scored', 0)
except Exception:
    _ice_n = _gl_n = -1
print(f'CLAW STAGE SCORED | ice={_ice_n} gl={_gl_n}', flush=True)
json.dump(dict(secs=round(time.time() - T0), merge=stats, n_errors=n_err,
               ice_health=CLAW_ICE_HEALTH, gl_health=CLAW_GL_HEALTH,
               ice_n_scored=_ice_n, gl_n_scored=_gl_n,
               families=MAN['families'],
               gpu=torch.cuda.get_device_name(0) if dev == 'cuda' else None), open('run_manifest.json', 'w'), indent=1)
print('done', sub.shape, f'{time.time()-T0:.0f}s')
sub.head()


    


# %% [code]

'''## 9. Two-engine fusion + final simulator re-rank

The two lists are combined by weighted reciprocal-rank fusion. A candidate's score is $\\frac{1}{3 + r_{\\text{main}}} + \\frac{0.6}{3 + r_{\\text{engine 2}}}$, and the top 40 are kept. ICEBERG and GLACIER then re-rank same-formula groups of the fused list one last time, and the top 25 become the submission.
'''


    


# %% [code]

ALPHA, KRR = 0.6, 3.0
CELL19_ICE = False       # b421: cell-19 ICE contribution EXPLICITLY off.
'''Reproduces BY DESIGN what b416 got by accident: when ICE errored it still wrote a'''
'''shell dict of all-None values, cell-19's `if ice:` gate passed, and ICE's column then'''
'''contributed nothing (covered count 0) -- leaving GLACIER alone. b420 (ICE real, both'''
'''applied) scored 0.402; b416 (GLACIER only) scored 0.417. This isolates that one'''
'''arithmetic difference on the healthy base.'''
from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _CMF2
from rdkit import Chem as _Ch2
def _form2(s):
    try:
        return _CMF2(_Ch2.MolFromSmiles(s))
    except Exception:
        return s
def fuse2(v_smis, e_smis, e_keys, n=40):
    sc, smi_of = {}, {}
    for r, s in enumerate(v_smis, 1):
        k = chem.score_key(s) or s
        sc[k] = sc.get(k, 0.0) + 1.0 / (KRR + r); smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
        if not k:
            continue
        sc[k] = sc.get(k, 0.0) + ALPHA / (KRR + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:n]
    return [smi_of[k] for k in order], order, [sc[k] for k in order]

PREFUSION_LISTS = {str(r.molecule_id): str(r.smiles).split(';') for r in pd.read_csv('submission.csv').itertuples(index=False)}
FUSION_OVERFLOW = {}
if ENG:
    v4 = pd.read_csv('submission.csv')
    changed = n_ice = 0; out = []
    for mid, s in zip(v4.molecule_id, v4.smiles):
        vs = [x for x in s.split(';') if x and x != 'CCO']
        e = ENG.get(str(mid))
        if e and e['smiles']:
            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])
            ice = ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}
            if ice:
                try:
                    o = ice_fuse.rerank(fsm, fk, fsc, [_form2(x) for x in fsm], (ice if CELL19_ICE else {}), lam=ICE_LAM, top_n=len(fsm))
                    _g = gl_of(mid)
                    if _g:
                        o = gl_fuse.rerank_multi(fsm, fk, fsc, [_form2(x) for x in fsm], [(ice if CELL19_ICE else {}), _g], [ICE_LAM, GL_LAM], top_n=len(fsm))
                    n_ice += int(o[:25] != list(range(min(25, len(o)))))
                    fsm = [fsm[i] for i in o]
                except Exception as ex:
                    print('fused ICE rerank failed', mid, repr(ex))
            changed += int(bool(vs) and fsm[:1] != vs[:1])
            FUSION_OVERFLOW[str(mid)] = list(fsm[25:])
            vs = fsm[:25]
        out.append((mid, ';'.join(vs[:25]) if vs else 'CCO'))
    fs = pd.DataFrame(out, columns=['molecule_id', 'smiles'])
    assert len(fs) == len(v4) and fs.molecule_id.is_unique and fs.smiles.notna().all()
    assert fs.smiles.map(lambda s: len(s.split(';'))).max() <= 25
    fs.to_csv('submission.csv', index=False)


print('V1 inference complete; applying separately audited final pass next.')


    


# %% [code]

'''## Final metric audit

The final pass cannot demote any existing metric-equivalent answer: it retains the first
occurrence of every valid identity in order, then appends new identities up to 25 candidates.
Runtime assertions check this for every active molecule. This guarantee applies to the
same predictions and current metric; it does not predict the score of a fresh inference run.

Outputs: `submission.csv`, `baseline_submission.csv`, `candidate_audit.csv`,
`candidate_audit.json`, and stage health in `run_manifest.json`.
'''


    


# %% [code]

from functools import lru_cache
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
_tautomer_canonicalizer = rdMolStandardize.TautomerEnumerator()
def canonicalize_smiles_to_ik14(smiles: str) -> str:
    """Canonicalize SMILES to InChIKey14, or None if parsing fails.

    Stereochemistry is removed *before* the tautomer canonicalization, not left to fall out of
    the InChIKey14 truncation afterwards. Truncating to InChIKey14 discards the stereo block, so
    it looks like it should already make stereochemistry free -- but the canonicalization runs
    first, and `TautomerEnumerator` will not move a double bond whose position a neighbouring
    stereocentre constrains. The same skeleton drawn with and without that stereocentre can
    therefore land on two different canonical tautomers, and different tautomers are different
    connectivity, which is the part of the InChIKey14 that survives truncation. Stripping first
    makes the canonical form depend only on the skeleton, which is what the competition promises.

    Both the answer key and the submission go through this function, so the two sides can never
    be treated asymmetrically.
    """
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        canonical_mol = _tautomer_canonicalizer.Canonicalize(mol)
        inchikey = Chem.MolToInchiKey(canonical_mol)
        return inchikey[:14]
    # RDKit's MolSanitizeException family subclasses ValueError; RuntimeError covers the C++
    # layer and TypeError covers Boost.Python argument errors. One participant's malformed
    # SMILES must never take the whole scoring run down, so a failure here is just a non-match.
    except (ValueError, RuntimeError, TypeError):
        return None

def heavy_atom_composition(smiles: str):
    """Heavy-atom element counts (excluding H, ignoring charge) as a hashable key, or None
    if parsing fails.

    Cheap (no tautomer enumeration) -- used to gate the expensive canonicalization. Anything
    that shares an InChIKey14 with an accepted structure has the same heavy-atom skeleton, so
    a heavy-atom-composition mismatch lets us skip canonicalization with no change to any
    score. We deliberately do NOT gate on full molecular formula: protomers / charge variants
    (e.g. a zwitterion C(=O)[O-]...[N+] vs its neutral-acid protomer C(=O)O...[N+]) share an
    InChIKey14 but differ in H count and charge, so a formula gate would wrongly reject them.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != 'H':
            counts[sym] = counts.get(sym, 0) + 1
    return tuple(sorted(counts.items()))

@lru_cache(maxsize=150000)
def metric_identity(smiles):
    key = canonicalize_smiles_to_ik14(smiles)
    return (key, heavy_atom_composition(smiles)) if key else None

def safe_extend(original, extra, limit=25):
    """Preserve first valid metric identities; fill only unoccupied ranks."""
    out, seen = [], set()
    for smi in list(original) + list(extra):
        smi = str(smi).strip()
        identity = metric_identity(smi)
        if identity is not None and identity not in seen:
            seen.add(identity)
            out.append(smi)
            if len(out) == limit:
                break
    return out

def first_ranks(smiles):
    ranks = {}
    for rank, smi in enumerate(smiles, 1):
        identity = metric_identity(smi)
        if identity is not None:
            ranks.setdefault(identity, rank)
    return ranks

def assert_monotone(original, revised):
    old, new = first_ranks(original), first_ranks(revised)
    assert all(identity in new and new[identity] <= rank for identity, rank in old.items())


    


# %% [code]

METRIC_SOURCE_SHA256 = 'c61d163f1bb768313fd0863822952f035f343924f37e3f5e4b5883ccef7e6201'
'''Final pass: no change to any V1 model, fusion weight, promotion gate or simulator.'''
import collections
import hashlib
from pathlib import Path
assert rdkit.__version__ == '2026.03.3'
assert n_err == 0, f'Base inference errors: {n_err}'
_before = pd.read_csv('submission.csv')
_before.to_csv('baseline_submission.csv', index=False)
_active = set(te.molecule_id.astype(str))
assert _active <= set(map(str, ENG)), 'Engine 2 did not cover all active molecules'
_stage_meta = {}
for _stage, _path in [('ice', 'ice_work/ice_out.json.meta.json'), ('glacier', 'gl_work/gl_out.json.meta.json')]:
    if Path(_path).is_file():
        _stage_meta[_stage] = json.loads(Path(_path).read_text())
    else:
        _stage_meta[_stage] = {'status': 'missing'}
assert _stage_meta['ice'].get('status') == 'ok', _stage_meta['ice']
if (te.adduct == '[M+H]+').any():
    assert _stage_meta['glacier'].get('status') == 'ok', _stage_meta['glacier']

_dm_smiles = _dm_mass = _dm_order = None
_dm_paths = sorted(glob.glob('/kaggle/input/**/dreams_np_mass.npy', recursive=True))
_dreams_used = 0
_audit_rows, _out = [], []
_targets = te.groupby('molecule_id').nm.median().to_dict()
for _row in _before.itertuples(index=False):
    _mid = str(_row.molecule_id)
    _original = [s.strip() for s in str(_row.smiles).split(';') if s.strip()]
    if _mid not in _active:  # Visible smoke placeholders; hidden rerun processes every molecule.
        _out.append((_row.molecule_id, _row.smiles))
        continue
    _ranks = first_ranks(_original)
    _clean = safe_extend(_original, [])
    _extras = list(FUSION_OVERFLOW.get(_mid, [])) + list(BASE_TAIL_CANDIDATES.get(_mid, []))
    _engine = ENG.get(_mid, {})
    _extras.extend(_engine.get('smiles', []))
    _pc = PC.get(_row.molecule_id, PC.get(_mid, {}))
    _extras.extend(_pc.get('pc', []))
    _revised = safe_extend(_clean, _extras)
    _after_ranked = len(_revised)
    _target = float(_targets.get(_row.molecule_id, _targets.get(_mid, np.nan)))
    if len(_revised) < 25 and np.isfinite(_target) and _target > 0 and _dm_paths:
        if _dm_mass is None:
            _dd = Path(_dm_paths[0]).parent
            _dm_mass = np.load(_dd / 'dreams_np_mass.npy').astype(np.float64)
            _dm_smiles = np.load(_dd / 'dreams_np_smiles.npy', allow_pickle=True)
            assert len(_dm_mass) == len(_dm_smiles)
            _dm_order = np.argsort(_dm_mass, kind='stable')
            _dm_sorted = _dm_mass[_dm_order]
        _lo, _hi = np.searchsorted(_dm_sorted, [_target * (1 - 10e-6), _target * (1 + 10e-6)])
        _ix = _dm_order[_lo:_hi]
        _ix = _ix[np.argsort(np.abs(_dm_mass[_ix] - _target), kind='stable')]
        _mass_candidates = [s.decode() if isinstance(s, bytes) else str(s) for s in _dm_smiles[_ix]]
        _revised = safe_extend(_revised, _mass_candidates)
        _dreams_used += len(_revised) - _after_ranked
    assert _revised, f'No valid candidates for {_mid}'
    assert len(_revised) <= 25 and len(first_ranks(_revised)) == len(_revised)
    assert_monotone(_original, _revised)
    _new_ranks = first_ranks(_revised)
    _audit_rows.append(dict(molecule_id=_mid, original_count=len(_original),
        original_unique=len(_ranks), invalid_count=sum(metric_identity(s) is None for s in _original),
        duplicate_count=sum(metric_identity(s) is not None for s in _original) - len(_ranks),
        revised_count=len(_revised), ranked_added=_after_ranked - len(_clean),
        dreams_added=len(_revised) - _after_ranked,
        promoted_original=sum(_new_ranks[k] < v for k, v in _ranks.items()),
        changed=_original != _revised))
    _out.append((_row.molecule_id, ';'.join(_revised)))
_final = pd.DataFrame(_out, columns=['molecule_id', 'smiles'])
_expected = pd.read_csv(os.path.join(COMP, 'sample_submission.csv')).molecule_id
assert _final.molecule_id.is_unique and set(_final.molecule_id) == set(_expected)
assert _final.smiles.notna().all() and _final.smiles.str.split(';').map(len).between(1, 25).all()
if IS_RERUN:
    assert len(_active) == len(_final), 'A hidden molecule was left unprocessed'
_final.to_csv('submission.csv', index=False)
_audit_df = pd.DataFrame(_audit_rows)
_audit_df.to_csv('candidate_audit.csv', index=False)
_summary = dict(experiment='V1.1 metric-safe tail', rdkit=rdkit.__version__,
    hidden_rerun=IS_RERUN, active_molecules=len(_active), output_rows=len(_final),
    changed_molecules=int(_audit_df.changed.sum()),
    invalid_removed=int(_audit_df.invalid_count.sum()),
    equivalent_duplicates_removed=int(_audit_df.duplicate_count.sum()),
    existing_candidates_promoted=int(_audit_df.promoted_original.sum()),
    ranked_candidates_added=int(_audit_df.ranked_added.sum()),
    dreams_candidates_added=int(_audit_df.dreams_added.sum()),
    monotone_rank_assertions_passed=True,
    metric_source_sha256=METRIC_SOURCE_SHA256,
    baseline_sha256=hashlib.sha256(Path('baseline_submission.csv').read_bytes()).hexdigest(),
    submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),
    stages=_stage_meta, runtime_seconds=round(time.time()-T0, 1),
    validation_note='Visible test is a smoke check only and overlaps training; no generalization score claimed.')
Path('candidate_audit.json').write_text(json.dumps(_summary, indent=2, default=str))
print(json.dumps({k:v for k,v in _summary.items() if k != 'stages'}, indent=2))
display(_audit_df)


    


# %% [code]

'''Independent experimental overlay
One vote per official metric identity per engine; compact unique ranks before the 40-candidate fusion shortlist. Preserve the existing PubChem gate and same-formula GLACIER logic. This tests duplicate-vote bias, not a coefficient search. It can change top-1 and reduce the score.'''

def metric_fusion(first, second, identity, alpha=0.6, offset=3.0, limit=40):
    scores, representatives = {}, {}
    collapsed = 0
    for candidates, weight in ((first, 1.0), (second, alpha)):
        seen = set()
        for candidate in candidates:
            key = identity(candidate)
            if key is None:
                continue
            if key in seen:
                collapsed += 1
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (offset + len(seen))
            representatives.setdefault(key, candidate)
    ordered = sorted(scores, key=lambda key: -scores[key])[:limit]
    return [representatives[key] for key in ordered], [scores[key] for key in ordered], collapsed

EXPERIMENT = 'metric_identity_fusion'
_reference = pd.read_csv('submission.csv')
_reference.to_csv('submission_reference.csv', index=False)
_rows, _audit = [], []
for _row in _reference.itertuples(index=False):
    _mid = str(_row.molecule_id)
    _original = str(_row.smiles).split(';')
    _revised, _collapsed = list(_original), 0
    _engine = ENG.get(_mid)
    if _mid in PREFUSION_LISTS and _engine and _engine.get('smiles'):
        _first = [s for s in PREFUSION_LISTS[_mid] if s and s != 'CCO']
        _candidates, _scores, _collapsed = metric_fusion(_first, _engine['smiles'], metric_identity,
                                                       alpha=ALPHA, offset=KRR)
        _keys = [chem.score_key(s) or s for s in _candidates]
        _ice = ICE_SCORES.get(_mid, {}) if isinstance(ICE_SCORES, dict) else {}
        _gl = gl_of(_mid)
        if _ice and _gl:
            _order = gl_fuse.rerank_multi(_candidates, _keys, _scores,
                [_form2(s) for s in _candidates], [{}, _gl], [ICE_LAM, GL_LAM], top_n=len(_candidates))
            _candidates = [_candidates[i] for i in _order]
        if _candidates:
            _revised = safe_extend(_candidates[:25], _candidates[25:] + _original)
    _rows.append((_row.molecule_id, ';'.join(_revised)))
    _audit.append(dict(molecule_id=_mid, collapsed_source_votes=_collapsed,
        changed=_revised != _original, changed_top1=_revised[:1] != _original[:1]))

_result = pd.DataFrame(_rows, columns=['molecule_id', 'smiles'])
assert list(_result.molecule_id) == list(_reference.molecule_id)
assert _result.molecule_id.is_unique and _result.smiles.notna().all()
assert _result.smiles.map(lambda value: 1 <= len(value.split(';')) <= 25).all()
_result.to_csv('submission.csv', index=False)
pd.DataFrame(_audit).to_csv('experiment_audit.csv', index=False)
_changed = sum(row['changed'] for row in _audit)
_report = dict(experiment=EXPERIMENT, changed_molecules=_changed,
               changed_top1=sum(row['changed_top1'] for row in _audit),
               reference_sha256=sha256('submission_reference.csv'),
               submission_sha256=sha256('submission.csv'), measured_score=None)
json.dump(_report, open('experiment_audit.json', 'w'), indent=2)
print(_report)
if not _changed:
    print('NO-OP: predictions match the reference; do not spend a second submission on this file.')


    


# %% [code]

"""Observe the frozen source output and preserve same-run fusion inputs for later work."""


def _e004_postflight(scope):
    import hashlib
    import json
    import time
    from pathlib import Path
    import pandas as pd
    import rdkit

    pre = scope['E004_PREFLIGHT']
    ids = set(pre['input_ids'])
    processed = set(scope['te'].molecule_id.astype(str))
    full = bool(scope['IS_RERUN'])
    assert processed == (ids if full else set(sorted(ids)[:scope['SMOKE_N']]))
    frame = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
    assert list(frame.columns) == ['molecule_id', 'smiles']
    assert frame.molecule_id.is_unique and set(frame.molecule_id) == ids
    assert frame.smiles.map(lambda s: 1 <= len(s.split(';')) <= 25).all()
    base, eng, pc = scope['BASE'], scope['ENG'], scope['PC']
    assert set(map(str, base)) == processed
    assert processed <= set(map(str, eng))
    assert scope['n_err'] == 0
    if not full:
        assert all(v[0] for v in base.values())
        assert all(eng[mid]['smiles'] and pc.get(mid, {}).get('pc') for mid in processed)
    stages = {}
    for stage in ('ice', 'gl'):
        path = Path(f'{stage}_work/{stage}_out.json.meta.json')
        meta = json.loads(path.read_text()) if path.is_file() else {'status': 'missing'}
        # The frozen source owns full-run failure policy. Extra acceptance checks
        # only gate the visible smoke, where all expected stages are exercised.
        if not full:
            assert meta['status'] == 'ok' and meta['device'].startswith('cuda'), (stage, meta)
            assert meta['rdkit'] == '2025.03.6'
            assert not meta.get('errors') and meta.get('n_pred_fail', 0) == 0
            assert meta.get('n_feat_fail', 0) == 0
            if meta.get('n_mols_covered', 0):
                assert meta['n_mols_scored'] > 0 and meta['n_scored'] > 0
        stages[stage] = meta
    payload = {
        'prefusion': {mid: scope['PREFUSION_LISTS'][mid] for mid in sorted(processed)},
        'engine2': {mid: eng[mid] for mid in sorted(processed)},
        'pubchem': {mid: pc.get(mid, {}) for mid in sorted(processed)},
        'base_tail': {mid: scope['BASE_TAIL_CANDIDATES'][mid] for mid in sorted(processed)},
        'fusion_overflow': {mid: scope['FUSION_OVERFLOW'].get(mid, []) for mid in sorted(processed)},
    }
    Path('e004_candidate_inputs.json').write_text(json.dumps(payload, indent=2, default=str) + '\n')
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
        'submission.csv', 'baseline_submission.csv', 'submission_reference.csv',
        'candidate_audit.json', 'experiment_audit.json', 'e004_candidate_inputs.json')}
    report = dict(experiment='E004', source_version_id=356740494, source_version=80,
                  source_code_sha256='eee496a371fce179b4aeda1c5082a58212c724eaf1ca4ba0c943bbe8d7196ef2',
                  **pre, is_full_run=full, processed_ids=sorted(processed),
                  processed_molecules=len(processed), input_molecules=len(ids),
                  output_rows=len(frame), placeholder_fill_rows=len(ids - processed),
                  base_errors=int(scope['n_err']), base_nonempty=sum(bool(v[0]) for v in base.values()),
                  engine2_nonempty=sum(bool(eng[mid]['smiles']) for mid in processed),
                  pubchem_nonempty=sum(bool(pc.get(mid, {}).get('pc')) for mid in processed),
                  main_rdkit=rdkit.__version__, stages=stages, hashes=hashes,
                  elapsed_seconds=round(time.time() - scope['T0'], 2),
                  limitations=['Visible smoke is not independent validation.',
                               'Author 0.436 is not this reproduction score.',
                               'Whole v80 recipe differs from E003 in multiple factors.'])
    Path('e004_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('E004_VERIFICATION', json.dumps(report))
    return report


E004_REPORT = _e004_postflight(globals())
del _e004_postflight


    


# %% [markdown]
# E006: window=40, ICE fallback=True
#
# All E004 code above, including both observers, is unchanged. The following isolated pass saves the E004 output bytes before treatment.


# %% [code]

# v7c: MetFrag-lite fragment evidence for molecules with NO forward-model coverage (no GLACIER, no finite ICEBERG)
FRAG_GAP = {}
try:
    FRAG_SRC = '"""Untrained MetFrag-lite re-scorer: how much MS/MS intensity a candidate\'s <=2-bond-cleavage fragments explain.\n\nIndependent of pv.py (same idea as pv.fragment_masses / explain_score_adduct, new code): enumerate the neutral masses\nof the connected pieces left after removing 1 or 2 bonds of the heavy-atom graph (implicit H stay on their atom),\nremember the fewest cleavages that give each mass, then match every peak against\n`neutral fragment + dh*H + charge carrier` for dh in `h_shifts` and the carriers the precursor adduct allows.\nA peak\'s credit is the weight of its cheapest explanation (1 for 0/1 cleavages, `w2` for 2, `w3` for 3),\nand a spectrum\'s score is the credited share of its (weighted) intensity. Defaults follow the forum ablation on a\n241-structure same-formula isomer panel (+0.065 MRR over the common baseline): w2=0.6, linear intensity,\ntol 0.005 Da, H shifts -2..+3, at most 2 cleavages. `BASELINE_CFG` is the common/pv.py-style setting.\n\n    score_molecule(spectra, cand_smiles, cfg=None, cache=None) -> list[float | None]\n\n`spectra`: list of dicts {mz: ndarray, it: ndarray (base peak = 1), adduct: str, prec: float, mode: +1/-1}.\nReturns one score per candidate (mean over the spectra that have peaks; higher = better), None if the SMILES does\nnot parse. Fragment masses are cached per SMILES in `cache` (a dict; pass one in to reuse it across molecules).\nThe enumeration is numba-compiled when numba is importable (pure-Python fallback otherwise, ~100x slower).\n"""\nfrom __future__ import annotations\n\nimport numpy as np\n\n# monoisotopic masses (same values as pv.AMU); other elements come from RDKit\'s periodic table\nAMU = {\'C\': 12.0, \'H\': 1.00782503207, \'N\': 14.0030740048, \'O\': 15.9949146196, \'P\': 30.97376163,\n       \'S\': 31.97207100, \'F\': 18.99840322, \'Cl\': 34.96885268, \'Br\': 78.9183371, \'I\': 126.904473,\n       \'Na\': 22.9897692809, \'K\': 38.96370668, \'Si\': 27.9769265325, \'B\': 11.0093054, \'Se\': 79.9165213}\nH = AMU[\'H\']\nE = 0.00054857990\nPROTON = H - E\nH2O = 2 * H + AMU[\'O\']\nNH4 = AMU[\'N\'] + 4 * H\nFORMATE = AMU[\'C\'] + 2 * H + 2 * AMU[\'O\']\n\n# charge carriers a fragment ion may keep, per precursor adduct (added to the neutral fragment mass)\nCARRIERS = {\n    \'[M+H]+\': (PROTON,), \'[M-H2O+H]+\': (PROTON,), \'[M-2H2O+H]+\': (PROTON,),\n    \'[M+NH4]+\': (PROTON, NH4 - E), \'[M+Na]+\': (PROTON, AMU[\'Na\'] - E), \'[M+K]+\': (PROTON, AMU[\'K\'] - E),\n    \'[M-H]-\': (-PROTON,), \'[M-H2O-H]-\': (-PROTON,), \'[M+CH2O2-H]-\': (-PROTON, FORMATE - PROTON),\n    \'[M+Cl]-\': (-PROTON, AMU[\'Cl\'] + E),\n}\n\nDEFAULT_CFG = dict(\n    w1=1.0,                      # credit for a peak explained by the precursor or a 1-bond cleavage\n    w2=0.6,                      # credit for a peak only explained by a 2-bond cleavage\n    w3=0.3,                      # credit for 3-bond cleavages (only used when max_breaks=3)\n    intensity=\'linear\',          # \'linear\' | \'sqrt\' | a float power applied to `it`\n    mz_power=0.0,                # extra peak weight (mz / prec) ** mz_power (0 = off)\n    tol=0.005,                   # absolute m/z tolerance, Da\n    h_shifts=tuple(range(-2, 4)),  # hydrogen shifts dh applied to neutral fragments (-2..+3)\n    max_breaks=2,                # bond cleavages per fragment (1..3)\n    max_bonds_2=100,             # above this many bonds only 1-bond cleavages are enumerated (cost cap)\n    max_bonds_3=40,              # above this many bonds 3-bond cleavages are skipped\n)\n# the common baseline (pv.py-style): all cleavages equal, sqrt intensity, 0.01 Da, H shifts -2..+2\nBASELINE_CFG = dict(DEFAULT_CFG, w2=1.0, intensity=\'sqrt\', tol=0.01, h_shifts=tuple(range(-2, 3)))\n\n\ndef resolve_cfg(cfg=None):\n    """DEFAULT_CFG updated with `cfg` (unknown keys raise, so typos in variants do not pass silently)."""\n    out = dict(DEFAULT_CFG)\n    for k, v in (cfg or {}).items():\n        if k not in out:\n            raise KeyError(f\'frag_rescore: unknown cfg key {k!r}\')\n        out[k] = v\n    out[\'h_shifts\'] = tuple(int(d) for d in out[\'h_shifts\'])\n    if not 1 <= int(out[\'max_breaks\']) <= 3:\n        raise ValueError(\'frag_rescore: max_breaks must be 1, 2 or 3\')\n    return out\n\n\n# ---------------------------------------------------------------------------------------------------------------\n# fragment enumeration (numba kernel with a pure-Python fallback: same source)\n# ---------------------------------------------------------------------------------------------------------------\ndef _find(parent, a):\n    while parent[a] != a:\n        parent[a] = parent[parent[a]]\n        a = parent[a]\n    return a\n\n\ndef _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, k):\n    """Union the kept bonds; if the graph splits, append each piece\'s mass with cleavage count k."""\n    for a in range(n):\n        parent[a] = a\n        acc[a] = 0.0\n    ncomp = n\n    for e in range(ea.shape[0]):\n        if removed[e]:\n            continue\n        ra = _find(parent, ea[e])\n        rb = _find(parent, eb[e])\n        if ra != rb:\n            parent[ra] = rb\n            ncomp -= 1\n    if ncomp < 2:\n        return cnt\n    for a in range(n):\n        r = _find(parent, a)\n        acc[r] += w[a]\n    for a in range(n):\n        if parent[a] == a:\n            out_m[cnt] = acc[a]\n            out_k[cnt] = k\n            cnt += 1\n    return cnt\n\n\ndef _enumerate(n, w, ea, eb, max_breaks, base_components):\n    """All piece masses after removing 1..max_breaks bonds (k = number removed); row 0.. are the intact pieces."""\n    nb = ea.shape[0]\n    n2 = nb * (nb - 1) // 2 if max_breaks >= 2 else 0\n    n3 = nb * (nb - 1) * (nb - 2) // 6 if max_breaks >= 3 else 0\n    cap = base_components + (base_components + 1) * nb + (base_components + 2) * n2 + (base_components + 3) * n3\n    out_m = np.empty(cap, np.float64)\n    out_k = np.empty(cap, np.int8)\n    parent = np.empty(n, np.int64)\n    acc = np.empty(n, np.float64)\n    removed = np.zeros(nb, np.bool_)\n    cnt = 0\n    for i in range(nb):\n        removed[i] = True\n        cnt = _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, 1)\n        if max_breaks >= 2:\n            for j in range(i + 1, nb):\n                removed[j] = True\n                cnt = _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, 2)\n                if max_breaks >= 3:\n                    for l in range(j + 1, nb):\n                        removed[l] = True\n                        cnt = _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, 3)\n                        removed[l] = False\n                removed[j] = False\n        removed[i] = False\n    return out_m[:cnt], out_k[:cnt]\n\n\ntry:  # numba is on the Kaggle image and in requirements.txt; NUMBA_CACHE_DIR is set by nb/01_setup.py\n    from numba import njit\n    _find = njit(cache=True, nogil=True)(_find)\n    _emit_components = njit(cache=True, nogil=True)(_emit_components)\n    _enumerate = njit(cache=True, nogil=True)(_enumerate)\n    HAVE_NUMBA = True\nexcept Exception:  # pragma: no cover - fallback path\n    HAVE_NUMBA = False\n\n\ndef mol_graph(smi):\n    """(atom masses incl. implicit H, bond begin idx, bond end idx, n connected pieces) or None if unparsable."""\n    from rdkit import Chem, RDLogger\n    RDLogger.DisableLog(\'rdApp.*\')\n    try:\n        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None\n    except Exception:\n        m = None\n    if m is None or m.GetNumAtoms() == 0:\n        return None\n    pt = Chem.GetPeriodicTable()\n    w = np.empty(m.GetNumAtoms(), np.float64)\n    for a in m.GetAtoms():\n        s = a.GetSymbol()\n        am = AMU.get(s)\n        if am is None:\n            am = pt.GetMostCommonIsotopeMass(s)\n        w[a.GetIdx()] = am + a.GetTotalNumHs() * H\n    ea = np.array([b.GetBeginAtomIdx() for b in m.GetBonds()], np.int64)\n    eb = np.array([b.GetEndAtomIdx() for b in m.GetBonds()], np.int64)\n    return w, ea, eb, len(Chem.GetMolFrags(m))\n\n\ndef fragment_levels(smi, max_breaks=2, max_bonds_2=100, max_bonds_3=40):\n    """Sorted neutral fragment masses grouped by the fewest cleavages that produce them: [k=0, k=1, ..., k=max].\n\n    k=0 holds the intact molecule (sum of its pieces, and each piece if the SMILES has several). None if the\n    SMILES does not parse.\n    """\n    g = mol_graph(smi)\n    if g is None:\n        return None\n    w, ea, eb, npieces = g\n    nb = len(ea)\n    mb = int(max_breaks)\n    if mb >= 3 and nb > max_bonds_3:\n        mb = 2\n    if mb >= 2 and nb > max_bonds_2:\n        mb = 1\n    levels = [np.array([w.sum()])]\n    if npieces > 1:  # salts / mixtures: each intact piece also counts as uncleaved\n        from rdkit import Chem\n        m = Chem.MolFromSmiles(smi)\n        levels = [np.unique(np.concatenate([[w.sum()], [w[list(ix)].sum() for ix in Chem.GetMolFrags(m)]]))]\n    if nb == 0:\n        return levels + [np.zeros(0) for _ in range(int(max_breaks))]\n    fm, fk = _enumerate(len(w), w, ea, eb, mb, npieces)\n    fm = np.concatenate([levels[0], fm])\n    fk = np.concatenate([np.zeros(len(levels[0]), np.int8), fk])\n    key = np.round(fm, 5)\n    order = np.lexsort((fk, key))                  # by mass, then fewest cleavages first\n    key, fm, fk = key[order], fm[order], fk[order]\n    first = np.ones(len(key), bool)\n    first[1:] = key[1:] != key[:-1]\n    fm, fk = fm[first], fk[first]\n    return [np.sort(fm[fk == k]) for k in range(int(max_breaks) + 1)]\n\n\n# ---------------------------------------------------------------------------------------------------------------\n# scoring\n# ---------------------------------------------------------------------------------------------------------------\ndef _carriers(adduct, mode):\n    c = CARRIERS.get(adduct)\n    if c is not None:\n        return c\n    if isinstance(adduct, str) and adduct.endswith(\'-\'):\n        return (-PROTON,)\n    if isinstance(adduct, str) and adduct.endswith(\'+\'):\n        return (PROTON,)\n    return (PROTON,) if (mode or 1) > 0 else (-PROTON,)\n\n\ndef _prep_spectrum(s, c):\n    """Peak weights and the neutral masses each peak would correspond to, for every (carrier, dh) pair."""\n    mz = np.asarray(s.get(\'mz\', ()), np.float64).ravel()\n    it = np.asarray(s.get(\'it\', ()), np.float64).ravel()\n    n = min(len(mz), len(it))\n    mz, it = mz[:n], it[:n]\n    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)\n    mz, it = mz[ok], it[ok]\n    if len(mz) == 0:\n        return None\n    p = c[\'intensity\']\n    iw = it if p == \'linear\' else np.sqrt(it) if p == \'sqrt\' else it ** float(p)\n    if c[\'mz_power\']:\n        prec = float(s.get(\'prec\') or mz.max())\n        iw = iw * (mz / prec) ** float(c[\'mz_power\'])\n    tot = iw.sum()\n    if not tot > 0:\n        return None\n    shifts = np.array([car + dh * H for car in _carriers(s.get(\'adduct\'), s.get(\'mode\')) for dh in c[\'h_shifts\']])\n    q = (mz[None, :] - shifts[:, None]).ravel()  # neutral mass needed, per (shift, peak)\n    return iw / tot, q, len(shifts), len(mz)\n\n\ndef _score_spectrum(levels, weights, prep, tol):\n    iw, q, ns, npk = prep\n    best = np.zeros(npk)\n    for arr, wt in zip(levels, weights):\n        if len(arr) == 0 or wt <= 0:\n            continue\n        a = np.searchsorted(arr, q - tol, \'left\')\n        b = np.searchsorted(arr, q + tol, \'right\')\n        hit = (b > a).reshape(ns, npk).any(axis=0)\n        best = np.where(hit & (wt > best), wt, best)\n    return float(iw @ best)\n\n\ndef candidate_levels(smi, cfg=None, cache=None):\n    """Cached fragment_levels for one SMILES under `cfg` (only the enumeration keys matter)."""\n    return _levels_cached(smi, resolve_cfg(cfg), cache)\n\n\ndef _levels_cached(smi, c, cache):\n    key = (smi, int(c[\'max_breaks\']), int(c[\'max_bonds_2\']), int(c[\'max_bonds_3\']))\n    if cache is not None and key in cache:\n        return cache[key]\n    try:\n        lv = fragment_levels(smi, c[\'max_breaks\'], c[\'max_bonds_2\'], c[\'max_bonds_3\'])\n    except Exception:\n        lv = None\n    if cache is not None:\n        cache[key] = lv\n    return lv\n\n\ndef score_molecule(spectra, cand_smiles, cfg=None, cache=None):\n    """One score per candidate: mean explained-intensity share over the molecule\'s spectra (None: unparsable)."""\n    c = resolve_cfg(cfg)\n    cache = {} if cache is None else cache\n    weights = [1.0, float(c[\'w1\']), float(c[\'w2\']), float(c[\'w3\'])][: int(c[\'max_breaks\']) + 1]\n    preps = [p for p in (_prep_spectrum(s, c) for s in (spectra or [])) if p is not None]\n    tol = float(c[\'tol\'])\n    out = []\n    for smi in cand_smiles:\n        lv = _levels_cached(smi, c, cache)\n        if lv is None:\n            out.append(None)\n            continue\n        if not preps:\n            out.append(0.0)\n            continue\n        out.append(float(np.mean([_score_spectrum(lv, weights, p, tol) for p in preps])))\n    return out\n\n\ndef warmup():\n    """Compile the numba kernel once (a few seconds) so the first real molecule is not charged for it."""\n    fragment_levels(\'CCOC(=O)c1ccccc1O\')\n    return HAVE_NUMBA\n'
    open('/kaggle/working/frag_rescore.py', 'w').write(FRAG_SRC)
    if '/kaggle/working' not in sys.path:
        sys.path.insert(0, '/kaggle/working')
    import frag_rescore
    frag_rescore.warmup()
    _fcache = {}
    _by = {str(m): g for m, g in te.groupby('molecule_id', sort=False)}
    _t0 = time.time(); _nf = 0
    for _mid, _lst in PREFUSION_LISTS.items():
        _ice = ICE_SCORES.get(_mid, {}) if isinstance(ICE_SCORES, dict) else {}
        if gl_of(_mid) or any(v is not None for v in (_ice or {}).values()):
            continue
        _e = ENG.get(_mid) or {}
        _c = [s for s in dict.fromkeys(list(_lst) + list(_e.get('smiles', []))) if s and s != 'CCO']
        _g = _by.get(_mid)
        if _g is None or len(_c) < 2:
            continue
        _spec = [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                      adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == 'positive' else -1)
                 for r in _g.itertuples()][:16]
        try:
            _sc = frag_rescore.score_molecule(_spec, _c, cache=_fcache)
            FRAG_GAP[_mid] = {s: float(v) for s, v in zip(_c, _sc) if v is not None}
            _nf += 1
        except Exception as _ex:
            print('frag failed', _mid, repr(_ex))
    print('v7c FRAG gap molecules', _nf, 'of', len(PREFUSION_LISTS), f'{time.time()-_t0:.0f}s', flush=True)
except Exception as _e:
    FRAG_GAP = {}
    print('v7c FRAG FAILED -> v7a behaviour:', repr(_e), flush=True)


    


# %% [code]

def _factorial_entry(scope):
    """Append-only E005–E007 overlay; original v80 functions remain authoritative.

    Embedded function-locally after the complete frozen E004 notebook. This module
    never installs dependencies, changes model scores, or recanonicalizes SMILES.
    """

    import hashlib
    import json
    import math
    from pathlib import Path


    def _finite_coverage(scores, candidates, keys):
        count = 0
        for smi, key in zip(candidates, keys):
            value = scores.get(smi)
            if value is None:
                value = scores.get(key)
            try:
                count += value is not None and math.isfinite(float(value))
            except (TypeError, ValueError):
                pass
        return count


    def rerank_row(scope, mid, original, *, window, ice_fallback, processed):
        """Rebuild final metric fusion from same-run prefusion and engine lists.

        ``original`` is the metric-safe reference BEFORE the frozen final overlay,
        not the E004 final output. The original identity, representative selection,
        same-formula slot rerank and tail fill are reused without modification.
        """
        if window not in (40, 80):
            raise ValueError("Only the prespecified 40/80 candidate windows are supported")
        original = list(original)
        audit = {
            "molecule_id": mid,
            "processed": bool(processed),
            "branch": "placeholder",
            "candidate_count": 0,
            "collapsed_source_votes": 0,
            "ice_finite_count": 0,
            "gl_finite_count": 0,
            "fallback_changed": False,
            "fallback_changed_top1": False,
        }
        if not processed:
            return original, audit
        audit["branch"] = "no_engine"
        engine = scope["ENG"].get(mid)
        if mid not in scope["PREFUSION_LISTS"] or not engine or not engine.get("smiles"):
            return original, audit
        first = [s for s in scope["PREFUSION_LISTS"][mid] if s and s != "CCO"]
        candidates, scores, collapsed = scope["metric_fusion"](
            first,
            engine["smiles"],
            scope["metric_identity"],
            alpha=scope["ALPHA"],
            offset=scope["KRR"],
            limit=window,
        )
        keys = [scope["chem"].score_key(s) or s for s in candidates]
        ice_scores = scope["ICE_SCORES"]
        ice = ice_scores.get(mid, {}) if isinstance(ice_scores, dict) else {}
        gl = scope["gl_of"](mid)
        audit.update(
            branch="unrescored",
            candidate_count=len(candidates),
            collapsed_source_votes=collapsed,
            ice_finite_count=_finite_coverage(ice, candidates, keys),
            gl_finite_count=_finite_coverage(gl, candidates, keys),
        )
        before = list(candidates)
        if ice and gl:
            # Preserve v80's exact GL-present gate and GL-only score arguments.
            order = scope["gl_fuse"].rerank_multi(
                candidates,
                keys,
                scores,
                [scope["_form2"](s) for s in candidates],
                [{}, gl],
                [scope["ICE_LAM"], scope["GL_LAM"]],
                top_n=len(candidates),
            )
            candidates = [candidates[i] for i in order]
            audit["branch"] = "glacier"
        elif (ice_fallback and not gl and scope.get("FRAG_GAP", {}).get(mid)
              and not any(v is not None for v in (ice or {}).values())):
            # v7c: no forward-model evidence at all -> MetFrag-lite gap re-rank (lambda 0.5), once
            order = scope["gl_fuse"].rerank_multi(
                candidates, keys, scores, [scope["_form2"](s) for s in candidates],
                [scope["FRAG_GAP"][mid]], [0.5], top_n=len(candidates))
            candidates = [candidates[i] for i in order]
            audit["branch"] = "frag_gap"
        elif ice_fallback and ice and not gl:
            # Public v1 / script 356785434: ICE is used only where gl_of(mid) is empty.
            # Keep its truthy-dict gate: all-None ICE is an observable harmless no-op.
            order = scope["ice_fuse"].rerank(
                candidates,
                keys,
                scores,
                [scope["_form2"](s) for s in candidates],
                ice,
                lam=scope["ICE_LAM"],
                top_n=len(candidates),
            )
            candidates = [candidates[i] for i in order]
            audit["branch"] = "ice_fallback"
        result = (
            scope["safe_extend"](candidates[:25], candidates[25:] + original)
            if candidates
            else original
        )
        if audit["branch"] == "ice_fallback":
            without_ice = (
                scope["safe_extend"](before[:25], before[25:] + original) if before else original
            )
            audit["fallback_changed"] = result != without_ice
            audit["fallback_changed_top1"] = result[:1] != without_ice[:1]
        return result, audit


    def run_overlay(scope, *, experiment, window, ice_fallback, provenance, workdir="."):
        """Save the E004 bytes, validate same-run control, then emit one treatment."""
        import pandas as pd

        root = Path(workdir)
        path = root / "submission.csv"
        control_bytes = path.read_bytes()
        control_sha = hashlib.sha256(control_bytes).hexdigest()
        base_report = json.loads((root / "e004_verification.json").read_text())
        assert base_report["experiment"] == "E004"
        assert base_report["hashes"]["submission.csv"] == control_sha
        # Copy bytes directly; never reserialize or overwrite original E004 audit files.
        (root / "submission_e004_control.csv").write_bytes(control_bytes)
        control = pd.read_csv(path, dtype=str, keep_default_na=False)
        reference = pd.read_csv(root / "submission_reference.csv", dtype=str, keep_default_na=False)
        assert list(reference.columns) == list(control.columns) == ["molecule_id", "smiles"]
        assert list(reference.molecule_id) == list(control.molecule_id)
        assert control.molecule_id.is_unique
        processed = set(base_report["processed_ids"])
        assert processed == set(scope["te"].molecule_id.astype(str))
        controls = dict(zip(control.molecule_id, control.smiles))
        output, audits = [], []
        for row in reference.itertuples(index=False):
            mid, original = str(row.molecule_id), str(row.smiles).split(";")
            active = mid in processed
            # The exact original pre-overlay tail is essential for paired equivalence.
            rebuilt, _ = rerank_row(
                scope, mid, original, window=40, ice_fallback=False, processed=active
            )
            assert ";".join(rebuilt) == controls[mid], f"E004 control reconstruction mismatch: {mid}"
            result, audit = rerank_row(
                scope, mid, original, window=window, ice_fallback=ice_fallback, processed=active
            )
            assert 1 <= len(result) <= 25
            if active:
                identities = [scope["metric_identity"](s) for s in result]
                assert None not in identities and len(identities) == len(set(identities)), mid
            else:
                assert ";".join(result) == controls[mid], f"Placeholder changed: {mid}"
            old = controls[mid].split(";")
            audit.update(changed=result != old, changed_top1=result[:1] != old[:1])
            audits.append(audit)
            output.append((mid, ";".join(result)))
        frame = pd.DataFrame(output, columns=["molecule_id", "smiles"])
        assert list(frame.molecule_id) == list(control.molecule_id)
        frame.to_csv(path, index=False)
        report = {
            "experiment": experiment,
            "window": window,
            "ice_fallback": bool(ice_fallback),
            "source_experiment": "E004",
            "base_report_file": "e004_verification.json",
            "control_file": "submission_e004_control.csv",
            "control_sha256": control_sha,
            "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "processed_ids": sorted(processed),
            "output_rows": len(frame),
            "changed_rows": sum(a["changed"] for a in audits),
            "changed_top1": sum(a["changed_top1"] for a in audits),
            "fallback_rows": sum(a["branch"] == "ice_fallback" for a in audits),
            "fallback_finite_rows": sum(
                a["branch"] == "ice_fallback" and a["ice_finite_count"] > 0 for a in audits
            ),
            "fallback_changed_rows": sum(a["fallback_changed"] for a in audits),
            "fallback_changed_top1": sum(a["fallback_changed_top1"] for a in audits),
            "same_run_control_reconstructed": True,
            "provenance": provenance,
            "limitations": [
                "Visible smoke is not independent validation.",
                "An inactive fallback in visible smoke says nothing about hidden benefit.",
            ],
        }
        pd.DataFrame(audits).to_csv(root / "experiment_audit_factorial.csv", index=False)
        (root / "experiment_audit_factorial.json").write_text(
            json.dumps(dict(report, rows=audits), indent=2, allow_nan=False) + "\n"
        )
        (root / "experiment_verification.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print("FACTORIAL_VERIFICATION", json.dumps(report))
        return report

    return run_overlay(scope, **{'experiment': 'E006', 'window': 40, 'ice_fallback': True, 'provenance': {'base_notebook_sha256': '1f0f9b3026a93ddb5ad6fa6993c8ac0a4bfaddc54594eb1dd6dda7a20d8ef0d6', 'base_source_code_sha256': 'eee496a371fce179b4aeda1c5082a58212c724eaf1ca4ba0c943bbe8d7196ef2', 'helper_source_sha256': 'a027a69ddda2c2e14639e3ec5196d5f1eb2ab57cba10ef929723148f31fc8f34', 'source_script_version_id': 356740494, 'ice_policy_source_script_version_id': 356785434, 'ice_policy_source': 'samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438/1'}})

FACTORIAL_REPORT = _factorial_entry(globals())
del _factorial_entry


    


# %% [code]

# ---- E2R2: engine-2 rank-2 insert after the final E006 overlay (MODE 'rank2' only) ----
# Idea: antriksharya, public notebook "CASMI26 | LB 0.443 | v7c + engine-2 rank-2 insert" (Apache 2.0); reimplemented.
# Change: candidates are keyed with this notebook's metric_identity (stereo stripped, tautomer-canonical InChIKey-14
# + heavy-atom composition) instead of chem.score_key, which keeps stereo.


def e2r2_make_key(identity, placeholder='CCO'):
    """SMILES -> metric identity, or None for a non-string, a blank string, the placeholder (or any SMILES with the
    placeholder's identity) and an unparseable SMILES. A token whose key is None is never inserted and never matched."""
    ph = identity(placeholder)

    def key(smiles):
        if not isinstance(smiles, str) or not smiles.strip() or smiles.strip() == placeholder:
            return None
        k = identity(smiles)
        return None if k is None or k == ph else k

    return key


def e2r2_row(toks, eng_list, key, limit=25, placeholder='CCO'):
    """Put engine 2's top candidate x2 at rank 2 of one final list (best first). Pure function.

    x2 is the first entry of `eng_list` that is a non-blank string other than the placeholder. It is used only when
    key(x2) is not None; if that entry is unusable the row is left alone (later entries are not tried).
      empty row, or rank 1 is the placeholder  -> unchanged, 'skipped'
      no engine list                           -> unchanged, 'no_eng'
      x2 missing, unparseable or holds ';'     -> unchanged, 'invalid_x2'
      key(x2) == key(rank 1)                   -> unchanged, 'same_as_rank1'
      x2's identity already at rank 2          -> that token stays, deeper twins of it are dropped, 'already_rank2'
      otherwise                                -> every token with key(x2) is dropped, x2 goes to rank 2, the other
                                                  tokens keep their order, the list is cut to `limit`, 'inserted'
    Returns (new_toks, tag, moved_from). moved_from is 'absent' or the old 1-based rank (str) of x2's identity when
    x2 ends at rank 2, else None.
    """
    toks = list(toks)
    if not toks or not isinstance(toks[0], str) or toks[0].strip() in ('', placeholder):
        return toks, 'skipped', None
    if isinstance(eng_list, str):
        eng_list = [eng_list]
    if not eng_list:
        return toks, 'no_eng', None
    x2 = next((s.strip() for s in eng_list if isinstance(s, str) and s.strip() not in ('', placeholder)), None)
    kx = key(x2) if x2 is not None and not any(c in x2 for c in ';\r\n') else None
    if kx is None:
        return toks, 'invalid_x2', None
    keys = [key(t) for t in toks]
    if kx == keys[0]:
        return toks, 'same_as_rank1', None
    hits = [i for i, k in enumerate(keys) if k == kx]
    if hits and hits[0] == 1:
        return [t for i, t in enumerate(toks) if i == 1 or keys[i] != kx][:limit], 'already_rank2', '2'
    rest = [t for t, k in zip(toks, keys) if k != kx]
    return (rest[:1] + [x2] + rest[1:])[:limit], 'inserted', (str(hits[0] + 1) if hits else 'absent')


def _e2r2_apply(scope, base_bytes, workdir='.'):
    """Apply e2r2_row to every row of the final submission.csv, check the result, then replace the file."""
    import hashlib
    import json
    import os
    from collections import Counter
    from pathlib import Path
    import pandas as pd

    root = Path(workdir)
    sub_path, tmp_path = root / 'submission.csv', root / 'submission_e2r2.tmp.csv'
    (root / 'submission_base_e2r2.csv').write_bytes(base_bytes)
    base = pd.read_csv(root / 'submission_base_e2r2.csv', dtype=str, keep_default_na=False)
    sample = pd.read_csv(os.path.join(scope['COMP'], 'sample_submission.csv'), dtype=str, keep_default_na=False)
    key = e2r2_make_key(scope['metric_identity'])
    eng = scope['ENG']
    assert isinstance(eng, dict), type(eng)
    active = set(scope['te'].molecule_id.astype(str))
    strict = not bool(scope.get('IS_RERUN', True))

    def engine_list(mid):
        e = eng.get(mid)
        if e is None and mid.strip().lstrip('-').isdigit():
            e = eng.get(int(mid))
        if isinstance(e, dict):
            return e.get('smiles') or []
        return list(e) if isinstance(e, (list, tuple)) else []

    def n_dup(ts):
        ks = [k for k in map(key, ts) if k is not None]
        return len(ks) - len(set(ks))

    def check(frame):
        assert list(frame.columns) == ['molecule_id', 'smiles'], list(frame.columns)
        assert list(frame.molecule_id) == list(base.molecule_id)
        assert len(frame) == len(sample) and frame.molecule_id.is_unique
        assert set(frame.molecule_id) == set(sample.molecule_id)
        assert frame.smiles.notna().all() and (frame.smiles.str.strip() != '').all()
        assert frame.smiles.map(lambda v: 1 <= len(v.split(';')) <= 25).all()

    check(base)
    tags, moved, rows, examples = Counter(), Counter(), [], []
    eligible_active = tail_dropped = twins_dropped = 0
    for mid, s in zip(base.molecule_id, base.smiles):
        old = s.split(';')
        new, tag, moved_from = e2r2_row(old, engine_list(mid), key)
        tags[tag] += 1
        eligible_active += int(tag != 'skipped' and mid in active)
        if moved_from is not None:
            moved[moved_from] += 1
        if new == old:
            rows.append((mid, s))
            continue
        # changed row: rank 1 kept, x2's identity exactly once (rank 2), other tokens in order, no new duplicates
        kx = key(new[1])
        n_tw = sum(key(t) == kx for t in old)
        assert tag in ('inserted', 'already_rank2') and mid in active, (mid, tag)
        assert new[0] == old[0] and kx is not None and 2 <= len(new) <= 25 and all(t.strip() for t in new), mid
        assert sum(key(t) == kx for t in new) == 1, mid
        assert [t for t in new if key(t) != kx] == [t for t in old if key(t) != kx][:len(new) - 1], mid
        assert n_dup(new) <= n_dup(old), mid
        twins_dropped += max(0, n_tw - 1)
        tail_dropped += len(old) - n_tw + 1 - len(new)
        if len(examples) < 3:
            examples.append(dict(molecule_id=mid, tag=tag, moved_from=moved_from, before=old[:3], after=new[:3]))
        rows.append((mid, ';'.join(new)))
    result = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
    check(result)
    if strict:  # commit run: every processed molecule must reach the overlay with an engine-2 list
        assert eligible_active == len(active) and tags['no_eng'] == 0, (eligible_active, len(active), dict(tags))
    changed = int(sum(a != b for a, b in zip(result.smiles, base.smiles)))
    stats = dict(mode='rank2', key='metric_identity', is_rerun=not strict, rows=len(base),
                 eligible=len(base) - tags['skipped'], inserted=tags['inserted'],
                 moved_from={k: moved[k] for k in sorted(moved, key=lambda k: -1 if k == 'absent' else int(k))},
                 same_as_rank1=tags['same_as_rank1'], no_eng=tags['no_eng'], invalid_x2=tags['invalid_x2'],
                 already_rank2=tags['already_rank2'], skipped=tags['skipped'], changed_rows=changed,
                 tail_dropped=tail_dropped, extra_twins_dropped=twins_dropped,
                 base_sha256=hashlib.sha256(base_bytes).hexdigest())
    new_bytes = base_bytes
    if changed:
        result.to_csv(tmp_path, index=False)
        back = pd.read_csv(tmp_path, dtype=str, keep_default_na=False)
        check(back)
        assert list(back.smiles) == list(result.smiles)
        new_bytes = tmp_path.read_bytes()
    stats['submission_sha256'] = hashlib.sha256(new_bytes).hexdigest()
    (root / 'e2r2_stats.json').write_text(json.dumps(dict(stats, status='applied', examples=examples), indent=1) + '\n')
    if changed:
        os.replace(tmp_path, sub_path)
    print('E2R2_APPLIED', json.dumps(stats), flush=True)
    for ex in examples:
        print('E2R2_EXAMPLE', json.dumps(ex), flush=True)
    return stats


import json as _e2r2_json
import os as _e2r2_os
import sys as _e2r2_sys
import traceback as _e2r2_tb
_E2R2_STRICT = not bool(globals().get('IS_RERUN', True))  # commit run: fail loudly; scoring rerun: keep the E006 file
_e2r2_base_bytes = None
E2R2_STATS = None
try:
    with open('submission.csv', 'rb') as _e2r2_f:  # the final E006 file (cwd is /kaggle/working)
        _e2r2_base_bytes = _e2r2_f.read()
    print('E2R2 reads', _e2r2_os.path.abspath('submission.csv'), len(_e2r2_base_bytes), 'bytes', flush=True)
    E2R2_STATS = _e2r2_apply(globals(), _e2r2_base_bytes)
except Exception as _e2r2_exc:
    if _E2R2_STRICT:
        raise
    _e2r2_state = 'not read'
    if _e2r2_base_bytes is not None:
        try:
            _e2r2_now = None
            if _e2r2_os.path.isfile('submission.csv'):
                with open('submission.csv', 'rb') as _e2r2_f:
                    _e2r2_now = _e2r2_f.read()
            if _e2r2_now != _e2r2_base_bytes:
                with open('submission.csv', 'wb') as _e2r2_f:
                    _e2r2_f.write(_e2r2_base_bytes)
            _e2r2_state = 'untouched' if _e2r2_now == _e2r2_base_bytes else 'restored from the E006 bytes'
        except Exception as _e2r2_exc2:
            _e2r2_state = f'RESTORE FAILED {_e2r2_exc2!r}'
    try:
        if _e2r2_os.path.exists('submission_e2r2.tmp.csv'):
            _e2r2_os.remove('submission_e2r2.tmp.csv')
        with open('e2r2_stats.json', 'w') as _e2r2_f:
            _e2r2_f.write(_e2r2_json.dumps(dict(status='failed', submission=_e2r2_state,
                                                error=f'{type(_e2r2_exc).__name__}: {_e2r2_exc}'[:500])) + '\n')
    except Exception:
        pass
    try:
        print('!' * 100, flush=True)
        print(f'E2R2_FAILED {type(_e2r2_exc).__name__}: {_e2r2_exc} | submission.csv {_e2r2_state} '
              '| the E006 list is the submission', flush=True)
        _e2r2_tb.print_exc(file=_e2r2_sys.stdout)
        print('!' * 100, flush=True)
    except Exception:
        print('E2R2_FAILED (error report failed) | submission.csv', _e2r2_state, flush=True)


    


