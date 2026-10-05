"""Runtime-only candidate features and groupwise ranker for NP-PairTail.

Truths enter labels/evaluation only. Source aliases are retained through mass
selection; molecular identifiers are never model features. No acceptance reader
is provided by this module.
"""
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from baseline import ADDUCT_MASS
from casmi_ml.chemical_priors import candidate_scores, extract_evidence
from casmi_ml.data import write_json
from casmi_ml.np_pairtail_structures import canonical_target, sha256
from casmi_ml.np_pairtail_fusion import local_module_gate

SOURCES = ('train/library', 'COCONUT', 'LOTUS', 'ChEBI', 'LIPID_MAPS')
FEATURES = ('library_score', 'library_rank_fraction', 'library_present',
    'analog_tanimoto', 'analog_weighted', 'mass_error_ppm',
    'library_spectra', 'library_multispectrum_mean', 'library_shifted_similarity',
    'library_adduct_match', 'library_instrument_match', 'instrument_orbitrap',
    'instrument_qtof', 'instrument_timstof', 'instrument_missing',
    'fp_dot', 'fp_length_normalized', 'fp_z', 'fp_rank_fraction', 'fp_on_bits',
    'fragment_one_coverage', 'fragment_two_coverage', 'neutral_loss_coverage',
    'fragment_one_missing', 'fragment_two_missing', 'chemistry_support', 'query_spectra', 'positive_fraction',
    'same_adduct', 'adduct_missing_fraction', 'ce_missing_fraction', 'ce_mean_ev',
    'precursor_mass', 'source_count', 'np_source', 'fp_library_agreement',
    'fp_analog_agreement', 'source_mass_interaction') + tuple('source_'+s for s in SOURCES)


def feature_matrix(frame):
    if set(frame.columns) != set(FEATURES):
        raise ValueError('Ranker accepts exactly the frozen runtime feature allowlist')
    value = frame.loc[:, FEATURES].to_numpy(dtype=np.float32)
    if np.isinf(value).any():
        raise ValueError('Infinite ranker feature')
    # NaN represents an explicitly unavailable approximate fragment or metadata.
    return value


class StructureIndex:
    def __init__(self, pool):
        required = {'identity', 'normalized_smiles', 'mass', 'sources'}
        if not required <= set(pool):
            raise ValueError('Canonical source-preserving pool required')
        if pool.identity.isna().any() or (~np.isfinite(pool.mass)).any() or pool.mass.le(0).any():
            raise ValueError('Invalid structure-pool row')
        self.pool = pool.sort_values(['mass', 'identity', 'normalized_smiles'], kind='stable').reset_index(drop=True)
        self.masses = self.pool.mass.to_numpy(float)
        self.sources = {}
        for row in self.pool.itertuples():
            self.sources.setdefault(row.identity, set()).update(str(row.sources).split(';'))
        self.fingerprints = {}

    def query(self, center):
        if not math.isfinite(center) or center <= 0:
            raise ValueError('Supported neutral query mass required')
        delta = max(center*35e-6, .006)
        rows = self.pool.iloc[np.searchsorted(self.masses, center-delta):
                              np.searchsorted(self.masses, center+delta, side='right')].copy()
        rows['distance'] = abs(rows.mass-center)
        rows = rows.sort_values(['distance', 'identity', 'normalized_smiles'], kind='stable')
        rows = rows.drop_duplicates('identity').drop(columns='distance')
        rows['sources'] = rows.identity.map(lambda key: ';'.join(sorted(self.sources[key])))
        return rows.sort_values('identity', kind='stable').reset_index(drop=True)

    def fps(self, rows, bits):
        result = []
        for row in rows.itertuples():
            if row.identity not in self.fingerprints:
                canonical, vector = canonical_target(row.normalized_smiles, row.identity)
                self.fingerprints[row.identity] = (canonical, vector)
            result.append(self.fingerprints[row.identity][1][bits])
        return np.asarray(result, dtype=np.float32).reshape(len(rows), len(bits))


def fragment_masses(smiles):
    """Neutral connected components after one/two bond cuts, without H guesses.

    These are approximate explanation features, not predicted fragmentation.
    For >24 bonds the two-cut feature is explicitly missing; one-cut remains
    complete. No candidate is excluded on this approximation's applicability.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Fragment feature requires a valid molecular graph')
    if len(Chem.GetMolFrags(mol)) != 1:
        return None,None,float(rdMolDescriptors.CalcExactMolWt(mol))
    # Preserve original explicit/implicit H and isotopes; no rehydrogenation of
    # cleavage products. Atomic isotope masses include original hydrogen counts.
    periodic = Chem.GetPeriodicTable()
    masses = np.array([(periodic.GetMassForIsotope(a.GetAtomicNum(), a.GetIsotope())
        if a.GetIsotope() else periodic.GetMostCommonIsotopeMass(a.GetAtomicNum()))
        + a.GetTotalNumHs()*periodic.GetMostCommonIsotopeMass(1) for a in mol.GetAtoms()])
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    adjacency = [[] for _ in masses]
    for number, (a, b) in enumerate(bonds):
        adjacency[a].append((b, number)); adjacency[b].append((a, number))
    def cut(ids):
        blocked, seen, values = set(ids), set(), []
        for a in range(len(masses)):
            if a in seen: continue
            stack, component = [a], []
            seen.add(a)
            while stack:
                u = stack.pop(); component.append(u)
                for v, bond in adjacency[u]:
                    if bond not in blocked and v not in seen:
                        seen.add(v); stack.append(v)
            if len(component) != len(masses):
                values.append(float(masses[component].sum()))
        return values
    one = sorted({round(v, 6) for i in range(len(bonds)) for v in cut((i,))})
    two = None if len(bonds) > 24 else sorted({round(v, 6)
        for ids in itertools.combinations(range(len(bonds)), 2) for v in cut(ids)})
    return np.asarray(one), None if two is None else np.asarray(two), float(masses.sum())


def peak_coverage(values, masses, intensity):
    if not len(masses) or not len(values): return 0.
    distance = np.min(abs(np.asarray(masses)[:, None]-np.asarray(values)[None, :]), axis=1)
    match = distance <= np.maximum(np.asarray(masses)*10e-6, .002)
    return float(np.asarray(intensity)[match].sum()/max(float(np.asarray(intensity).sum()), 1e-12))


class CandidateFeatures:
    def __init__(self, index, bits, rules=()):
        self.index, self.bits, self.rules = index, np.asarray(bits), tuple(rules)
        self.fragments = {}
        self.reference_fingerprints = {}

    def build(self, group, rows, logits, library, library_metadata=None, training_token=None):
        if len(logits) != len(self.bits) or not np.isfinite(logits).all():
            raise ValueError('Fingerprint prediction does not match frozen bits')
        neutral = group.precursor_mz-group.adduct.map(ADDUCT_MASS)
        center = float(np.median(neutral))
        if not np.isfinite(neutral).all() or not math.isfinite(center) or center <= 0:
            raise ValueError('Unsupported query adduct/mass')
        fps = self.index.fps(rows, self.bits)
        dot = fps @ np.asarray(logits, np.float32)
        norm = dot / np.maximum(np.sqrt(fps.sum(1)), 1)
        z = (dot-dot.mean())/max(float(dot.std()), 1e-8) if len(dot) else dot
        rank = np.empty(len(dot), float)
        rank[np.lexsort((rows.identity.to_numpy(), -dot))] = np.arange(len(dot))/max(len(dot)-1, 1)
        lib = {key: (i, float(score)) for i, (key, _, score) in enumerate(library)}
        # The fixed training proposal uses only these two inexpensive scores.
        # Preserve full-pool normalization/ranking, but explain only rows that
        # can actually enter fitting. Evaluation/inference always explains all.
        explanation_rows = None if training_token is None else set(select_negative_rows(
            pd.DataFrame({'library_score':[lib.get(key,(len(library),0.))[1]
                for key in rows.identity], 'fp_dot':dot}), training_token).tolist())
        reference = []
        for key, smi, score in library[:25]:
            cache_key=(smi,key)
            if cache_key not in self.reference_fingerprints:
                _, fp = canonical_target(smi,key)
                self.reference_fingerprints[cache_key]=fp[self.bits]
            reference.append((self.reference_fingerprints[cache_key], score))
        if reference and len(fps):
            rf = np.array([r[0] for r in reference], np.float32)
            intersection = fps @ rf.T
            tanimoto = intersection/np.maximum(fps.sum(1)[:, None]+rf.sum(1)[None, :]-intersection, 1)
            analog = (tanimoto*(.5+.5*np.array([r[1] for r in reference])[None, :])).max(1)
            sim = tanimoto.max(1)
        else: analog = sim = np.zeros(len(rows))
        evidence = extract_evidence(group, self.rules) if self.rules else []
        chemistry, _ = candidate_scores(dict(zip(rows.identity, rows.normalized_smiles)), evidence, self.rules)
        energies = [np.asarray([] if v is None else v, float) for v in group.collision_energy_ev]
        finite_energies = np.concatenate(energies) if energies else np.array([])
        finite_energies = finite_energies[np.isfinite(finite_energies)]
        features = []
        library_metadata = library_metadata or {}
        instruments = group.instrument_type.fillna('<missing>').astype(str).str.lower()
        query_constants = {
            'query_spectra':len(group),
            'positive_fraction':float(group.ionization_mode.eq('positive').mean()),
            'same_adduct':float(group.adduct.nunique()==1),
            'adduct_missing_fraction':float(group.adduct.isna().mean()),
            'ce_missing_fraction':float(np.mean([not np.isfinite(v).any() for v in energies])),
            'ce_mean_ev':float(finite_energies.mean()) if len(finite_energies) else math.nan,
            'instrument_orbitrap':float(instruments.str.contains('orbitrap',regex=False).mean()),
            'instrument_qtof':float(instruments.str.contains('qtof',regex=False).mean()),
            'instrument_timstof':float(instruments.str.contains('timstof',regex=False).mean()),
            'instrument_missing':float(instruments.eq('<missing>').mean())}
        measurements = []
        for query in group.itertuples():
            mz, intensity = np.asarray(query.ms2_mzs, float), np.asarray(query.ms2_normalized_intensities, float)
            keep = np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0) & (mz > 0)
            measurements.append((mz[keep], intensity[keep], ADDUCT_MASS[query.adduct], query.precursor_mz))
        for i, row in enumerate(rows.itertuples()):
            smi = self.index.fingerprints[row.identity][0]
            explain = explanation_rows is None or i in explanation_rows
            if explain:
                if smi not in self.fragments: self.fragments[smi] = fragment_masses(smi)
                one, two, total = self.fragments[smi]
            else:
                one, two, total = None, None, math.nan
            one_cov, two_cov, loss_cov = [], [], []
            for mz, intensity, offset, precursor in measurements if explain else ():
                if one is not None:one_cov.append(peak_coverage(one+offset, mz, intensity))
                if two is not None: two_cov.append(peak_coverage(two+offset, mz, intensity))
                if one is not None:loss_cov.append(peak_coverage(total-one, precursor-mz, intensity))
            source = set(row.sources.split(';'))
            library_rank, library_score = lib.get(row.identity, (len(library), 0.))
            mass_error = abs(row.mass-center)/center*1e6
            value = {'library_score':library_score, 'library_rank_fraction':library_rank/max(len(library),1),
                'library_present':float(row.identity in lib), 'analog_tanimoto':float(sim[i]), 'analog_weighted':float(analog[i]),
                'mass_error_ppm':mass_error, 'fp_dot':float(dot[i]), 'fp_length_normalized':float(norm[i]),
                'fp_z':float(z[i]), 'fp_rank_fraction':float(rank[i]), 'fp_on_bits':float(fps[i].sum()),
                'fragment_one_coverage':max(one_cov) if one_cov else math.nan, 'fragment_two_coverage':max(two_cov) if two_cov else math.nan,
                'neutral_loss_coverage':max(loss_cov) if loss_cov else math.nan, 'fragment_one_missing':float(one is None), 'fragment_two_missing':float(two is None),
                'chemistry_support':chemistry.get(row.identity,0.),
                'precursor_mass':center, 'source_count':len(source), 'np_source':float(bool(source & {'COCONUT','LOTUS'})),
                'fp_library_agreement':float(z[i])*library_score, 'fp_analog_agreement':float(z[i])*float(analog[i]),
                'source_mass_interaction':len(source)*mass_error}
            lm = library_metadata.get(row.identity, {})
            value.update({
                'library_spectra':float(lm.get('spectra',0)),
                'library_multispectrum_mean':float(lm.get('multispectrum_mean',math.nan)),
                'library_shifted_similarity':float(lm.get('shifted_similarity',math.nan)),
                'library_adduct_match':float(lm.get('adduct_match',math.nan)),
                'library_instrument_match':float(lm.get('instrument_match',math.nan))})
            value.update(query_constants)
            value.update({'source_'+s:float(s in source) for s in SOURCES})
            features.append(value)
        output = pd.DataFrame(features, columns=FEATURES)
        output.attrs['training_explanation_rows'] = None if explanation_rows is None else sorted(explanation_rows)
        feature_matrix(output)
        return output


def select_negative_rows(features, query_token, maximum=256, seed=20261004):
    """Label-blind union of high library, high FP and deterministic random rows."""
    import hashlib
    if len(features) <= maximum: return np.arange(len(features))
    if maximum != 256: raise ValueError('Negative proposal cap differs from PRD')
    count = maximum//3
    library = np.argsort(-features.library_score.to_numpy(),kind='stable')[:count]
    fp = np.argsort(-features.fp_dot.to_numpy(),kind='stable')[:count]
    digest = hashlib.sha256((str(seed)+':'+str(query_token)).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8],'big'))
    ids = set(library) | set(fp)
    remaining = [i for i in rng.permutation(len(features)) if i not in ids]
    ids.update(remaining[:maximum-len(ids)])
    return np.array(sorted(ids),int)


def labeled_group(features, candidate_ids, truth, selection):
    # Selection is already frozen without truth, and never inserts a positive.
    selection = np.asarray(selection,int)
    if len(set(selection)) != len(selection) or np.any(selection < 0) or np.any(selection >= len(features)):
        raise ValueError('Invalid label-blind candidate selection')
    x = feature_matrix(features.iloc[selection])
    y = np.array([int(candidate_ids[i] == truth) for i in selection],np.int32)
    return x, y


def fit_boosters(x, y, groups, depth, min_leaf, destination, deadline):
    import lightgbm as lgb
    if depth not in (5,6) or min_leaf not in (50,100): raise ValueError('Ranker outside frozen PRD grid')
    if sum(groups) != len(x) or len(x) != len(y) or any(g < 1 for g in groups):
        raise ValueError('Ranker query groups do not partition labeled rows')
    if x.shape[1] != len(FEATURES) or not set(np.unique(y)) <= {0,1}:
        raise ValueError('Ranker matrix/labels malformed')
    destination = Path(destination); destination.mkdir(parents=True,exist_ok=True)
    models, manifests = [], []
    for seed in (20261004,20261005,20261006,20261007):
        if time.monotonic() >= deadline: raise TimeoutError('Ranker module/research budget exhausted')
        def budget_callback(env):
            if time.monotonic() >= deadline: raise TimeoutError('Ranker module/research budget exhausted')
        model = lgb.LGBMRanker(objective='lambdarank',n_estimators=800,learning_rate=.03,
            max_depth=depth,num_leaves=2**depth,min_child_samples=min_leaf,random_state=seed,
            n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
        model.fit(x,y,group=groups,feature_name=list(FEATURES),callbacks=[budget_callback])
        path = destination/f'booster_{seed}.txt'; model.booster_.save_model(str(path))
        models.append(model.booster_)
        manifests.append({'seed':seed,'trees':model.booster_.num_trees(),'sha256':sha256(path)})
    write_json(destination/'ranker_manifest.json',{'objective':'lambdarank','lightgbm':lgb.__version__,
        'max_depth':depth,'min_data_in_leaf':min_leaf,'n_estimators_max':800,'learning_rate':.03,
        'features':list(FEATURES),'query_groups':len(groups),'rows':len(y),'positive_rows':int(y.sum()),
        'boosters':manifests,'acceptance_opened':False})
    return models


def score_boosters(models, features):
    if len(models) != 4: raise ValueError('Incomplete four-seed ranker ensemble')
    x = feature_matrix(features)
    score = np.mean([m.predict(x,num_threads=4) for m in models],axis=0)
    if not np.isfinite(score).all(): raise FloatingPointError('Nonfinite ranking score')
    return score


def isolated_booster_fit(features, labels, groups, evaluation, depth, min_leaf, destination, deadline):
    """Fit/predict in a fresh CPU process, isolated from the GPU OpenMP runtime."""
    import os
    import subprocess
    import sys
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    x=feature_matrix(features);ex=feature_matrix(evaluation)
    np.save(destination/'train_features.npy',x);np.save(destination/'labels.npy',labels)
    np.save(destination/'evaluation_features.npy',ex)
    write_json(destination/'groups.json',list(map(int,groups)))
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('No budget remains for isolated CPU ranker')
    command=[sys.executable,'-m','casmi_ml.np_pairtail_ranker',str(destination),str(depth),str(min_leaf),str(deadline)]
    environment=dict(os.environ)
    environment['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+environment.get('PYTHONPATH','')
    subprocess.run(command,env=environment,check=True,timeout=remaining)
    score=np.load(destination/'evaluation_scores.npy')
    if score.shape!=(len(evaluation),) or not np.isfinite(score).all():
        raise ValueError('CPU ranking worker returned malformed predictions')
    return score


def _worker():
    import sys
    destination=Path(sys.argv[1]);depth,min_leaf=int(sys.argv[2]),int(sys.argv[3]);deadline=float(sys.argv[4])
    x=np.load(destination/'train_features.npy',mmap_mode='r')
    y=np.load(destination/'labels.npy',mmap_mode='r')
    groups=json.loads((destination/'groups.json').read_text())
    models=fit_boosters(x,y,groups,depth,min_leaf,destination,deadline)
    evaluation=pd.DataFrame(np.load(destination/'evaluation_features.npy',mmap_mode='r'),columns=FEATURES)
    np.save(destination/'evaluation_scores.npy',score_boosters(models,evaluation))
    print(json.dumps({'stage':'M3_CPU_ranker_complete','depth':depth,'min_leaf':min_leaf,
        'groups':len(groups),'training_rows':len(y),'evaluation_rows':len(evaluation)}),flush=True)


if __name__=='__main__':_worker()
