"""Bounded V1 research run, deliberately separate from competition inference."""
import gc
import hashlib
import json
import math
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from baseline import ADDUCT_MASS, load_candidates, vectorize
from casmi_ml.chemical_priors import load_rules
from casmi_ml.hybrid_chemistry import COCONUT_COLUMNS, HybridChemistry
from casmi_ml.v1_routing import CONFIGS, RoutingHybrid, acceptance, choose, metrics

ENUMERATOR = rdMolStandardize.TautomerEnumerator()
NAMESPACE = 'casmi-v1-routing-20261003-v1:'


@lru_cache(maxsize=500000)
def identity(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or not mol.GetNumAtoms():
        return None
    return Chem.MolToInchiKey(ENUMERATOR.Canonicalize(mol))[:14]


def canonical_chunk(item):
    smiles, deadline = item
    output = []
    for smi in smiles:
        if time.monotonic() > deadline:
            raise TimeoutError('Structure preparation exceeded total research budget')
        try:
            output.append((smi, identity(smi)))
        except (RuntimeError, ValueError):
            output.append((smi, None))
    return output


def digest(namespace, value):
    return int.from_bytes(hashlib.sha256((namespace + value).encode()).digest()[:8], 'big')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def signature(vector):
    return tuple(sorted((int(k), float(v)) for k, v in vector.items()))


class Budget:
    def __init__(self, root, seconds=7200):
        self.root, self.started, self.seconds = Path(root), time.monotonic(), seconds
        self.events = []

    def checkpoint(self, stage, **details):
        elapsed = time.monotonic() - self.started
        row = {'stage': stage, 'elapsed_seconds': elapsed,
               'maxrss_native': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **details}
        self.events.append(row)
        write(self.root / 'progress.json', self.events)
        print(json.dumps(row), flush=True)
        if elapsed > self.seconds:
            raise TimeoutError('Research exceeded frozen total budget; no release authorized')


def prepare(train_path, root, prior, budget, dev_count=500, acceptance_count=1000):
    meta = pd.read_parquet(train_path, columns=['inchikey14', 'normalized_smiles', 'ingest_lib'])
    raw = meta.drop_duplicates('inchikey14')[['inchikey14', 'normalized_smiles']].copy()
    smiles = sorted(set(meta.normalized_smiles.dropna()))
    mapping = {}
    chunks = [(smiles[i:i+128], budget.started + budget.seconds) for i in range(0, len(smiles), 128)]
    with ProcessPoolExecutor(max_workers=4) as executor:
        for number, result in enumerate(executor.map(canonical_chunk, chunks), 1):
            mapping.update(result)
            if number % 80 == 0:
                budget.checkpoint('canonical_structures', completed=len(mapping), total=len(smiles))
    write(root / 'structure_identity_map.json', mapping)
    raw['identity'] = raw.normalized_smiles.map(mapping)
    raw = raw[raw.identity.notna()]
    diagnostics = set(meta.loc[meta.ingest_lib.eq('enveda-np-examples'), 'normalized_smiles'].map(mapping))
    blocked = set(prior['identities']) | diagnostics
    # Conservatively remove all old raw train/holdout buckets, and the first
    # 2,000 old development keys; no old neural weights are used in this run.
    raw['old_hash'] = raw.inchikey14.map(lambda k: digest('42:', k))
    old_development = raw[(raw.old_hash % 10000 >= 8000) & (raw.old_hash % 10000 < 9000)]
    old_dev_keys = set(old_development.sort_values('old_hash').head(2000).inchikey14)
    blocked.update(raw.loc[(raw.old_hash % 10000 < 8000) | (raw.old_hash % 10000 >= 9000)
                          | raw.inchikey14.isin(old_dev_keys), 'identity'])
    # Recover v2's pre-quality dev/acceptance identity selection, conservatively
    # excluding the whole selected set, not only retained acceptance cases.
    previous = raw[['identity']].drop_duplicates().copy()
    previous['order'] = previous.identity.map(lambda k: digest('casmi-conditional-fingerprint-gan-v1:20261002:', k))
    previous = previous[~previous.identity.isin(prior['v2_excluded_identities'])]
    for lo, hi in [(8000, 9000), (9000, 10000)]:
        blocked.update(previous[(previous.order % 10000 >= lo) & (previous.order % 10000 < hi)]
                       .sort_values('order').head(1000).identity)
    eligible = raw[~raw.identity.isin(blocked)].copy()
    eligible['order'] = eligible.identity.map(lambda k: digest(NAMESPACE, k))
    chosen = eligible.sort_values(['order', 'identity']).drop_duplicates('identity')
    chosen = chosen.head(dev_count + acceptance_count)
    if len(chosen) != dev_count + acceptance_count:
        raise ValueError('Insufficient untouched candidate identities')
    selected = set(chosen.identity)
    best = {}
    columns = ['inchikey14', 'normalized_smiles', 'ingest_lib', 'precursor_mz', 'adduct',
               'ionization_mode', 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_error_ppm']
    for number, batch in enumerate(pq.ParquetFile(train_path).iter_batches(batch_size=16384, columns=columns)):
        frame = batch.to_pandas()
        frame['identity'] = frame.normalized_smiles.map(mapping)
        for offset, row in frame[frame.identity.isin(selected)].iterrows():
            adduct = ADDUCT_MASS.get(row.adduct)
            error = row.precursor_error_ppm
            if (adduct is None or not np.isfinite(row.precursor_mz)
                    or not 157 <= row.precursor_mz - adduct <= 1159
                    or (error is not None and np.isfinite(error) and abs(error) > 30)
                    or not vectorize(row.ms2_mzs, row.ms2_normalized_intensities)):
                continue
            rank = digest(NAMESPACE + 'spectrum:', f'{number}:{offset}')
            group = best.setdefault(row.identity, [])
            group.append((rank, row.to_dict()))
            group.sort(key=lambda item: item[0])
            del group[2:]
        if number % 25 == 0:
            budget.checkpoint('query_sampling', batches=number, retained_identities=len(best))
    # Quality omissions stay omissions. No replacements chosen using scores.
    dev_ids = set(chosen.head(dev_count).identity)
    frames = {}
    for name, is_dev in [('development', True), ('acceptance', False)]:
        rows = [row for key, records in best.items() if (key in dev_ids) == is_dev for _, row in records]
        frames[name] = pd.DataFrame(rows).sort_values('identity').reset_index(drop=True)
        frames[name].to_parquet(root / f'{name}.parquet', index=False)
        if frames[name].identity.nunique() < (dev_count if is_dev else acceptance_count) * .6:
            raise ValueError('Too many quality/mass omissions; frozen cohort is insufficient')
    manifest = {'namespace': NAMESPACE, 'selected': chosen[['inchikey14', 'identity']].to_dict('records'),
                'development_identities': sorted(dev_ids), 'excluded_identities': sorted(blocked),
                'selected_actual': {name: int(frame.identity.nunique()) for name, frame in frames.items()},
                'rdkit_version': rdBase.rdkitVersion, 'prior_manifest': prior['scope'],
                'no_neural_models_used': True, 'maximum_spectra_per_identity': 2,
                'selection_uses_no_ranking_results': True}
    write(root / 'cohort_manifest.json', manifest)
    del meta
    gc.collect()
    return frames, mapping


def evaluate(frame, records, coconut, catalog, rules, mapping, budget, label, configs=CONFIGS):
    engine = RoutingHybrid(records, coconut, catalog, rules)
    old = HybridChemistry.__new__(HybridChemistry)
    old.__dict__ = engine.__dict__
    rows = []
    for number, (truth, group) in enumerate(frame.groupby('identity', sort=True), 1):
        if number <= 50:
            expected = old.variants(group, {'v1': (True, True)})['v1'][0]
        outputs, audit = engine.controls(group, configs)
        if number <= 50 and outputs['B0'] != expected:
            raise AssertionError('Frozen v1 control mismatch')
        rr, rankings = {}, {}
        for name, pairs in outputs.items():
            ids = [mapping[smi] if smi in mapping else identity(smi) for _, smi in pairs]
            rr[name] = 1. / (ids.index(truth) + 1) if truth in ids else 0.
            rankings[name] = [{'raw_key': key, 'smiles': smi, 'identity': ident}
                              for (key, smi), ident in zip(pairs, ids)]
        rows.append({'identity': truth, 'raw_key': str(group.iloc[0].inchikey14),
                     'rr': rr, 'rankings': rankings, **audit})
        if number in (10, 50) or number % 100 == 0:
            budget.checkpoint(label, completed=number, total=frame.identity.nunique())
            if number == 50 and sum(r['seconds'] for r in rows) / 50 > 6:
                raise TimeoutError('Small-batch ranking exceeds six seconds/query; optimize before full run')
    write(budget.root / f'{label}_cases.json', rows)
    write(budget.root / f'{label}_metrics.json', {c['name']: metrics(rows, c['name']) for c in configs})
    return rows


def run(train_path, coconut_path, catalog_path, dictionary_path, root, prior, protocol):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    write(root / 'protocol.json', protocol)
    budget = Budget(root, protocol['research_budget_seconds'])
    frames, mapping = prepare(train_path, root, prior, budget,
                              protocol['development_selected'], protocol['acceptance_selected'])
    query = pd.concat(list(frames.values()), ignore_index=True)
    masses = query.precursor_mz.to_numpy() - query.adduct.map(ADDUCT_MASS).to_numpy()
    records, _ = load_candidates(train_path, masses[np.isfinite(masses)])
    def official(smi):
        return mapping[smi] if smi in mapping else identity(smi)
    # Copy exclusion applies to both conditions; identity exclusion only unknown.
    vectors = {signature(vectorize(r.ms2_mzs, r.ms2_normalized_intensities)) for r in query.itertuples()}
    records = [r for r in records if signature(r[3]) not in vectors]
    excluded = set(query.identity)
    unknown_records = [r for r in records if official(r[2]) not in excluded]
    coconut = pd.read_parquet(coconut_path, columns=COCONUT_COLUMNS)
    catalog = pd.read_parquet(catalog_path)
    rules = load_rules(dictionary_path)
    budget.checkpoint('reference_ready', known_spectra=len(records), unknown_spectra=len(unknown_records))
    dev_u = evaluate(frames['development'], unknown_records, coconut, catalog, rules, mapping, budget, 'dev_unknown')
    remaining = {official(r[2]) for r in records}
    dev_known = frames['development'][frames['development'].identity.isin(remaining)]
    dev_k = evaluate(dev_known, records, coconut, catalog, rules, mapping, budget, 'dev_known')
    if not len(dev_k):
        raise ValueError('No eligible known development queries')
    selected = choose(dev_u, dev_k)
    write(root / 'selection.json', {'selected': selected, 'frozen_before_acceptance': True,
                                  'protocol_sha256': hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest()})
    budget.checkpoint('selection_frozen', selected=selected)
    # Acceptance only computes the baseline and already frozen winning policy.
    confirmatory = [CONFIGS[0]] if selected['name'] == 'B0' else [CONFIGS[0], selected]
    acc_u = evaluate(frames['acceptance'], unknown_records, coconut, catalog, rules, mapping, budget, 'acceptance_unknown', confirmatory)
    known = frames['acceptance'][frames['acceptance'].identity.isin(remaining)]
    acc_k = evaluate(known, records, coconut, catalog, rules, mapping, budget, 'acceptance_known', confirmatory)
    if not len(acc_k):
        raise ValueError('No eligible known acceptance queries')
    report = acceptance(acc_u, acc_k, selected)
    report['limitations'] = [prior['scope'], 'Proxy cohorts are not hidden test labels.']
    write(root / 'acceptance_report.json', report)
    budget.checkpoint('research_complete', accepted=report['accepted'])
    return report
