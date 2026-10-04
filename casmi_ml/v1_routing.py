"""V1-only routing controls; no neural training or graph generation."""
import math
import time

import numpy as np
import pandas as pd

from hybrid import blend, coconut_rank, library_rank
from casmi_ml.chemical_priors import candidate_scores, extract_evidence, rerank
from casmi_ml.hybrid_chemistry import HybridChemistry, validate_submission


CONFIGS = (
    {'name': 'B0', 'guard': .5, 'margin': None},
    {'name': 'guard065', 'guard': .65, 'margin': None},
    {'name': 'guard075', 'guard': .75, 'margin': None},
    {'name': 'guard095', 'guard': .95, 'margin': None},
    {'name': 'always_expand', 'guard': 1.01, 'margin': None},
    {'name': 'margin010', 'guard': .5, 'margin': .1},
)


def protected(confidence, margin, config):
    return (confidence >= config['guard']
            and (config['margin'] is None or margin >= config['margin']))


class RoutingHybrid(HybridChemistry):
    """Share expensive retrieval across the six predeclared routing controls."""

    def controls(self, group, configs=CONFIGS):
        started = time.monotonic()
        center, library = library_rank(group, self.records, self.matrix, self.masses, self.order)
        old = coconut_rank(center, library, self.coconut, self.old_mass, self.old_order, self.cache)
        historical = blend(library, old)
        confidence = float(library[0][2]) if library else 0.
        margin = confidence - float(library[1][2]) if len(library) > 1 else confidence
        guards = {c['name']: protected(confidence, margin, c) for c in configs}
        expanded = None
        evidence = []
        if not all(guards.values()):
            analog = coconut_rank(center, library, self.unified, self.new_mass, self.new_order, self.cache)
            pairs = blend(library, analog)
            structures = dict(pairs)
            evidence = extract_evidence(group, self.rules)
            scores, _ = candidate_scores(structures, evidence, self.rules) if evidence else ({}, {})
            keys = rerank([key for key, _ in pairs], scores, self.weight) if evidence else list(structures)
            expanded = [(key, structures[key]) for key in keys]
        outputs = {name: historical if guard else expanded for name, guard in guards.items()}
        return outputs, {'confidence': confidence, 'margin': margin, 'protected': guards,
                         'rule_matches': len(evidence), 'seconds': time.monotonic() - started}

    def predict_selected(self, test, config):
        rows, audits = [], []
        for number, (molecule_id, group) in enumerate(test.groupby('molecule_id', sort=False), 1):
            outputs, audit = self.controls(group, [config])
            rows.append({'molecule_id': molecule_id,
                         'smiles': ';'.join(smi for _, smi in outputs[config['name']])})
            audits.append({'molecule_id': molecule_id, **audit})
            if number % 25 == 0:
                print(f'V1 inference {number}/{test.molecule_id.nunique()}', flush=True)
        result = pd.DataFrame(rows)
        validate_submission(test, result)
        return result, audits


def paired(values, seed=20261003, repeats=10000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError('Empty paired comparison')
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats // 100):
        estimates.extend(values[rng.integers(len(values), size=(100, len(values)))].mean(1))
    return {'difference': float(values.mean()),
            'ci95': np.quantile(estimates, [.025, .975]).tolist(),
            'molecules': len(values), 'repeats': len(estimates), 'seed': seed}


def metrics(rows, name):
    rr = np.asarray([r['rr'][name] for r in rows], dtype=float)
    return {'molecules': len(rows), 'mrr25': float(rr.mean()) if len(rr) else None,
            'top1': float((rr == 1).mean()) if len(rr) else None,
            'top25': float((rr > 0).mean()) if len(rr) else None}


def choose(dev_unknown, dev_known):
    baseline = metrics(dev_known, 'B0')
    eligible = []
    for order, config in enumerate(CONFIGS):
        unknown, known = metrics(dev_unknown, config['name']), metrics(dev_known, config['name'])
        if (known['mrr25'] >= baseline['mrr25'] - .001
                and known['top1'] >= baseline['top1'] - .002):
            eligible.append((unknown['mrr25'], -order, config))
    return max(eligible, key=lambda item: item[:2])[2]


def acceptance(unknown, known, selected):
    name = selected['name']
    u = paired([r['rr'][name] - r['rr']['B0'] for r in unknown])
    k = paired([r['rr'][name] - r['rr']['B0'] for r in known])
    top1_delta = metrics(known, name)['top1'] - metrics(known, 'B0')['top1']
    accepted = (name != 'B0' and u['difference'] >= .001 and u['ci95'][0] > 0
                and k['ci95'][0] >= -.001 and top1_delta >= -.002)
    return {'accepted': bool(accepted), 'selected': selected, 'unknown': u,
            'known': k, 'known_top1_difference': top1_delta,
            'release_config': selected if accepted else CONFIGS[0]}
