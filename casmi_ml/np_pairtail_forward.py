"""Fail-closed applicability checks for future forward spectrum engines.

This module executes no model. A missing/unsupported measurement is explicit
missing evidence, not a numeric score. Unverified public model provenance is
never promoted to strict identity-isolated evaluation.
"""
import math

from rdkit import Chem


def forward_applicability(measurement, smiles, manifest):
    reasons = []
    for field in ('weight_usage_verified', 'training_identity_audit_passed',
                  'energy_units_verified', 'runtime_verified'):
        if manifest.get(field) is not True:
            reasons.append(field+' is not verified')
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or not mol.GetNumAtoms():
        reasons.append('invalid candidate structure')
    elif mol.GetNumHeavyAtoms() > manifest.get('maximum_heavy_atoms', 0):
        reasons.append('candidate exceeds atom domain')
    for field, supported in (('adduct', 'supported_adducts'),
                             ('ionization_mode', 'supported_polarities')):
        if measurement.get(field) not in manifest.get(supported, []):
            reasons.append('unsupported '+field)
    instrument = measurement.get('instrument_type')
    instrument = manifest.get('instrument_aliases', {}).get(instrument, instrument)
    if instrument not in manifest.get('supported_instruments', []):
        reasons.append('unsupported instrument family')
    value = measurement.get('precursor_mz')
    bounds = manifest.get('precursor_range')
    if (not isinstance(value, (int, float)) or not math.isfinite(value)
            or not bounds or not bounds[0] <= value <= bounds[1]):
        reasons.append('unsupported precursor range')
    # Deliberately no implicit eV/NCE conversion, including for instrument aliases.
    energy = measurement.get('collision_energy')
    energy_bounds = manifest.get('collision_energy_range')
    if measurement.get('collision_energy_unit') != manifest.get('collision_energy_unit'):
        reasons.append('collision energy units differ')
    if (not isinstance(energy, (int, float)) or not math.isfinite(energy)
            or not energy_bounds or not energy_bounds[0] <= energy <= energy_bounds[1]):
        reasons.append('collision energy missing or outside domain')
    return {'eligible':not reasons, 'reasons':reasons, 'score':None,
            'model_executed':False}


def forward_coverage_gate(eligible_queries, completed_queries, cascade=False):
    eligible, completed = set(eligible_queries), set(completed_queries)
    if not completed <= eligible:
        raise ValueError('Forward outputs contain ineligible queries')
    threshold = .9 if cascade else .8
    coverage = len(completed)/len(eligible) if eligible else None
    return {'eligible_molecules':len(eligible), 'completed_molecules':len(completed),
            'coverage':coverage, 'minimum_coverage':threshold,
            'enabled_in_confirmatory':bool(eligible) and coverage >= threshold,
            'missing_queries':sorted(eligible-completed)}
