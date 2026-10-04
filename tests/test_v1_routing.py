import numpy as np
import pytest

from casmi_ml.hybrid_chemistry import HybridChemistry
from casmi_ml.chemical_priors import load_rules
from casmi_ml.v1_routing import CONFIGS, RoutingHybrid, acceptance, choose, protected
from test_hybrid_chemistry import DICTIONARY, fixture


def test_b0_matches_frozen_v1_on_protected_and_rule_active_queries():
    _, records, coconut, catalog, test = fixture()
    old = HybridChemistry(records, coconut, catalog, load_rules(DICTIONARY))
    new = RoutingHybrid(records, coconut, catalog, load_rules(DICTIONARY))
    for _, group in test.groupby('molecule_id'):
        expected = old.variants(group, {'v1': (True, True)})['v1'][0]
        outputs, audit = new.controls(group)
        assert outputs['B0'] == expected
        assert np.isfinite(audit['seconds'])


def test_single_policy_path_matches_shared_research_path():
    _, records, coconut, catalog, test = fixture()
    new = RoutingHybrid(records, coconut, catalog, load_rules(DICTIONARY))
    for _, group in test.groupby('molecule_id'):
        outputs, _ = new.controls(group)
        for config in CONFIGS:
            selected, _ = new.controls(group, [config])
            assert selected[config['name']] == outputs[config['name']]


def test_margin_guard_opens_ambiguous_hits_and_keeps_clear_hits():
    config = CONFIGS[-1]
    assert not protected(.9, .01, config)
    assert protected(.9, .2, config)


def rows(baseline, candidate):
    return [{'rr': {**{c['name']: a for c in CONFIGS}, 'guard075': b}}
            for a, b in zip(baseline, candidate)]


def test_selection_rejects_known_damage_and_breaks_ties_to_b0():
    unknown = rows([0.] * 40, [.2] * 40)
    assert choose(unknown, rows([1.] * 40, [.5] * 40))['name'] == 'B0'
    assert choose(rows([0.] * 40, [0.] * 40), rows([1.] * 40, [1.] * 40))['name'] == 'B0'
    assert choose(unknown, rows([1.] * 40, [1.] * 40))['name'] == 'guard075'


def test_acceptance_requires_improvement_and_known_noninferiority():
    config = CONFIGS[2]
    ok = acceptance(rows([0.] * 100, [.02] * 100), rows([1.] * 100, [1.] * 100), config)
    assert ok['accepted'] and ok['release_config'] == config
    failed = acceptance(rows([0.] * 100, [.02] * 100), rows([1.] * 100, [.99] * 100), config)
    assert not failed['accepted'] and failed['release_config']['name'] == 'B0'
