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


def test_competition_entry_reads_dynamic_ids_and_reproduces_v1(tmp_path, monkeypatch):
    import pandas as pd
    from casmi_ml import v1_routing
    _, records, coconut, catalog, test = fixture()
    expanded = pd.concat([test.assign(molecule_id=test.molecule_id + f'_{i}') for i in range(19)], ignore_index=True)
    expanded.to_parquet(tmp_path / 'test.parquet', index=False)
    coconut.to_parquet(tmp_path / 'coconut.parquet', index=False)
    catalog.to_parquet(tmp_path / 'catalog.parquet', index=False)
    monkeypatch.setattr(v1_routing, 'load_candidates', lambda path, masses: (records, {}))
    result, report = v1_routing.inference(tmp_path, tmp_path / 'coconut.parquet',
                                         tmp_path / 'catalog.parquet', DICTIONARY,
                                         tmp_path / 'submission.csv', CONFIGS[0])
    expected, _ = HybridChemistry(records, coconut, catalog, load_rules(DICTIONARY)).predict(expanded)
    assert result.equals(expected)
    assert len(result) == 38 and set(result.molecule_id) == set(expanded.molecule_id)
    assert not report['neural_training'] and not report['research_acceptance_executed']


def test_unfrozen_configuration_is_rejected_before_loading_data(tmp_path):
    from casmi_ml.v1_routing import inference
    with pytest.raises(ValueError, match='Unknown or altered'):
        inference(tmp_path, 'absent', 'absent', 'absent', tmp_path / 'out.csv',
                  {'name': 'guard075', 'guard': .8, 'margin': None})


def test_output_normalization_keeps_first_tautomer_and_never_worsens_truth_rank():
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from casmi_ml.v1_routing import unique_official_candidates
    pairs = [('first', 'CC(=O)C'), ('duplicate', 'CC(O)=C'), ('truth', 'CCO')]
    cache, enumerator = {}, rdMolStandardize.TautomerEnumerator()
    result = unique_official_candidates(pairs, cache, enumerator)
    assert result == [pairs[0], pairs[2]]
    assert cache[pairs[0][1]] == cache[pairs[1][1]]
    assert unique_official_candidates(result, cache, enumerator) == result
    assert result.index(pairs[2]) < pairs.index(pairs[2])
