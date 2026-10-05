"""Independently check retained M2 binaries, rankings and paired intervals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SEEDS = (20261004, 20261005, 20261006)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def paired(values):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEEDS[0])
    samples = np.concatenate([values[rng.integers(len(values), size=(100, len(values)))].mean(1)
                              for _ in range(100)])
    return {'difference':float(values.mean()), 'ci95':np.quantile(samples, [.025, .975]).tolist(),
            'molecules':len(values), 'repeats':len(samples), 'seed':SEEDS[0]}


def audit(root, expected):
    root = Path(root)
    candidates = [p.parent for p in root.rglob('M2_training_summary.json')]
    assert len(candidates) == 1, 'One retained M2 result required'
    model_root = candidates[0]
    status = read(model_root/'run_status.json')
    assert status['stage'] == 'M2_complete', status
    provenance = read(model_root/'data_manifest.json')
    assert provenance['embedded_source_sha256'] == expected['embedded_source_sha256']
    assert not provenance['acceptance_file_read']
    code = root/'chemistry_code'
    for name, checksum in expected['source_files'].items():
        assert digest(code/name) == checksum, name
    summary = read(model_root/'M2_training_summary.json')
    assert summary['bce_gate_pass'] and not summary['acceptance_opened']
    assert summary['ensemble']['bce'] <= .9*summary['constant_train_frequency']['bce']
    statistics = read(model_root/'train_fitted_statistics.json')
    selection = []
    for seed in SEEDS:
        history = read(model_root/f'seed_{seed}_history.json')
        assert len(history) >= 6
        assert [r['epoch'] for r in history] == list(range(1, len(history)+1))
        assert all(r['identity_examples'] == 119977 and r['optimizer_updates'] == 1875*r['epoch'] for r in history)
        checkpoint = torch.load(model_root/f'fpnet_seed_{seed}.pt', map_location='cpu', weights_only=True)
        assert checkpoint['seed'] == seed and checkpoint['bits'] == statistics['bits']
        assert all(torch.isfinite(v).all().item() for v in checkpoint['state_dict'].values())
        chosen = history[checkpoint['selected_epoch']-1]
        assert checkpoint['optimizer_updates'] == chosen['optimizer_updates']
        assert abs(checkpoint['development_bce']-chosen['development']['bce']) < 1e-6
        assert abs(checkpoint['development_bce']-min(r['development']['bce'] for r in history)) < 1e-6
        for view in ('single','fused'):
            logits = np.load(model_root/f'seed_{seed}_{view}_logits.npy')
            assert logits.shape == (1930, len(statistics['bits'])) and np.isfinite(logits).all()
        selection.append({'seed':seed, 'completed_epochs':len(history),
                          'selected_epoch':checkpoint['selected_epoch'],
                          'selected_updates':checkpoint['optimizer_updates'],
                          'selected_bce':checkpoint['development_bce'],
                          'last_completed_updates':history[-1]['optimizer_updates'],
                          'checkpoint_sha256':digest(model_root/f'fpnet_seed_{seed}.pt')})
        del checkpoint
    for view in ('single','fused'):
        independently_averaged = np.mean([np.load(model_root/f'seed_{s}_{view}_logits.npy') for s in SEEDS], axis=0)
        np.testing.assert_array_equal(independently_averaged, np.load(model_root/f'ensemble_{view}_logits.npy'))
    cases = read(model_root/'M2_development_candidate_cases.json')
    identities = read(model_root/'development_identity_order.json')
    assert len(cases) == 1930 and [c['identity'] for c in cases] == identities
    assert len(set(identities)) == len(identities)
    rng = np.random.default_rng(SEEDS[0])
    for case in cases:
        ids = case['candidate_identities']
        assert len(ids) == case['mass_window_candidates'] and len(set(ids)) == len(ids)
        assert case['truth_in_mass_window'] == (case['identity'] in ids)
        np.testing.assert_array_equal(case['random_order'], rng.permutation(len(ids)))
        for view in ('single','fused','random'):
            ranking = case[view+'_order']
            assert sorted(ranking) == list(range(len(ids)))
            top = [ids[i] for i in ranking[:25]]
            rr = 1/(top.index(case['identity'])+1) if case['identity'] in top else 0.
            assert rr == case[view+'_rr']
    covered = [c for c in cases if c['truth_in_mass_window']]
    assert covered
    delta = float(np.mean([c['fused_rr']-c['single_rr'] for c in covered]))
    view = 'single' if delta < -.001 else 'fused'
    comparisons = {'conditional_fused_vs_random':paired([c['fused_rr']-c['random_rr'] for c in covered]),
                   'selected_vs_random':paired([c[view+'_rr']-c['random_rr'] for c in covered])}
    gate = read(model_root/'M2_development_candidate_gate.json')
    assert gate['selected_view'] == view and gate['conditional_truth_present'] == len(covered)
    for key, result in comparisons.items():
        assert result == gate[key], (key, result, gate[key])
    assert gate['passed'] and comparisons['selected_vs_random']['ci95'][0] > 0
    return {'phase':'independent_retained_M2_output_check', 'passed':True,
            'source_sha256':provenance['embedded_source_sha256'], 'source_files_exact':len(expected['source_files']),
            'selected_models':selection, 'ensemble_logits_exact_average':True,
            'all_selected_weights_and_logits_finite':True, 'candidate_rankings_and_random_orders_exact':True,
            'queries':len(cases), 'conditional_truth_present':len(covered), 'selected_view':view,
            'paired_comparisons':comparisons, 'bce_gate':{'ensemble':summary['ensemble']['bce'],
                'constant':summary['constant_train_frequency']['bce']},
            'cumulative_seconds':status['cumulative_seconds'], 'acceptance_opened':False,
            'limitations':'No reconstruction of raw training rows or full candidate pool; hashes bind those immutable inputs.',
            'competition_submitted':False, 'public_score':None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_directory', type=Path)
    parser.add_argument('expected_build_manifest', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    result = audit(args.output_directory, read(args.expected_build_manifest))
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
