import pytest

from casmi_ml.np_pairtail_fusion import (
    reciprocal_fusion, pairtail, local_module_gate, independent_acceptance_gate,
)


def test_missing_engine_candidates_get_only_observed_rank_evidence():
    ranking, scores = reciprocal_fusion(['A','B'], ['B','C'], 3, .6)
    assert scores['A'] == .25
    assert scores['B'] == .2+.6/4
    assert scores['C'] == .6/5
    assert ranking == ['B','A','C']
    with pytest.raises(ValueError):
        reciprocal_fusion(['A','A'], ['B'], 3, .6)


def test_top1_guard_requires_two_independent_valid_domain_challengers():
    baseline, proposal = ['A','B','C'], ['B','C','A','D']
    assert pairtail(baseline, proposal, 'tail_only') == ['A','B','C','D']
    evidence=dict(independent_tops=[('fpnet','B'),('forward','B')],margins=[.2,.3],
                  thresholds=[.1,.1],compatible=[True,True])
    assert pairtail(baseline,proposal,'consensus_unlock',**evidence) == proposal
    evidence['compatible']=[True,False]
    assert pairtail(baseline,proposal,'consensus_unlock',**evidence)[0] == 'A'
    evidence['compatible']=[True,True]
    evidence['independent_tops']=[('fpnet','B'),('fpnet','B')]
    assert pairtail(baseline,proposal,'consensus_unlock',**evidence)[0] == 'A'


def test_representative_regression_is_rejected_despite_unknown_improvement():
    previous={'unknown':{'mrr25':.012504930907038563},
              'known':{'mrr25':.7972298008302819,'top1':.7444352844187964}}
    candidate={'unknown':{'mrr25':.012977602911159753},
               'known':{'mrr25':.7730905953495276,'top1':.7320692497938994}}
    result=local_module_gate(previous,candidate)
    assert not result['passed']
    assert result['checks']['unknown_mrr']
    assert not result['checks']['known_mrr']


def test_acceptance_known_protection_and_paired_cohort_match():
    protocol={'seed':20261004,'repeats':10000,'unknown_delta_min':.001,
              'unknown_ci_lower_gt':0,'known_ci_lower_min':-.001,
              'known_top1_drop_max':.002,'known_recall25_drop_max':.002}
    before={m:{str(i):{'rr':.1,'top1':0,'recall25':1} for i in range(10)} for m in ('unknown','known')}
    after={m:{k:dict(v) for k,v in rows.items()} for m,rows in before.items()}
    for value in after['unknown'].values():value['rr']=.2
    assert independent_acceptance_gate(before,after,protocol)['passed']
    for value in after['known'].values():value['rr']=.09
    assert not independent_acceptance_gate(before,after,protocol)['passed']
    after['known'].pop('0')
    with pytest.raises(ValueError,match='cohorts'):
        independent_acceptance_gate(before,after,protocol)
