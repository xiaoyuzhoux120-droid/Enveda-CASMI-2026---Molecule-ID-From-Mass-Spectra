"""Frozen PRD fusion and evaluation gates, independent of query identifiers.

Candidate identities are canonical structure keys, never visible test IDs.
Ranking functions do not accept truth labels; evaluation gates do.
"""
import math
from collections import defaultdict

from casmi_ml.v1_routing import paired


def reciprocal_fusion(primary, secondary, k, alpha):
    if k not in (3, 10) or alpha not in (.3, .6):
        raise ValueError('RRF configuration outside frozen PRD grid')
    scores = defaultdict(float)
    for ranking, weight in ((primary, 1.), (secondary, alpha)):
        if len(ranking) != len(set(ranking)):
            raise ValueError('Merge canonical aliases before engine ranking')
        for rank, identity in enumerate(ranking, 1):
            scores[identity] += weight/(k+rank)
    return sorted(scores, key=lambda identity: (-scores[identity], identity)), dict(scores)


def pairtail(baseline, proposed, strategy, independent_tops=(), margins=(), thresholds=(), compatible=()):
    """Apply a previously frozen policy to runtime evidence only.

    Consensus uses at least two distinct engine names, a common challenger,
    strictly exceeded margins and a valid deployment domain for each engine.
    Numeric thresholds must come from the separately frozen development policy.
    """
    if len(baseline) != len(set(baseline)) or len(proposed) != len(set(proposed)):
        raise ValueError('Canonical alias duplicates in PairTail inputs')
    if strategy not in ('tail_only', 'consensus_unlock', 'free_rerank'):
        raise ValueError('Unknown frozen PairTail policy')
    if not proposed:
        return baseline[:25]
    if not baseline or strategy == 'free_rerank':
        return proposed[:25]
    unlock = False
    if strategy == 'consensus_unlock':
        if not (len(independent_tops) == len(margins) == len(thresholds) == len(compatible)):
            raise ValueError('Incomplete independent engine evidence')
        challenger = proposed[0]
        votes = set()
        for (name, top), margin, threshold, domain in zip(independent_tops, margins, thresholds, compatible):
            if (domain is True and top == challenger and top != baseline[0]
                    and math.isfinite(margin) and math.isfinite(threshold) and margin > threshold):
                votes.add(name)
        unlock = len(votes) >= 2
    if unlock:
        return proposed[:25]
    top = baseline[0]
    return ([top]+[identity for identity in proposed if identity != top])[:25]


def local_module_gate(previous, candidate):
    checks = {
        'unknown_mrr': candidate['unknown']['mrr25']-previous['unknown']['mrr25'] >= -.0005,
        'known_mrr': candidate['known']['mrr25']-previous['known']['mrr25'] >= -.001,
        'known_top1': candidate['known']['top1']-previous['known']['top1'] >= -.002,
    }
    if not all(math.isfinite(float(v)) for arm in (previous, candidate)
               for mode in ('known', 'unknown') for v in arm[mode].values()):
        raise ValueError('Nonfinite development metric')
    return {'passed':all(checks.values()), 'checks':checks,
            'decision':'retain candidate' if all(checks.values()) else 'fall back to previous module'}


def independent_acceptance_gate(baseline, candidate, protocol):
    """Called only after candidate freeze and one sealed acceptance run.

    Each mode maps canonical query identities to evaluator metrics. Reject
    missing, additional or mismatched cohort identities before pairing.
    """
    result = {}
    for mode in ('unknown', 'known'):
        before, after = baseline[mode], candidate[mode]
        if not before or set(before) != set(after):
            raise ValueError('Acceptance comparison does not have identical nonempty cohorts')
        keys = sorted(before)
        for metric in ('rr', 'top1', 'recall25'):
            if any(not math.isfinite(float(arm[key][metric])) for arm in (before, after) for key in keys):
                raise ValueError('Nonfinite acceptance metric')
        result[mode] = {
            'molecules':len(keys),
            'paired_mrr_delta':paired([after[key]['rr']-before[key]['rr'] for key in keys],
                                      seed=protocol['seed'], repeats=protocol['repeats']),
            'top1_delta':sum(after[key]['top1']-before[key]['top1'] for key in keys)/len(keys),
            'recall25_delta':sum(after[key]['recall25']-before[key]['recall25'] for key in keys)/len(keys),
        }
    unknown, known = result['unknown'], result['known']
    # paired() provides a mean delta and CI; retain the actual arithmetic value
    # directly instead of depending on a display-oriented summary key.
    unknown_delta = sum(candidate['unknown'][key]['rr']-baseline['unknown'][key]['rr']
                        for key in baseline['unknown'])/len(baseline['unknown'])
    checks = {
        'unknown_delta_min':unknown_delta >= protocol['unknown_delta_min'],
        'unknown_ci_lower':unknown['paired_mrr_delta']['ci95'][0] > protocol['unknown_ci_lower_gt'],
        'known_ci_lower':known['paired_mrr_delta']['ci95'][0] >= protocol['known_ci_lower_min'],
        'known_top1':known['top1_delta'] >= -protocol['known_top1_drop_max'],
        'known_recall25':known['recall25_delta'] >= -protocol['known_recall25_drop_max'],
    }
    return {'passed':all(checks.values()), 'checks':checks, 'metrics':result,
            'unknown_mrr_delta':unknown_delta, 'competition_submission_allowed':all(checks.values())}
