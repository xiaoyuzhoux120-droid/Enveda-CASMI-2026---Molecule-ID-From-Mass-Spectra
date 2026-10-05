"""Frozen deployment policy: runtime evidence only, no query label argument."""
import numpy as np
from casmi_ml.np_pairtail_fusion import pairtail,reciprocal_fusion

def ranked_identities(identities,scores):
    scores=np.asarray(scores)
    if len(scores)!=len(identities) or not np.isfinite(scores).all():raise ValueError('Invalid candidate scores')
    return [identities[i] for i in np.lexsort((np.asarray(identities),-scores))]

def deploy_policy(c0,c1,ranked,fp,analog,fp_margin,analog_margin,confidence,policy):
    primary=(c1 if confidence>=.5 else ranked)[:25]
    proposed,_=reciprocal_fusion(primary,fp,policy['K'],policy['alpha'])
    if confidence>=.5:proposed=primary
    return pairtail(c0,proposed,policy['strategy'],
        independent_tops=[('fingerprint',fp[0]),('spectrum_analog',analog[0])],
        margins=[fp_margin,analog_margin],thresholds=policy['thresholds'],compatible=[True,True])
