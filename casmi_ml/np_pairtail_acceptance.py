"""Exactly one sealed acceptance of an already frozen development candidate."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from casmi_ml.data import write_json
from casmi_ml.chemical_priors import load_rules
from casmi_ml.np_pairtail_structures import sha256
from casmi_ml.np_pairtail_foundation import Progress,mass_keep
from casmi_ml.np_pairtail_learning_runtime import (reference_scan,build_retrieval,center_mass,prediction_metrics,evaluate_order)
from casmi_ml.np_pairtail_ranker import StructureIndex,CandidateFeatures
from casmi_ml.np_pairtail_m3 import query_features,cpu_or_gpu_worker,bind_c0_rankings,ranking_policy,ranked_cases,metrics,_C4_WORKER
from casmi_ml.np_pairtail_fusion import independent_acceptance_gate


def verify_candidate(ranker,models,inputs):
    ranker,models=Path(ranker),Path(models)
    if json.loads((ranker/'run_status.json').read_text())['stage']!='M3_development_complete':raise ValueError('No approved development freeze')
    freeze=json.loads((ranker/'candidate_freeze.json').read_text())
    if freeze['acceptance_opened'] or freeze['competition_submission_allowed']:raise ValueError('Invalid pre-acceptance freeze')
    for name,expected in freeze['input_sha256'].items():
        if name not in inputs or sha256(inputs[name])!=expected:raise ValueError('Frozen input mismatch: '+name)
    source=Path(__file__).parents[1]
    for name,expected in freeze['all_packaged_sources_sha256'].items():
        if sha256(source/name)!=expected:raise ValueError('Frozen implementation mismatch: '+name)
    if sha256(ranker/freeze['ranker']/'ranker_manifest.json')!=freeze['ranker_manifest_sha256']:raise ValueError('Frozen ranker manifest changed')
    for name,expected in freeze['selected_boosters_sha256'].items():
        if sha256(ranker/freeze['ranker']/name)!=expected:raise ValueError('Frozen booster changed')
    if sha256(models/'M2_training_summary.json')!=freeze['model_manifest_sha256']:raise ValueError('FPNet selection changed')
    if sha256(ranker/'data_manifest.json')!=freeze['data_manifest_sha256']:raise ValueError('Frozen data provenance changed')
    if sha256(ranker/'protocol_ranker_20261005.json')!=freeze['protocol_sha256']:raise ValueError('Scientific protocol changed')
    return freeze


def run(foundation,models,ranker,train_path,coconut_path,catalog_path,dictionary_path,lotus_path,lotus_manifest,root,preparation_seconds=0.):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);foundation=Path(foundation);ranker=Path(ranker);models=Path(models)
    mapping_path=ranker/'structure_identity_map.json';pool_path=ranker/'base_structure_pool.parquet'
    inputs={Path(p).name:p for p in [train_path,coconut_path,catalog_path,dictionary_path,lotus_path,lotus_manifest,mapping_path,ranker/'reference_identity_map.json',pool_path]}
    freeze=verify_candidate(ranker,models,inputs)
    prior=float(json.loads((ranker/'run_status.json').read_text())['cumulative_seconds'])+float(preparation_seconds)
    protocol=json.loads((foundation/'protocol_np_pairtail_20261004.json').read_text())
    if prior>=protocol['total_research_budget_seconds']:raise TimeoutError('No research budget for acceptance')
    deadline=time.monotonic()+protocol['total_research_budget_seconds']-prior
    progress=Progress(root,prior_seconds=prior,total=protocol['total_research_budget_seconds'])
    marker=root/'acceptance_opened.json'
    with marker.open('x') as handle:json.dump({'candidate_freeze_sha256':sha256(ranker/'candidate_freeze.json'),
        'scientific_parameters_may_change':False,'repeat_selection_allowed':False},handle)
    try:
        frame=pd.read_parquet(foundation/'acceptance.parquet')
        keys=set(frame.identity);split=json.loads((foundation/'split_manifest.json').read_text())
        if len(keys)<1500 or not keys<=set(split['identity_lists']['acceptance']):raise ValueError('Acceptance membership mismatch')
        if keys&(set(split['identity_lists']['train'])|set(split['identity_lists']['development'])):raise ValueError('Acceptance identity leakage')
        frame['molecule_id']=frame.identity
        frame.to_parquet(root/'runtime_queries.parquet',index=False)
        cpu_or_gpu_worker(['-m','casmi_ml.np_pairtail_fp_inference','--runtime',str(models),
            str(root/'runtime_queries.parquet'),freeze['fp_view'],str(root/'fingerprints')],deadline)
        fpkeys=json.loads((root/'fingerprints/identities.json').read_text())
        logits={key:value for key,value in zip(fpkeys,np.load(root/'fingerprints/logits.npy'))}
        bits=np.load(root/'fingerprints/bits.npy')
        mapping=json.loads(mapping_path.read_text())
        audited_pool=pd.read_parquet(pool_path)
        for row in audited_pool.itertuples(index=False):
            if row.normalized_smiles in mapping and mapping[row.normalized_smiles]!=row.identity:raise ValueError('Canonical cache disagreement')
            mapping[row.normalized_smiles]=row.identity
        from casmi_ml.np_pairtail_fpnet import prepare_base_structure_pool
        target_pool=root/'acceptance_structure_pool.parquet'
        prepare_base_structure_pool(train_path,coconut_path,catalog_path,mapping,target_pool,deadline,
            [center_mass(g) for _,g in frame.groupby('identity')])
        pool=pd.read_parquet(target_pool)
        if freeze['LOTUS_enabled']:pool=pd.concat([pool,pd.read_parquet(lotus_path)],ignore_index=True)
        rules=load_rules(dictionary_path);coconut=pd.read_parquet(coconut_path);catalog=pd.read_parquet(catalog_path)
        records,meta=reference_scan(train_path,{'acceptance':frame},json.loads((ranker/'reference_identity_map.json').read_text()),root,progress,deadline)
        known_keys={m['identity'] for m in meta if not m['is_acceptance_query_signature'] and
                    mass_keep(m,30,True) and m['mass_source']=='formula'}
        known=frame[frame.identity.isin(known_keys)]
        if not known.identity.nunique():raise ValueError('Empty frozen C0 known acceptance cohort')
        index=StructureIndex(pool);builder=CandidateFeatures(index,bits,rules)
        c0={'unknown':[],'known':[]};parts=[];cases=[];offset=0
        for mode,query in [('unknown',frame),('known',known)]:
            baseline=build_retrieval(records,meta,coconut,catalog,rules,mapping,frame,'acceptance',mode,None)
            for truth,group in query.groupby('identity',sort=True):
                if time.monotonic()>=deadline:raise TimeoutError('Acceptance baseline budget exhausted')
                rows,_,_,_=baseline.score(group)
                c0[mode].append(evaluate_order(truth,[r['identity'] for r in rows]))
            del baseline
            engine=build_retrieval(records,meta,coconut,catalog,rules,mapping,frame,'acceptance',mode,freeze['afix'])
            features,current=query_features(query,engine,index,builder,logits,mode,root,progress,deadline)
            for c in current:c['offset']+=offset
            offset+=len(features);parts.append(features);cases.extend(current);del engine
        features=pd.concat(parts,ignore_index=True);features.to_parquet(root/'frozen_features.parquet',index=False)
        cpu_or_gpu_worker(['-c',_C4_WORKER,str(ranker/freeze['ranker']),str(root/'frozen_features.parquet'),str(root/'ranker_scores.npy')],deadline)
        primary=ranked_cases(cases,np.load(root/'ranker_scores.npy'));bind_c0_rankings(cases,c0)
        p=freeze['policy'];candidate=ranking_policy(cases,primary,p['K'],p['alpha'],p['strategy'],p['thresholds'])
        write_json(root/'C0_acceptance_cases.json',c0);write_json(root/'candidate_acceptance_cases.json',candidate)
        gate=independent_acceptance_gate({m:{r['identity']:r for r in rs} for m,rs in c0.items()},
            {m:{r['identity']:r for r in rs} for m,rs in candidate.items()},protocol['acceptance'])
        if time.monotonic()>=deadline:raise TimeoutError('Acceptance research budget exhausted')
        write_json(root/'acceptance_gate.json',gate)
        write_json(root/'acceptance_summary.json',{'C0':metrics(c0),'candidate':metrics(candidate),'gate':gate,
            'candidate_freeze_sha256':sha256(ranker/'candidate_freeze.json'),'parameter_changes':False})
        progress.emit('M5_acceptance_passed' if gate['passed'] else 'M5_acceptance_rejected',
            acceptance_opened=True,competition_submission_allowed=gate['passed'])
        return gate
    except Exception as exc:
        progress.emit('M5_acceptance_failed',error_type=type(exc).__name__,error=str(exc),
            acceptance_opened=True,competition_submission_allowed=False)
        raise
