"""Inference-only competition entry: dynamic IDs, frozen weights, no fitting."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from casmi_ml.data import write_json
from casmi_ml.chemical_priors import load_rules
from casmi_ml.np_pairtail_structures import sha256,canonical_target
from casmi_ml.np_pairtail_foundation import Progress,spectrum_signature
from casmi_ml.np_pairtail_learning_runtime import reference_scan,build_retrieval,center_mass,candidate_union
from casmi_ml.np_pairtail_ranker import StructureIndex,CandidateFeatures
from casmi_ml.np_pairtail_policy import ranked_identities,deploy_policy
from casmi_ml.np_pairtail_m3 import cpu_or_gpu_worker,_C4_WORKER
from casmi_ml.np_pairtail_acceptance import verify_candidate

RUNTIME_COLUMNS=('molecule_id','precursor_mz','adduct','ionization_mode','instrument_type',
                 'collision_energy_ev','ms2_mzs','ms2_normalized_intensities')

def runtime_frame(test_path):
    # Explicit column projection prevents copying visible dummy structure labels.
    frame=pd.read_parquet(test_path,columns=list(RUNTIME_COLUMNS))
    if not len(frame) or frame.molecule_id.isna().any():raise ValueError('Missing runtime query identifiers')
    frame['record_id']=np.arange(len(frame),dtype=np.int64)
    frame['identity']=frame.molecule_id.map(str)
    if frame.identity.nunique()!=frame.molecule_id.nunique():raise ValueError('Ambiguous identifier string conversion')
    frame['binned_signature']=[spectrum_signature(r)[1] for r in frame.to_dict('records')]
    return frame


def validate_submission(output,expected,identity_rows):
    if list(output.columns)!=['molecule_id','smiles'] or output.molecule_id.duplicated().any() or set(output.molecule_id)!=set(expected):
        raise ValueError('Submission query identity/schema mismatch')
    counts=[]
    for row in output.itertuples(index=False):
        candidates=row.smiles.split(';');ids=identity_rows[str(row.molecule_id)]
        if not 1<=len(candidates)<=25 or len(ids)!=len(candidates) or len(ids)!=len(set(ids)):
            raise ValueError('Illegal/duplicate canonical candidate count')
        for smi,key in zip(candidates,ids):canonical_target(smi,key)
        counts.append(len(ids))
    return {'rows':len(output),'minimum_candidates':min(counts),'maximum_candidates':max(counts),
            'all_smiles_valid':True,'canonical_identities_unique':True,'dynamic_query_ids':True}


def run(models,ranker,acceptance,train_path,test_path,coconut_path,catalog_path,dictionary_path,lotus_path,lotus_manifest,root,preparation_seconds=0.):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);models=Path(models);ranker=Path(ranker);acceptance=Path(acceptance)
    inputs={Path(p).name:p for p in [train_path,coconut_path,catalog_path,dictionary_path,lotus_path,lotus_manifest,
        ranker/'structure_identity_map.json',ranker/'reference_identity_map.json',ranker/'base_structure_pool.parquet']}
    freeze=verify_candidate(ranker,models,inputs)
    gate=json.loads((acceptance/'acceptance_gate.json').read_text())
    receipt=json.loads((acceptance/'acceptance_summary.json').read_text())
    freeze_sha=sha256(ranker/'candidate_freeze.json')
    if not gate['passed'] or not gate['competition_submission_allowed'] or receipt['candidate_freeze_sha256']!=freeze_sha:
        raise ValueError('Independent acceptance did not qualify this exact candidate')
    started=time.monotonic()-float(preparation_seconds);deadline=started+180*60
    progress=Progress(root,prior_seconds=0,total=180*60)
    frame=runtime_frame(test_path);original_ids=frame.molecule_id.drop_duplicates().tolist()
    frame.to_parquet(root/'runtime_queries.parquet',index=False)
    cpu_or_gpu_worker(['-m','casmi_ml.np_pairtail_fp_inference','--runtime',str(models),
        str(root/'runtime_queries.parquet'),freeze['fp_view'],str(root/'fingerprints')],deadline)
    keys=json.loads((root/'fingerprints/identities.json').read_text())
    logits={key:row for key,row in zip(keys,np.load(root/'fingerprints/logits.npy'))};bits=np.load(root/'fingerprints/bits.npy')
    mapping=json.loads((ranker/'structure_identity_map.json').read_text())
    for r in pd.read_parquet(ranker/'base_structure_pool.parquet').itertuples(index=False):
        if r.normalized_smiles in mapping and mapping[r.normalized_smiles]!=r.identity:raise ValueError('Frozen canonical cache inconsistency')
        mapping[r.normalized_smiles]=r.identity
    from casmi_ml.np_pairtail_fpnet import prepare_base_structure_pool
    pool_path=root/'runtime_structure_pool.parquet'
    prepare_base_structure_pool(train_path,coconut_path,catalog_path,mapping,pool_path,deadline,
        [center_mass(g) for _,g in frame.groupby('identity')])
    pool=pd.read_parquet(pool_path)
    if freeze['LOTUS_enabled']:pool=pd.concat([pool,pd.read_parquet(lotus_path)],ignore_index=True)
    index=StructureIndex(pool);rules=load_rules(dictionary_path);builder=CandidateFeatures(index,bits,rules)
    coconut=pd.read_parquet(coconut_path);catalog=pd.read_parquet(catalog_path)
    records,meta=reference_scan(train_path,{'runtime':frame},json.loads((ranker/'reference_identity_map.json').read_text()),root,progress,deadline)
    c0=build_retrieval(records,meta,coconut,catalog,rules,mapping,frame,'runtime','competition',None)
    c1=build_retrieval(records,meta,coconut,catalog,rules,mapping,frame,'runtime','competition',freeze['afix'])
    features=[];cases=[];offset=0
    for n,(key,group) in enumerate(frame.groupby('identity',sort=True),1):
        if time.monotonic()>=deadline:raise TimeoutError('Inference exceeds hard 180-minute budget')
        zero,_,_,_=c0.score(group);one,library,metadata,audit=c1.score(group)
        rows=candidate_union(index.query(center_mass(group)),one,index)
        if not len(rows):raise ValueError('Empty real inference candidate universe')
        f=builder.build(group,rows,logits[key],library,metadata);ids=rows.identity.tolist()
        fp=ranked_identities(ids,f.fp_dot.to_numpy());analog=ranked_identities(ids,f.analog_weighted.to_numpy())
        fpvalues=sorted(f.fp_dot.to_numpy(),reverse=True);av=sorted(f.analog_weighted.to_numpy(),reverse=True)
        smi={r['identity']:r['normalized_smiles'] for r in zero+one}
        smi.update(dict(zip(rows.identity,rows.normalized_smiles)))
        cases.append({'query':key,'offset':offset,'count':len(rows),'ids':ids,'smiles':smi,
            'c0':[r['identity'] for r in zero],'c1':[r['identity'] for r in one],'fp':fp,'analog':analog,
            'fp_margin':float(fpvalues[0]-fpvalues[1]) if len(fpvalues)>1 else 0.,
            'analog_margin':float(av[0]-av[1]) if len(av)>1 else 0.,'confidence':audit['confidence']})
        features.append(f);offset+=len(f)
        if n%25==0:progress.emit('M6_inference_features',queries=n,total=len(original_ids),candidate_rows=offset)
    pd.concat(features,ignore_index=True).to_parquet(root/'runtime_features.parquet',index=False)
    cpu_or_gpu_worker(['-c',_C4_WORKER,str(ranker/freeze['ranker']),str(root/'runtime_features.parquet'),str(root/'ranker_scores.npy')],deadline)
    scores=np.load(root/'ranker_scores.npy');predictions={};identity_rows={}
    for case in cases:
        ranked=ranked_identities(case['ids'],scores[case['offset']:case['offset']+case['count']])
        final=deploy_policy(case['c0'],case['c1'],ranked,case['fp'],case['analog'],
            case['fp_margin'],case['analog_margin'],case['confidence'],freeze['policy'])
        strings=[canonical_target(case['smiles'][key],key)[0] for key in final]
        predictions[case['query']]=';'.join(strings);identity_rows[case['query']]=final
    output=pd.DataFrame({'molecule_id':original_ids,'smiles':[predictions[str(k)] for k in original_ids]})
    validation=validate_submission(output,original_ids,identity_rows)
    elapsed=time.monotonic()-started
    if elapsed>=180*60:raise TimeoutError('Validated inference exceeds hard 180-minute budget')
    output.to_csv(root/'submission.csv',index=False)
    write_json(root/'inference_validation.json',validation|{'actual_seconds':elapsed,'target_150_minutes_passed':elapsed<=150*60,
        'hard_180_minutes_passed':True,'candidate_freeze_sha256':freeze_sha,'submission_sha256':sha256(root/'submission.csv'),
        'query_label_columns_read':False,'training_executed':False,'holdout_evaluation_executed':False})
    progress.emit('M6_inference_complete',competition_submission_allowed=True,competition_submitted=False)
    return output
