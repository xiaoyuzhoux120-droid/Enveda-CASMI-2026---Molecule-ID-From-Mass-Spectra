"""M3 development-only training/selection; sealed acceptance is never read."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.chemical_priors import load_rules
from casmi_ml.np_pairtail_foundation import order,Progress
from casmi_ml.np_pairtail_learning_runtime import (reference_scan,build_retrieval,supported_training_measurements,
    candidate_union,center_mass,prediction_metrics,evaluate_order)
from casmi_ml.np_pairtail_ranker import (FEATURES,StructureIndex,CandidateFeatures,
    select_negative_rows,labeled_group,isolated_booster_fit)
from casmi_ml.np_pairtail_fusion import local_module_gate,pairtail,reciprocal_fusion
from casmi_ml.np_pairtail_structures import sha256
from casmi_ml.np_pairtail_policy import deploy_policy


def cpu_or_gpu_worker(command,deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('M3 research/module budget exhausted')
    env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,*command],env=env,check=True,timeout=remaining)


def cached_query_logits(models,cache,keys,view,root,deadline):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    write_json(root/'requested_identities.json',keys)
    cpu_or_gpu_worker(['-m','casmi_ml.np_pairtail_fp_inference',str(models),str(cache),
        str(root/'requested_identities.json'),view,str(root)],deadline)
    identities=json.loads((root/'identities.json').read_text());logits=np.load(root/'logits.npy')
    if identities!=keys or len(logits)!=len(keys):raise ValueError('Fingerprint worker query identity order mismatch')
    return {key:logits[i] for i,key in enumerate(keys)},np.load(root/'bits.npy')


def query_features(frame,engine,index,builder,logits,mode,root,progress,deadline):
    cases=[];parts=[];offset=0
    for n,(truth,group) in enumerate(frame.groupby('identity',sort=True),1):
        if time.monotonic()>=deadline:raise TimeoutError('M3 candidate feature budget exhausted')
        baseline,library,meta,audit=engine.score(group)
        rows=candidate_union(index.query(center_mass(group)),baseline,index)
        if not len(rows):raise ValueError('Empty deployment candidate universe')
        token=hashlib.sha256('|'.join(sorted(group.binned_signature)).encode()).hexdigest()
        features=builder.build(group,rows,logits[truth],library,meta,
            training_token=token if mode.startswith('train_') else None)
        ids=rows.identity.tolist();baseline_ids=[r['identity'] for r in baseline]
        fp=np.lexsort((np.array(ids),-features.fp_dot.to_numpy()))
        analog=np.lexsort((np.array(ids),-features.analog_weighted.to_numpy()))
        cases.append({'identity':truth,'mode':mode,'offset':offset,'count':len(rows),
            'candidate_identities':ids,'candidate_smiles':rows.normalized_smiles.tolist(),
            'candidate_sources':rows.sources.tolist(),'baseline':baseline_ids,
            'fp_ranking':[ids[i] for i in fp],'analog_ranking':[ids[i] for i in analog],
            'fp_margin':float(features.fp_dot.iloc[fp[0]]-features.fp_dot.iloc[fp[1]]) if len(fp)>1 else 0.,
            'analog_margin':float(features.analog_weighted.iloc[analog[0]]-features.analog_weighted.iloc[analog[1]]) if len(analog)>1 else 0.,
            'measurement_token':token,'confidence':audit['confidence'],
            'training_explanation_rows':features.attrs['training_explanation_rows']})
        parts.append(features);offset+=len(rows)
        if n%25==0:progress.emit('M3_'+mode+'_features',completed=n,total=int(frame.identity.nunique()),
            candidate_rows=offset,fingerprint_cache=len(index.fingerprints),fragment_cache=len(builder.fragments))
    if not parts:raise ValueError('Empty feature/evaluation cohort')
    all_features=pd.concat(parts,ignore_index=True)
    all_features.to_parquet(Path(root)/(mode+'_features.parquet'),index=False)
    write_json(Path(root)/(mode+'_cases.json'),cases)
    return all_features,cases


def ranked_cases(cases,scores):
    result={'unknown':[],'known':[]}
    for case in cases:
        value=scores[case['offset']:case['offset']+case['count']]
        ids=case['candidate_identities']
        ranking=[ids[i] for i in np.lexsort((np.array(ids),-value))]
        # Preserve original V1 deployment guard on measurement evidence.
        if case['confidence']>=.5:ranking=case['baseline']
        result[case['mode']].append(evaluate_order(case['identity'],ranking))
    return result


def metrics(cases):return {mode:prediction_metrics(rows) for mode,rows in cases.items()}


def baseline_cases(cases):
    result={'unknown':[],'known':[]}
    for case in cases:result[case['mode']].append(evaluate_order(case['identity'],case['baseline']))
    return result


def compare_tail(before,after):
    result={}
    for mode in ('unknown','known'):
        b={r['identity']:r for r in before[mode]};a={r['identity']:r for r in after[mode]}
        if set(b)!=set(a):raise ValueError('Tail cohort changed')
        result[mode]={'rescued25':sum(b[k]['rank']==0 and a[k]['rank']>0 for k in b),
            'lost25':sum(b[k]['rank']>0 and a[k]['rank']==0 for k in b),
            'promoted_top1':sum(b[k]['rank']!=1 and a[k]['rank']==1 for k in b),
            'demoted_top1':sum(b[k]['rank']==1 and a[k]['rank']!=1 for k in b),
            'net_tail_gain':sum(a[k]['rr']-b[k]['rr'] for k in b if 2<=b[k]['rank']<=25)}
    return result


def bind_c0_rankings(cases, c0):
    """Attach frozen C0 predictions, never C1 or evaluator truth ranks."""
    original={mode:{r['identity']:r for r in rows} for mode,rows in c0.items()}
    for case in cases:
        row=original[case['mode']][case['identity']]
        ranking=[item['identity'] if isinstance(item,dict) else item for item in row['ranking']]
        if len(ranking)!=len(set(ranking)):raise ValueError('Frozen C0 canonical duplicates')
        case['c0_baseline']=ranking


def np_slice_metrics(cases, rankings, pool):
    """Evaluation-only rare source-record proxy; never a ranking input."""
    rare=set(pool.groupby('identity').size().loc[lambda n:n<=1].index)
    np_ids=set(pool.loc[pool.sources.str.contains('COCONUT|LOTUS',regex=True),'identity'])
    keys=rare&np_ids
    output={}
    for mode in ('unknown','known'):
        rows=[r for r in rankings[mode] if r['identity'] in keys]
        output[mode]=prediction_metrics(rows) if rows else None
    return output



def candidate_funnel(cases):
    """Evaluator-only proposal coverage; no truth-dependent candidate changes."""
    result={}
    for mode in ('unknown','known'):
        cohort=[c for c in cases if c['mode']==mode]
        by_source={}
        for source in ('train/library','COCONUT','LOTUS','ChEBI','LIPID_MAPS'):
            proposed=sum(sum(source in flags for flags in c['candidate_sources']) for c in cohort)
            covered=sum(any(key==c['identity'] and source in flags for key,flags in
                        zip(c['candidate_identities'],c['candidate_sources'])) for c in cohort)
            by_source[source]={'proposals':proposed,'exact_truth_covered_queries':covered}
        result[mode]={'queries':len(cohort),'proposals':sum(c['count'] for c in cohort),
            'exact_truth_covered_queries':sum(c['identity'] in c['candidate_identities'] for c in cohort),
            'sources':by_source}
    return result


def ranking_policy(cases,ranker_cases,k,alpha,strategy,thresholds):
    primary={mode:{r['identity']:r['ranking'] for r in rows} for mode,rows in ranker_cases.items()}
    output={'unknown':[],'known':[]}
    for case in cases:
        mode,key=case['mode'],case['identity']
        ranking=deploy_policy(case['c0_baseline'],case['baseline'],primary[mode][key],
            case['fp_ranking'],case['analog_ranking'],case['fp_margin'],case['analog_margin'],case['confidence'],
            {'K':k,'alpha':alpha,'strategy':strategy,'thresholds':thresholds})
        output[mode].append(evaluate_order(key,ranking))
    return output


def run(foundation,models,cache,train_path,coconut_path,catalog_path,dictionary_path,
        mapping_path,lotus_path,lotus_manifest,protocol,pool_path,root,preparation_seconds=0.,reference_mapping_path=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);foundation=Path(foundation);models=Path(models)
    reference_mapping_path=reference_mapping_path or mapping_path
    m2=json.loads((models/'run_status.json').read_text())
    if m2['stage']!='M2_complete':raise ValueError('M2 gate incomplete; M3 forbidden')
    restart_seconds=float(protocol.get('restart_additional_research_seconds',0.))
    prior=float(m2['cumulative_seconds'])+float(protocol.get('restart_prior_m3_seconds',0.))+float(preparation_seconds)
    if restart_seconds:
        progress=Progress(root,prior_seconds=prior,total=prior+restart_seconds)
        started=time.monotonic();deadline=started+restart_seconds-float(preparation_seconds)
    else:
        progress=Progress(root,prior_seconds=prior,total=protocol['total_research_seconds'])
        started=time.monotonic();deadline=min(started+protocol['ranker_module_seconds']-float(preparation_seconds),
            started+protocol['total_research_seconds']-prior)
    write_json(root/'protocol_ranker_20261005.json',protocol)
    try:
        if time.monotonic()>=deadline:raise TimeoutError('M3 preparation exhausted module/research budget')
        # No acceptance reader, even metadata, in the M3 entry.
        train=pd.read_parquet(foundation/'train.parquet');dev=pd.read_parquet(foundation/'development.parquet')
        split=json.loads((foundation/'split_manifest.json').read_text())
        training_keys=sorted(set(train.identity),key=lambda k:(order(k,'ranker-query'),k))[:protocol['training_query_cap']]
        training=train[train.identity.isin(training_keys)].copy()
        training, domain_audit = supported_training_measurements(training)
        write_json(root/'training_measurement_domain_audit.json', domain_audit)
        train_keys=sorted(training_keys);dev_keys=sorted(set(dev.identity))
        if set(train_keys)&set(dev_keys):raise ValueError('Ranker train/development identity leakage')
        write_json(root/'ranker_training_identities.json',train_keys)
        selection=json.loads((models/'M2_development_candidate_gate.json').read_text())['selected_view']
        train_logits,bits=cached_query_logits(models,Path(cache)/'train_cache',train_keys,selection,root/'train_logits',deadline)
        dev_logits,dev_bits=cached_query_logits(models,Path(cache)/'development_cache',dev_keys,selection,root/'dev_logits',deadline)
        if not np.array_equal(bits,dev_bits):raise ValueError('Training/development fingerprint bits differ')
        mapping=json.loads(Path(mapping_path).read_text())
        if not Path(pool_path).exists():
            from casmi_ml.np_pairtail_fpnet import prepare_base_structure_pool
            centers=[center_mass(g) for f in (training,dev) for _,g in f.groupby('identity')]
            prepare_base_structure_pool(train_path,coconut_path,catalog_path,mapping,pool_path,deadline,centers)
            prepared=json.loads(Path(pool_path).with_suffix('.manifest.json').read_text())
            prepared['scope']='ranker train/development deployment mass-window union'
            write_json(Path(pool_path).with_suffix('.manifest.json'),prepared)
        pool=pd.read_parquet(pool_path);manifest=json.loads(Path(pool_path).with_suffix('.manifest.json').read_text())
        if manifest['scope']!='ranker train/development deployment mass-window union' or manifest['sha256']!=sha256(pool_path):
            raise ValueError('M3 requires its own full-source scoped pool, not the M2 dev-only pool')
        mapping=json.loads(Path(mapping_path).read_text())
        coconut=pd.read_parquet(coconut_path,columns=['inchikey','canonical_smiles','exact_mass']);catalog=pd.read_parquet(catalog_path)
        rules=load_rules(dictionary_path)
        reference_mapping=json.loads(Path(reference_mapping_path).read_text()) if reference_mapping_path else mapping
        records,metadata=reference_scan(train_path,{'train':training,'development':dev},reference_mapping,root,progress,deadline)
        afix=json.loads((foundation/'M1_afix_selection.json').read_text())['threshold_ppm']
        # Freeze exact C0 known cohort from the completed foundation evaluator.
        old_known=json.loads((foundation/'C0_known_cases.json').read_text())
        known_keys={r['identity'] for r in old_known}
        if not known_keys<=set(dev.identity):raise ValueError('Frozen known cohort not in development')
        dknown=dev[dev.identity.isin(known_keys)]
        feature_parts=[];training_cases=[];training_groups=[];training_labels=[]
        index=StructureIndex(pool);builder=CandidateFeatures(index,bits,rules)
        for mode in ('unknown','known'):
            engine=build_retrieval(records,metadata,coconut,catalog,rules,mapping,training,'train',mode,afix,set(split['identity_lists']['train']))
            frame=training if mode=='unknown' else training[training.identity.isin({m['identity'] for m in engine.meta})]
            features,cases=query_features(frame,engine,index,builder,train_logits,'train_'+mode,root,progress,deadline)
            for case in cases:
                case['mode']=mode
                f=features.iloc[case['offset']:case['offset']+case['count']]
                selected=select_negative_rows(f,case['measurement_token'])
                x,y=labeled_group(f,case['candidate_identities'],case['identity'],selected)
                feature_parts.append(pd.DataFrame(x,columns=FEATURES));training_groups.append(len(y));training_labels.extend(y)
                training_cases.append(case)
            del engine
        xtrain=pd.concat(feature_parts,ignore_index=True);ytrain=np.array(training_labels,np.int32)
        if not ytrain.any():raise ValueError('No positive groups under real label-blind training proposal')
        write_json(root/'training_candidate_audit.json',{'identities':len(training_keys),'groups':len(training_groups),
            'rows':len(ytrain),'positives':int(ytrain.sum()),'truth_injected':False,
            'features':list(FEATURES),'acceptance_opened':False})
        all_dev_features=[];dev_cases=[];offset=0
        for mode,frame in [('unknown',dev),('known',dknown)]:
            engine=build_retrieval(records,metadata,coconut,catalog,rules,mapping,dev,'development',mode,afix)
            features,cases=query_features(frame,engine,index,builder,dev_logits,mode,root,progress,deadline)
            for c in cases:c['offset']+=offset
            offset+=len(features);all_dev_features.append(features);dev_cases.extend(cases)
            del engine
        xdev=pd.concat(all_dev_features,ignore_index=True);base=baseline_cases(dev_cases);base_metrics=metrics(base)
        # Compare with original C1 frozen metrics before fitting a booster.
        frozen=json.loads((foundation/'development_foundation_results.json').read_text())
        label='C0' if afix is None else 'C1_ppm'+str(afix)
        for mode in ('unknown','known'):
            for metric in ('mrr25','top1','recall25'):
                if abs(base_metrics[mode][metric]-frozen[label+'_'+mode][metric])>1e-12:
                    raise ValueError('Frozen foundation baseline parity failed: '+mode+' '+metric)
        grid=[];selected=None
        for depth,leaf in ((5,50),(5,100),(6,50),(6,100)):
            name=f'depth{depth}_leaf{leaf}'
            score=isolated_booster_fit(xtrain,ytrain,training_groups,xdev,depth,leaf,root/name,deadline)
            ranked=ranked_cases(dev_cases,score);summary=metrics(ranked);gate=local_module_gate(base_metrics,summary)
            grid.append({'name':name,'max_depth':depth,'min_leaf':leaf,'metrics':summary,'gate':gate})
            write_json(root/(name+'_development_cases.json'),ranked)
            if gate['passed'] and (selected is None or (summary['unknown']['mrr25'],-depth,-leaf)>
                (selected['metrics']['unknown']['mrr25'],-selected['depth'],-selected['leaf'])):
                selected={'name':name,'depth':depth,'leaf':leaf,'metrics':summary,'cases':ranked}
        report={'grid':grid,'C3_selected':selected['name'] if selected else None,'C3_fallback':selected is None,
            'C1':base_metrics,'C3':selected['metrics'] if selected else base_metrics,'acceptance_opened':False,
            'competition_submission_allowed':False}
        write_json(root/'M3_development_results.json',report)
        # PRD section 10 requires fallback to the previous qualified
        # module and continued C6 evaluation, not abandonment at C3.
        if selected is None:
            report['C4']={'enabled':False,'reason':'No qualified learned ranker; retain C1 and continue frozen C6'}
            report['C4_gate']={'passed':False,'executed':False,'fallback':'C1'}
            chosen_cases,chosen_ranked=dev_cases,base
            lotus_enabled=False
        else:
            # C4 is solely the LOTUS candidate/source increment, using C3 weights.
            lm=json.loads(Path(lotus_manifest).read_text())
            if lm['scope']!='full source, no query window or label inputs' or lm['sha256']!=sha256(lotus_path):
                raise ValueError('Unqualified LOTUS artifact')
            lotus=pd.read_parquet(lotus_path)
            union=pd.concat([pool,lotus[['identity','normalized_smiles','mass','sources']]],ignore_index=True)
            c4index=StructureIndex(union);c4builder=CandidateFeatures(c4index,bits,rules)
            c4parts=[];c4cases=[];offset=0
            for mode,frame in [('unknown',dev),('known',dknown)]:
                engine=build_retrieval(records,metadata,coconut,catalog,rules,mapping,dev,'development',mode,afix)
                features,cases=query_features(frame,engine,c4index,c4builder,dev_logits,'C4_'+mode,root,progress,deadline)
                for c in cases:c['offset']+=offset;c['mode']=mode
                offset+=len(features);c4parts.append(features);c4cases.extend(cases);del engine
            # CPU-only worker loads selected boosters; never fits a new NP-specific
            # model against held-out labels or changes the four-seed training freeze.
            c4frame=pd.concat(c4parts,ignore_index=True);c4frame.to_parquet(root/'C4_features.parquet',index=False)
            cpu_or_gpu_worker(['-c',_C4_WORKER,str(root/selected['name']),str(root/'C4_features.parquet'),str(root/'C4_scores.npy')],deadline)
            c4ranked=ranked_cases(c4cases,np.load(root/'C4_scores.npy'));c4metrics=metrics(c4ranked)
            c4gate=local_module_gate(selected['metrics'],c4metrics)
            # Slice membership is identical in both arms and fixed by the union
            # source-record count, never by whether the candidate predicted truth.
            rare_before=np_slice_metrics(cases,selected['cases'],union)
            rare_after=np_slice_metrics(c4cases,c4ranked,union)
            rare_checks={mode:(rare_before[mode] is None or
                rare_after[mode]['mrr25']-rare_before[mode]['mrr25']>=-.002)
                for mode in ('unknown','known')}
            c4gate['checks']['NP_low_frequency']=all(rare_checks.values())
            c4gate['passed']=all(c4gate['checks'].values())
            c4gate['NP_low_frequency_proxy']={'before':rare_before,'after':rare_after,'checks':rare_checks}
            write_json(root/'C4_development_cases.json',c4ranked)
            report.update(C4=c4metrics,C4_gate=c4gate,
                candidate_funnel_C3=candidate_funnel(dev_cases),candidate_funnel_C4=candidate_funnel(c4cases))
            chosen_cases,chosen_ranked=(c4cases,c4ranked) if c4gate['passed'] else (dev_cases,selected['cases'])
            lotus_enabled=c4gate['passed']
        # Unqualified external forward assets remain disabled, explicitly.
        report['C5']={'enabled':False,'eligible_queries':None,'scored_queries':0,
            'reason':protocol['forward'],'fallback':('C4' if lotus_enabled else 'C3') if selected else 'C1'}
        thresholds=[float(np.median([c[name] for c in training_cases if c[name]>0]))
                    if any(c[name]>0 for c in training_cases) else float('inf')
                    for name in ['fp_margin','analog_margin']]
        if not all(np.isfinite(thresholds)):thresholds=[1e30,1e30]
        # C6 uses the real C0 rankings, fixed before any new ranking selection.
        c0={'unknown':json.loads((foundation/'C0_unknown_cases.json').read_text()),'known':old_known}
        bind_c0_rankings(chosen_cases,c0)
        c0_metrics=metrics(c0);policies=[];winner=None
        for k in (3,10):
            for alpha in (.3,.6):
                for strategy in ('tail_only','consensus_unlock','free_rerank'):
                    prediction=ranking_policy(chosen_cases,chosen_ranked,k,alpha,strategy,thresholds)
                    pm=metrics(prediction);tail=compare_tail(c0,prediction);gate=local_module_gate(c0_metrics,pm)
                    tail_pass=(tail['unknown']['net_tail_gain']>0 and tail['unknown']['rescued25']>tail['unknown']['lost25'])
                    policy={'K':k,'alpha':alpha,'strategy':strategy,'thresholds':thresholds,'metrics':pm,
                            'tail':tail,'local_gate':gate,'tail_gate':tail_pass}
                    policies.append(policy)
                    if gate['passed'] and tail_pass and (winner is None or pm['unknown']['mrr25']>winner['metrics']['unknown']['mrr25']):
                        winner=policy|{'cases':prediction}
        report['C6_grid']=policies
        if winner is None:
            report['decision']='development rejected: no PairTail policy passed frozen tail protection'
            write_json(root/'M3_development_results.json',report)
            write_json(root/'run_status.json',{'stage':'M3_development_rejected','cumulative_seconds':prior+time.monotonic()-started,
                'acceptance_opened':False,'competition_submission_allowed':False})
            return report
        policy={key:value for key,value in winner.items() if key!='cases'}
        freeze={'ranker_enabled':selected is not None,'ranker':selected['name'] if selected else None,
            'ranker_manifest_sha256':sha256(root/selected['name']/'ranker_manifest.json') if selected else None,
            'fp_view':selection,'afix':afix,'LOTUS_enabled':lotus_enabled,'policy':policy,
            'model_manifest_sha256':sha256(models/'M2_training_summary.json'),'features':list(FEATURES),
            'C3_routing':protocol['C3_routing'],'forward_enabled':False,'popularity_enabled':False,
            'acceptance_opened':False,'competition_submission_allowed':False,
            'input_sha256':{str(Path(path).name):sha256(path) for path in
                [train_path,coconut_path,catalog_path,dictionary_path,mapping_path,reference_mapping_path,lotus_path,lotus_manifest,pool_path]},
            'protocol_sha256':sha256(root/'protocol_ranker_20261005.json'),
            'data_manifest_sha256':sha256(root/'data_manifest.json'),
            'all_packaged_sources_sha256':{str(p.relative_to(Path(__file__).parents[1])):sha256(p)
                for p in Path(__file__).parents[1].rglob('*.py')},
            'source_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('np_pairtail*.py')},
            'reference_files_sha256':{name:sha256(root/name) for name in
                ['reference_rows.parquet','reference_metadata.parquet','reference_spectra.npz']},
            'selected_boosters_sha256':{p.name:sha256(p) for p in (root/selected['name']).glob('booster_*.txt')} if selected else {}}
        if time.monotonic()>=deadline:raise TimeoutError('Budget exhausted before candidate freeze')
        write_json(root/'candidate_freeze.json',freeze)
        write_json(root/'selected_development_cases.json',winner['cases'])
        report['selected_policy']=policy;report['decision']='frozen for one sealed acceptance only'
        write_json(root/'M3_development_results.json',report)
        progress.emit('M3_development_complete',acceptance_opened=False,competition_submission_allowed=False,
                      candidate_freeze_sha256=sha256(root/'candidate_freeze.json'),
                      cumulative_seconds=prior+time.monotonic()-started)
        return report
    except Exception as exc:
        write_json(root/'run_status.json',{'stage':'M3_failed','error_type':type(exc).__name__,'error':str(exc),
            'cumulative_seconds':prior+time.monotonic()-started,'acceptance_opened':False,'competition_submission_allowed':False})
        raise


_C4_WORKER='''import sys\nfrom pathlib import Path\nimport lightgbm as lgb\nimport numpy as np\nimport pandas as pd\nfrom casmi_ml.np_pairtail_ranker import score_boosters\nroot=Path(sys.argv[1])\nmodels=[lgb.Booster(model_file=str(root/f'booster_{seed}.txt')) for seed in (20261004,20261005,20261006,20261007)]\nnp.save(sys.argv[3],score_boosters(models,pd.read_parquet(sys.argv[2])))\n'''
