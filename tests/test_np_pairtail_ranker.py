import time

import numpy as np
import pandas as pd
import pytest


def test_training_retrieval_domain_never_uses_truth_or_backfills_identities():
    from casmi_ml.np_pairtail_learning_runtime import supported_training_measurements, center_mass
    frame = pd.DataFrame({'identity':['a','a','b','c'],
        'precursor_mz':[201.007276466621, 100., 100., np.nan],
        'adduct':['[M+H]+','[M+2H]2+','unknown','[M+H]+'],
        'molecular_formula':['C999','C999','C999','C999']})
    kept, audit = supported_training_measurements(frame)
    assert kept.index.tolist() == [0]
    assert np.isclose(center_mass(kept), 200.)
    assert audit['unsupported_only_identities'] == ['b','c']
    assert audit['dropped_rows'] == 3 and not audit['identity_backfill']
    assert not audit['query_truth_mass_used']
    with pytest.raises(ValueError, match='Unsupported'):
        center_mass(frame)  # Evaluation retains its strict existing behavior.

from casmi_ml.np_pairtail_ranker import (FEATURES, StructureIndex, feature_matrix,
    fit_boosters, fragment_masses, labeled_group, peak_coverage, score_boosters,
    select_negative_rows)


def test_alias_mass_window_keeps_eligible_alias_and_all_sources():
    pool=pd.DataFrame([
        {'identity':'a','normalized_smiles':'CC','mass':200.,'sources':'COCONUT'},
        {'identity':'a','normalized_smiles':'CCC','mass':199.999,'sources':'LOTUS'},
        {'identity':'a','normalized_smiles':'CCCC','mass':250.,'sources':'ChEBI'},
        {'identity':'b','normalized_smiles':'CO','mass':250.,'sources':'train/library'}])
    rows=StructureIndex(pool).query(200.)
    assert rows.identity.tolist()==['a']
    assert rows.normalized_smiles.tolist()==['CC']
    assert rows.sources.tolist()==['COCONUT;ChEBI;LOTUS']


def test_truth_ids_cannot_be_model_features():
    features=pd.DataFrame(np.zeros((2,len(FEATURES))),columns=FEATURES)
    assert feature_matrix(features).shape==(2,len(FEATURES))
    with pytest.raises(ValueError,match='allowlist'):
        feature_matrix(features.assign(molecule_id=['m_1','m_2']))
    with pytest.raises(ValueError,match='allowlist'):
        feature_matrix(features.assign(truth_in_pool=[1,0]))


def test_label_blind_negative_selection_never_inserts_truth():
    features=pd.DataFrame(np.zeros((600,len(FEATURES))),columns=FEATURES)
    features.library_score=np.arange(600)
    features.fp_dot=np.arange(600)[::-1]
    ids=select_negative_rows(features,'training-feature-group')
    assert len(ids)==256 and len(set(ids))==256
    assert np.array_equal(ids,select_negative_rows(features,'training-feature-group'))
    omitted=next(i for i in range(600) if i not in ids)
    _,labels=labeled_group(features,list(range(600)),omitted,ids)
    assert labels.sum()==0


def test_fragment_mass_preserves_original_hydrogen_not_product_cap():
    one,two,total=fragment_masses('CCO')
    assert np.isclose(total,46.041864812)
    # Original terminal CH3 and OH radical components, no cleavage H capping.
    assert any(np.isclose(one,15.023475096))
    assert any(np.isclose(one,17.002739652))
    assert len(two)>0
    _,two_long,_=fragment_masses('C'*27)
    assert two_long is None


def test_fragment_coverage_is_intensity_weighted():
    assert np.isclose(peak_coverage([100.],np.array([100.,200.]),np.array([1.,3.])),.25)


def test_four_seed_ranker_groupwise_and_id_free(tmp_path):
    # Torch and LightGBM ship different OpenMP runtimes on this Mac. Test the
    # real CPU booster in its own process, as the production worker will run.
    import subprocess
    import sys
    script = r"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from casmi_ml.np_pairtail_ranker import FEATURES, fit_boosters, score_boosters
rng=np.random.default_rng(2)
x=rng.normal(size=(600,len(FEATURES))).astype(np.float32)
groups=[6]*100
y=np.zeros(600,np.int32)
for n in range(100): y[6*n+np.argmax(x[6*n:6*n+6,0])]=1
models=fit_boosters(x,y,groups,5,50,Path(__import__('sys').argv[1]),time.monotonic()+60)
frame=pd.DataFrame(x,columns=FEATURES)
score=score_boosters(models,frame).reshape(100,6)
assert np.mean(np.argmax(score,1)==y.reshape(100,6).argmax(1))>.9
assert np.array_equal(score_boosters(models,frame),score_boosters(models,frame.copy()))
try: fit_boosters(x,y,[1],5,50,Path(__import__('sys').argv[1]),time.monotonic()+60)
except ValueError as e: assert 'partition' in str(e)
else: raise AssertionError('Malformed query groups accepted')
print('FOUR_SEED_CPU_WORKER_PASSED')
"""
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'FOUR_SEED_CPU_WORKER_PASSED' in result.stdout
    assert len(list(tmp_path.glob('booster_*.txt')))==4


def test_cpu_only_target_helper_matches_frozen_fpnet_definition():
    import ast
    from pathlib import Path
    def definition(path):
        tree=ast.parse(Path(path).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='canonical_target'))
    assert definition('casmi_ml/np_pairtail_structures.py')==definition('casmi_ml/np_pairtail_fpnet.py')


def test_validated_pool_target_matches_full_canonical_target():
    from casmi_ml.np_pairtail_structures import canonical_record, canonical_target, validated_pool_target
    for raw in ('CCO', 'c1ccccc1O', 'CC(=O)OC1=CC=CC=C1C(=O)O'):
        key, canonical = canonical_record(raw)
        expected = canonical_target(canonical, key)
        observed = validated_pool_target(canonical, key)
        assert observed[0] == expected[0]
        np.testing.assert_array_equal(observed[1], expected[1])
    with pytest.raises(ValueError, match='identity'):
        validated_pool_target('CCO', 'AAAAAAAAAAAAAA')


def test_m3_tail_only_preserves_actual_c0_not_c1():
    from casmi_ml.np_pairtail_m3 import bind_c0_rankings, ranking_policy
    cases=[{'identity':'truth','mode':'unknown','baseline':['c1','truth'],
        'fp_ranking':['truth','c1','c0'],'analog_ranking':['truth','c1','c0'],
        'fp_margin':1.,'analog_margin':1.,'confidence':0.}]
    original={'unknown':[{'identity':'truth','ranking':[{'identity':'c0'},{'identity':'truth'}]}],'known':[]}
    bind_c0_rankings(cases,original)
    prediction=ranking_policy(cases,{'unknown':[{'identity':'truth','ranking':['truth','c1','c0']}],'known':[]},
        3,.3,'tail_only',[.1,.1])
    assert prediction['unknown'][0]['ranking'][0]=='c0'
    assert prediction['unknown'][0]['ranking'][1]=='truth'


def test_fingerprint_cached_inputs_do_not_open_targets(tmp_path):
    import json
    from casmi_ml.np_pairtail_fp_inference import InputViews
    for name,array in [('peaks',np.zeros((3,2,19),np.float32)),
                       ('mask',np.ones((3,2),bool)),('meta',np.zeros((3,4),np.float32))]:
        np.save(tmp_path/(name+'.npy'),array)
    (tmp_path/'layout.json').write_text(json.dumps({'identities':['a','b'],
        'views':[{'single':[0],'merged':[1]},{'single':[2],'merged':[2]}], 'preprocessing':{}}))
    dataset=InputViews(tmp_path,['b','a'])
    assert dataset.identities==['b','a'] and len(dataset)==4
    assert dataset.source_indices==[2,2,0,1]
    assert not (tmp_path/'target.npy').exists()
    assert set(dataset[0])=={'peaks','mask','meta','index'}


@pytest.mark.parametrize('count',[37,400,800])
def test_runtime_fp_inputs_are_dynamic_and_invariant_to_query_id_remapping(count):
    from casmi_ml.data import fit_preprocessing
    from casmi_ml.np_pairtail_fp_inference import RuntimeViews
    frame=pd.DataFrame([{'molecule_id':f'm_{i:04d}','record_id':i,
        'ms2_mzs':[50.+i*.01,80.],'ms2_normalized_intensities':[.25,1.],
        'precursor_mz':181.,'adduct':'[M+H]+','ionization_mode':'positive',
        'instrument_type':'Orbitrap','collision_energy_ev':[10.,20.]} for i in range(count)])
    preprocessing=fit_preprocessing(frame)
    original=RuntimeViews(frame,preprocessing)
    renames={f'm_{i:04d}':f'opaque_{count-i:05d}' for i in range(count)}
    changed=RuntimeViews(frame.assign(molecule_id=frame.molecule_id.map(renames)),preprocessing)
    assert len(original.identities)==len(changed.identities)==count
    reverse={key:i for i,key in enumerate(changed.identities)}
    for n,key in enumerate(original.identities):
        other=reverse[renames[key]]
        for kind in ('single','merged'):
            for left,right in zip(original.layout[n][kind],changed.layout[other][kind]):
                for a,b in zip(original.rows[left],changed.rows[right]):np.testing.assert_array_equal(a,b)


def test_runtime_features_do_not_read_truth_columns():
    from casmi_ml.np_pairtail_ranker import CandidateFeatures
    from casmi_ml.np_pairtail_structures import canonical_record
    ident,smiles=canonical_record('CCO')
    pool=pd.DataFrame([{'identity':ident,'normalized_smiles':smiles,'mass':46.041864812,'sources':'LOTUS'}])
    index=StructureIndex(pool);builder=CandidateFeatures(index,np.arange(2048))
    group=pd.DataFrame([{'precursor_mz':47.04914,'adduct':'[M+H]+','ionization_mode':'positive',
        'instrument_type':'Orbitrap','collision_energy_ev':[10.],
        'ms2_mzs':[17.],'ms2_normalized_intensities':[1.]}])
    before=builder.build(group,pool,np.zeros(2048),[])
    poisoned=group.assign(identity='SECRET_TRUTH',normalized_smiles='SECRET_ANSWER',molecule_id='fixed_400_id')
    after=builder.build(poisoned,pool,np.zeros(2048),[])
    np.testing.assert_allclose(feature_matrix(before),feature_matrix(after),equal_nan=True)


def test_disconnected_fragment_domain_is_missing_without_candidate_removal():
    from casmi_ml.np_pairtail_ranker import fragment_masses
    one,two,total=fragment_masses('CCO.C')
    assert one is None and two is None and np.isfinite(total)


def test_inference_projection_drops_dummy_answers_and_preserves_dynamic_ids(tmp_path):
    from casmi_ml.np_pairtail_deployment import runtime_frame
    rows=pd.DataFrame([{'molecule_id':i,'ms2_mzs':[50.,80.],'ms2_normalized_intensities':[.2,1.],
        'precursor_mz':181.,'adduct':'[M+H]+','ionization_mode':'positive',
        'instrument_type':'Orbitrap','collision_energy_ev':[10.],
        'normalized_smiles':'MUST_NOT_READ','inchikey14':'MUST_NOT_READ','molecular_formula':'MUST_NOT_READ'} for i in range(37)])
    path=tmp_path/'queries.parquet';rows.to_parquet(path,index=False)
    loaded=runtime_frame(path)
    assert loaded.molecule_id.tolist()==list(range(37))
    assert not {'normalized_smiles','inchikey14','molecular_formula'}&set(loaded)
    changed=rows.assign(molecule_id=rows.molecule_id.map(lambda i:1000-i));changed.to_parquet(path,index=False)
    remapped=runtime_frame(path)
    assert loaded.binned_signature.tolist()==remapped.binned_signature.tolist()
    assert loaded.record_id.tolist()==remapped.record_id.tolist()


def test_submission_canonical_duplicates_cannot_be_hidden_by_smiles_aliases():
    from casmi_ml.np_pairtail_deployment import validate_submission
    from casmi_ml.np_pairtail_structures import canonical_record
    key,_=canonical_record('CCO')
    output=pd.DataFrame({'molecule_id':['runtime-id'],'smiles':['CCO;OCC']})
    with pytest.raises(ValueError,match='duplicate canonical'):
        validate_submission(output,['runtime-id'],{'runtime-id':[key,key]})
    output.smiles=['CCO']
    assert validate_submission(output,['runtime-id'],{'runtime-id':[key]})['rows']==1


def test_frozen_deployment_uses_measurement_guard_and_actual_c0_top1():
    from casmi_ml.np_pairtail_policy import deploy_policy
    policy={'K':3,'alpha':.6,'strategy':'tail_only','thresholds':[.1,.1]}
    result=deploy_policy(['old-top','old-tail'],['afix-top','old-top'],['learned','afix-top'],
        ['fingerprint','learned','old-top'],['analog','learned'],.5,.5,.8,policy)
    assert result==['old-top','afix-top']
    result=deploy_policy(['old-top','old-tail'],['afix-top','old-top'],['learned','afix-top'],
        ['fingerprint','learned','old-top'],['analog','learned'],.5,.5,.2,policy)
    assert result[0]=='old-top' and 'learned' in result


def test_library_fp_cache_preserves_features_and_keys_complete_alias(monkeypatch):
    import casmi_ml.np_pairtail_ranker as module
    from casmi_ml.np_pairtail_structures import canonical_record
    key,smi=canonical_record('CCO')
    pool=pd.DataFrame([{'identity':key,'normalized_smiles':smi,'mass':46.041864812,'sources':'LOTUS'}])
    index=StructureIndex(pool);builder=module.CandidateFeatures(index,np.arange(2048))
    group=pd.DataFrame([{'precursor_mz':47.04914,'adduct':'[M+H]+','ionization_mode':'positive',
        'instrument_type':'Orbitrap','collision_energy_ev':[10.],
        'ms2_mzs':[17.],'ms2_normalized_intensities':[1.]}])
    calls=[];original=module.canonical_target
    def counted(*args,**kwargs):calls.append(args);return original(*args,**kwargs)
    monkeypatch.setattr(module,'canonical_target',counted)
    before=builder.build(group,pool,np.zeros(2048),[(key,smi,.5)])
    previous=len(calls)
    after=builder.build(group,pool,np.zeros(2048),[(key,smi,.5)])
    assert len(calls)==previous
    np.testing.assert_array_equal(feature_matrix(before),feature_matrix(after))
    builder.build(group,pool,np.zeros(2048),[(key,'OCC',.5)])
    assert len(calls)==previous+1


def test_training_only_explanations_preserve_selected_features_and_proposals(monkeypatch):
    import casmi_ml.np_pairtail_ranker as module
    rng=np.random.default_rng(7)
    pool=pd.DataFrame([{'identity':str(i),'normalized_smiles':f'[{i+1}CH3]O',
        'mass':100.,'sources':'COCONUT'} for i in range(600)])
    vectors=rng.integers(0,2,(600,32)).astype(np.float32)
    def make_builder():
        index=StructureIndex(pool)
        index.fingerprints={str(i):(f'[{i+1}CH3]O',vectors[i]) for i in range(600)}
        return module.CandidateFeatures(index,np.arange(32))
    group=pd.DataFrame([{'precursor_mz':101.007276,'adduct':'[M+H]+','ionization_mode':'positive',
        'instrument_type':'Orbitrap','collision_energy_ev':[10.],
        'ms2_mzs':[17.,31.],'ms2_normalized_intensities':[.2,1.]}])
    logits=rng.normal(size=32).astype(np.float32)
    calls=[];original=module.fragment_masses
    def counted(smi):calls.append(smi);return original(smi)
    monkeypatch.setattr(module,'fragment_masses',counted)
    full=make_builder().build(group,pool,logits,[])
    assert len(calls)==600
    calls.clear()
    fast=make_builder().build(group,pool,logits,[],training_token='fixed-measurements')
    selected=select_negative_rows(full,'fixed-measurements')
    np.testing.assert_array_equal(selected,select_negative_rows(fast,'fixed-measurements'))
    np.testing.assert_array_equal(selected,fast.attrs['training_explanation_rows'])
    assert len(calls)==256
    np.testing.assert_array_equal(feature_matrix(full.iloc[selected]),feature_matrix(fast.iloc[selected]))
    omitted=next(i for i in range(600) if i not in selected)
    _,labels=labeled_group(fast,pool.identity.tolist(),str(omitted),selected)
    assert labels.sum()==0
