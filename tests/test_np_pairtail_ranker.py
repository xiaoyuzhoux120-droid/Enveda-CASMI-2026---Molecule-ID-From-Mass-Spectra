import time

import numpy as np
import pandas as pd
import pytest

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
