import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from casmi_ml.data import fit_preprocessing
from casmi_ml.np_pairtail_fpnet import (
    FPNet, canonical_target, token_features, merged_views, single_view,
    fit_target_statistics, grouped_logits, build_cache,
)


def row(smiles='CCO', identity=None, **values):
    identity = identity or Chem.MolToInchiKey(rdMolStandardize.TautomerEnumerator().Canonicalize(Chem.MolFromSmiles(smiles)))[:14]
    return {'normalized_smiles':smiles, 'identity':identity, 'record_id':1,
            'ms2_mzs':[50.,80.], 'ms2_normalized_intensities':[.25,1.],
            'precursor_mz':181., 'adduct':'[M+H]+', 'ionization_mode':'positive',
            'instrument_type':'Orbitrap', 'collision_energy_ev':[10.,20.], **values}


def test_canonical_alias_targets_and_mismatch_fail_closed():
    a='CC(=O)C'; b='C=C(O)C'
    key=Chem.MolToInchiKey(rdMolStandardize.TautomerEnumerator().Canonicalize(Chem.MolFromSmiles(a)))[:14]
    assert canonical_target(a,key)[0]==canonical_target(b,key)[0]
    np.testing.assert_array_equal(canonical_target(a,key)[1],canonical_target(b,key)[1])
    with pytest.raises(ValueError,match='identity differs'):
        canonical_target(a,'AAAAAAAAAAAAAA')


def test_empty_peak_mask_and_real_merged_loss():
    peaks,mask=token_features(np.array([]),np.array([]),np.array([]))
    assert mask.sum()==1 and np.isfinite(peaks).all()
    records=[row(),row(record_id=2,precursor_mz=204.,adduct='[M+Na]+',ms2_normalized_intensities=[1.,.25])]
    config=fit_preprocessing(pd.DataFrame(records))
    p,m,meta=merged_views(records,config)[0]
    assert len(p)==64 and meta[-2]==1 and np.isfinite(meta).all()
    # The 50Da strongest peak came from the sodium-adduct spectrum: loss154.
    assert p[0,2]==pytest.approx(154./1250)
    assert meta[-1]==pytest.approx(np.log(3.))
    # ID renaming has no effect on spectrum or metadata inputs.
    renamed=[{**r,'molecule_id':'whatever'} for r in records]
    for before,after in zip(merged_views(records,config)[0],merged_views(renamed,config)[0]):
        np.testing.assert_array_equal(before,after)


def test_train_only_bits_and_weight_cap():
    target=np.array([[1,1,0,0],[1,0,0,0],[1,0,0,1],[1,0,0,0]],np.uint8)
    bits,freq,weight=fit_target_statistics(target)
    np.testing.assert_array_equal(bits,[1,3])
    np.testing.assert_allclose(freq,[.25,.25]);np.testing.assert_allclose(weight,[3,3])
    tiny=np.zeros((100,1),np.uint8);tiny[0]=1
    assert fit_target_statistics(tiny)[2][0]==10


def test_polarity_groups_and_equal_view_fusion():
    records=[row(),row(record_id=2,ionization_mode='negative',adduct='[M-H]-')]
    assert len(merged_views(records,fit_preprocessing(pd.DataFrame(records))))==2
    single,fused=grouped_logits(np.array([[1.,3.],[3.,5.],[8.,10.]]),{'views':[{'single':[0,1],'merged':[2]}]})
    np.testing.assert_allclose(single,[[2,4]]);np.testing.assert_allclose(fused,[[5,7]])


def test_small_cache_is_identity_balanced_and_acceptance_unused(tmp_path):
    rows=[row(),row(record_id=2),row('CCC',record_id=3)]
    path=tmp_path/'training.parquet';pd.DataFrame(rows).to_parquet(path)
    dataset,config=build_cache(path,tmp_path/'cache')
    dataset.training=True;dataset.set_epoch(1)
    assert len(dataset)==2 and len(dataset.selection)==2
    for i,selected in enumerate(dataset.selection):
        assert selected in dataset.layout['views'][i]['single']+dataset.layout['views'][i]['merged']
    assert len(dataset[0]['meta'])==8+sum(len(config['categories'][c])+1 for c in config['categories'])


def test_required_transformer_is_finite_for_empty_spectrum():
    torch.set_num_threads(2)
    model=FPNet(8,17).eval()
    assert len(model.encoder.layers)==6
    mask=torch.zeros(2,64,dtype=torch.bool);mask[:,0]=True
    with torch.no_grad():
        output=model(torch.zeros(2,64,19),mask,torch.zeros(2,8))
    assert output.shape==(2,17) and torch.isfinite(output).all()
