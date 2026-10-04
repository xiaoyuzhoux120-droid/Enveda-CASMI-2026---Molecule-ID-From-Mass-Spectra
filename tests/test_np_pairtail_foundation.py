import numpy as np
import pandas as pd

from casmi_ml.np_pairtail_foundation import (
    partition, order, reference_mass, mass_keep, representative_priority,
    spectrum_signature, RepresentativeHybrid,
)
from casmi_ml.v1_routing import CONFIGS


def test_formula_mass_survives_bad_precursor_and_exposes_anomaly():
    row={'molecular_formula':'C6H12O6','precursor_mz':999.,'adduct':'[M+H]+','precursor_error_ppm':0.}
    info=reference_mass(row)
    assert info['mass_source']=='formula'
    assert abs(info['mass']-180.063388)<.00001
    assert abs(info['recomputed_ppm'])>30
    assert mass_keep(info,30,baseline=True)
    assert not mass_keep(info,30)


def test_fallback_deterministic_and_unsupported_missing():
    info=reference_mass({'molecular_formula':'Bad','precursor_mz':181.070665,'adduct':'[M+H]+'})
    assert info['mass_source']=='precursor_adduct'
    assert mass_keep(info,10)
    missing=reference_mass({'molecular_formula':'Bad','precursor_mz':181.,'adduct':'unsupported'})
    assert missing['mass_source']=='unavailable'
    assert not mass_keep(missing,30)


def test_corrupted_provided_error_recomputed_independently():
    info=reference_mass({'molecular_formula':'C6H12O6','precursor_mz':181.070665,'adduct':'[M+H]+','precursor_error_ppm':1000.})
    assert not mass_keep(info,30,True)
    assert mass_keep(info,10)


def test_split_and_record_order_ignore_test_ids():
    assert partition('ABCDEFGHIJKLMN') in ['train','development','acceptance']
    assert order('ABCDEFGHIJKLMN')==order('ABCDEFGHIJKLMN')
    assert order('ABCDEFGHIJKLMN')!=order('ABCDEFGHIJKLMN','spectrum')


def test_instrument_preference_only_matches_query():
    query=pd.DataFrame({'ionization_mode':['positive'],'adduct':['[M+H]+'],'instrument_type':['Orbitrap']})
    a={'ionization_mode':'positive','adduct':'[M+H]+','instrument_type':'timsTOF','valid_peaks':100,'record_id':1}
    b={**a,'instrument_type':'Orbitrap','valid_peaks':10,'record_id':2}
    assert representative_priority(b,query,True)<representative_priority(a,query,True)
    assert representative_priority(a,query,False)<representative_priority(b,query,False)


def test_numerical_copy_signatures_ignore_annotation():
    r={'ms2_mzs':[50.,80.],'ms2_normalized_intensities':[.5,1.],'molecule_id':'original'}
    assert spectrum_signature(r)==spectrum_signature({**r,'molecule_id':'renamed'})


def test_real_representative_engine_id_remapping():
    from baseline import vectorize
    r=(180.063388,'ABCDEFGHIJKLMN','OCC(O)C(O)C(O)C(O)CO',vectorize([50.,80.],[.5,1.]))
    meta={'identity':'ABCDEFGHIJKLMN','record_id':1,'adduct':'[M+H]+','ionization_mode':'positive','instrument_type':'Orbitrap','valid_peaks':2}
    coconut=pd.DataFrame(columns=['canonical_smiles','exact_mass','inchikey'])
    # Catalog disabled matches the original analog pool, with no chemistry rules.
    engine=RepresentativeHybrid([r],[meta],coconut,None,(),True)
    q=pd.DataFrame({'precursor_mz':[181.070665],'adduct':['[M+H]+'],'ionization_mode':['positive'],
                    'instrument_type':['Orbitrap'],'ms2_mzs':[[50.,80.]],'ms2_normalized_intensities':[[.5,1.]],'molecule_id':['m_original']})
    before=engine.controls(q,[CONFIGS[0]])[0]
    after=engine.controls(q.assign(molecule_id='arbitrary-name'),[CONFIGS[0]])[0]
    assert before==after
    assert len(before['B0'])==1
