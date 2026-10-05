from casmi_ml.np_pairtail_forward import forward_applicability, forward_coverage_gate


def fixture():
    manifest=dict(weight_usage_verified=True,training_identity_audit_passed=True,
        energy_units_verified=True,runtime_verified=True,maximum_heavy_atoms=100,
        supported_adducts=['[M+H]+'],supported_polarities=['positive'],
        supported_instruments=['QTOF','Orbitrap'],instrument_aliases={},
        precursor_range=[60,995.556],collision_energy_range=[0,358],collision_energy_unit='eV')
    measurement=dict(adduct='[M+H]+',ionization_mode='positive',instrument_type='QTOF',
                     precursor_mz=200.,collision_energy=30.,collision_energy_unit='eV')
    return measurement,manifest


def test_instrument_and_energy_are_never_silently_remapped():
    measurement,manifest=fixture()
    assert forward_applicability(measurement,'CCO',manifest)['eligible']
    for change in ({'instrument_type':'timsTOF'},{'collision_energy_unit':'NCE'},
                   {'adduct':'[M+Na]+'},{'collision_energy':None}):
        result=forward_applicability(measurement|change,'CCO',manifest)
        assert not result['eligible']
        assert result['score'] is None
        assert not result['model_executed']


def test_public_checkpoint_is_not_proof_of_strict_training_isolation():
    measurement,manifest=fixture()
    result=forward_applicability(measurement,'CCO',manifest|{'training_identity_audit_passed':False})
    assert not result['eligible']
    assert 'training_identity_audit_passed is not verified' in result['reasons']


def test_partial_outputs_cannot_be_presented_as_confirmatory_full_cascade():
    assert forward_coverage_gate(range(100),range(80))['enabled_in_confirmatory']
    assert not forward_coverage_gate(range(100),range(80),cascade=True)['enabled_in_confirmatory']
    assert not forward_coverage_gate([],[])['enabled_in_confirmatory']
