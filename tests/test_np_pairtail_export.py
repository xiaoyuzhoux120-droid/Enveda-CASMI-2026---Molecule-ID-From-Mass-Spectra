import json
import zipfile

import pytest

from scripts.export_np_pairtail_summary import export_summary


def test_summary_export_is_truthfully_scoped_and_omits_raw_and_model_files(tmp_path):
    run=tmp_path/'actual';run.mkdir()
    (run/'run_status.json').write_text(json.dumps({'stage':'M2_failed','acceptance_opened':False}))
    (run/'seed_20261004_history.json').write_text('[{"epoch":1,"optimizer_updates":10}]')
    (run/'acceptance.parquet').write_bytes(b'private labels')
    (run/'train_cache.npy').write_bytes(b'raw spectrum cache')
    (run/'model.pt').write_bytes(b'checkpoint')
    (run/'M2_development_candidate_cases.json').write_text('[{"identity":"private case"}]')
    archive=tmp_path/'summary.zip'
    manifest=export_summary(run,archive)
    assert not manifest['full_run_archived']
    assert not manifest['models_included']
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())=={'run_status.json','seed_20261004_history.json','summary_export_manifest.json'}
        assert json.loads(z.read('run_status.json'))['stage']=='M2_failed'


def test_missing_or_truncated_actual_status_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='actual run status'):
        export_summary(tmp_path,tmp_path/'empty.zip')
    (tmp_path/'run_status.json').write_text('{"stage":')
    with pytest.raises(json.JSONDecodeError):
        export_summary(tmp_path,tmp_path/'bad.zip')
