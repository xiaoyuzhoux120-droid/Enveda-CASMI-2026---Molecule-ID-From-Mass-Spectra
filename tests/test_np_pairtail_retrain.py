import hashlib
import json
import numpy as np
import pytest


def fixture_cache(tmp_path):
    source=tmp_path/'train.parquet';source.write_bytes(b'frozen-source')
    root=tmp_path/'train_cache';root.mkdir()
    layout={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'identities':['A','B'],'views':[{'single':[0],'merged':[1]},{'single':[2],'merged':[3]}],
        'preprocessing':{'max_peaks':64}}
    (root/'layout.json').write_text(json.dumps(layout))
    arrays={'peaks':np.zeros((4,64,19),np.float32),'mask':np.ones((4,64),bool),
        'meta':np.zeros((4,8),np.float32),'owner':np.array([0,0,1,1],np.int32),
        'target':np.zeros((2,2048),np.uint8)}
    for k,v in arrays.items():np.save(root/f'{k}.npy',v)
    (root/'canonical_targets.parquet').write_bytes(b'preserved-cache-target-asset')
    return source,root


def test_verified_cache_preserves_views_and_records_every_hash(tmp_path):
    from casmi_ml.np_pairtail_fpnet import verified_cached_views
    source,root=fixture_cache(tmp_path)
    cached,audit=verified_cached_views(root,source,True)
    assert audit['identities']==2 and audit['views']==4
    assert len(audit['files'])==7 and cached.training
    assert np.array_equal(cached.arrays['owner'],[0,0,1,1])


@pytest.mark.parametrize('corruption',['source','owner','features','target','duplicate'])
def test_verified_cache_rejects_stale_or_corrupt_inputs(tmp_path,corruption):
    from casmi_ml.np_pairtail_fpnet import verified_cached_views
    source,root=fixture_cache(tmp_path)
    if corruption=='source':source.write_bytes(b'changed')
    elif corruption=='owner':np.save(root/'owner.npy',np.array([0,1,1,1]))
    elif corruption=='features':
        v=np.load(root/'peaks.npy');v[0,0,0]=np.nan;np.save(root/'peaks.npy',v)
    elif corruption=='target':
        v=np.load(root/'target.npy');v[0,0]=2;np.save(root/'target.npy',v)
    else:
        p=root/'layout.json';v=json.loads(p.read_text());v['views'][0]['merged']=[0];p.write_text(json.dumps(v))
    with pytest.raises(ValueError):verified_cached_views(root,source)


def test_retrain_artifact_keeps_scientific_protocol_and_old_failure_freeze():
    import base64,io,zipfile
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    notebook=json.loads((root/'kaggle_release_np_retrain/np_cache_retrain.ipynb').read_text())
    code=''.join(notebook['cells'][1]['source']); lines=code.splitlines()
    import ast
    payload=base64.b64decode(ast.literal_eval(lines[0].split('=',1)[1].strip()))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        protocol=json.loads(archive.read('protocol_fpnet_20261004.json'))
        declaration=json.loads(archive.read('retrain_declaration.json'))
    assert protocol==json.loads((root/'kaggle_release_np_pairtail/protocol_fpnet_20261004.json').read_text())
    assert declaration['historical_worker_seconds']==10528.1 and declaration['all_three_seeds_from_scratch']
    assert not declaration['old_weights_reused']
    old=json.loads((root/'kaggle_release_np_pairtail/fpnet_build_manifest.json').read_text())
    assert old['embedded_source_sha256']=='e77ab46e968c298b2504726fc639bc5292bea8446f104100907832eef22ef85e'
    runner=''.join(notebook['cells'][2]['source'])
    assert 'prepare_base_structure_pool(' not in runner and "experiment_prior_seconds=declaration['prior_failed_mount_budget_charge_seconds']" in runner
