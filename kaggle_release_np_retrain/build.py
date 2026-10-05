"""New user-authorized cache-reuse retraining; preserve the failed freeze."""
import base64, hashlib, io, json, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from kaggle_release_np_pairtail.build_foundation import SOURCES

def build():
    original=json.loads((ROOT/'kaggle_release_np_pairtail/notebook/np_pairtail_fpnet.ipynb').read_text())
    protocol=json.loads((ROOT/'kaggle_release_np_pairtail/protocol_fpnet_20261004.json').read_text())
    recovery={'experiment':'np-cache-retrain-20261005-v1','authorization':'user: 重新训练并提交kaggle',
        'new_experiment_budget_seconds':14400,'historical_worker_seconds':10528.1,
        'failed_input_script_version':355386045,'failed_embedded_source_sha256':'e77ab46e968c298b2504726fc639bc5292bea8446f104100907832eef22ef85e',
        'failed_fpnet_source_sha256':'b256662a344a67135b282d21e96a1887b1a4cdd3fcf0bb12df6b6e0678560ba3',
        'expected_pool_sha256':'98add466d604fc6574c046aa7eaa600e8beff2a207a331d20d139bc3937abece',
        'scientific_protocol_unchanged':True,'all_three_seeds_from_scratch':True,
        'old_weights_reused':False,'old_budget_erased':False,'acceptance_opened':False,
        'cache_dataset':'xiaoyuzhoux120/casmi-np-frozen-fpnet-view-cache-20261005',
        'failed_retrain_script_version':355524627,
        'prior_failed_mount_budget_charge_seconds':616.8,
        'abi_failed_retrain_script_version':355532778,
        'abi_failed_worker_seconds':16.8,
        'required_container_environment_date':'2026-06-30',
        'mount_failure_charge_basis':'conservative budget allowance, not a claim of measured worker time'}
    names=(*SOURCES,'casmi_ml/np_pairtail_fpnet.py')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            data=(ROOT/name).read_bytes(); compile(data,name,'exec');archive.writestr(name,data)
        archive.writestr('protocol_fpnet_20261004.json',json.dumps(protocol))
        archive.writestr('retrain_declaration.json',json.dumps(recovery))
    payload=buffer.getvalue(); digest=hashlib.sha256(payload).hexdigest()
    bootstrap=''.join(original['cells'][1]['source'])
    bootstrap=bootstrap[bootstrap.index('SUMS_SHA256 = '):bootstrap.index('CATALOG_PATH = ')]
    bootstrap=f'CODE_BASE64 = {base64.b64encode(payload).decode()!r}\nCODE_SHA256 = {digest!r}\n'+bootstrap
    runner='''import time
from casmi_ml.np_pairtail_fpnet import run, sha256
started=time.monotonic()
protocol=json.loads((code_dir/'protocol_fpnet_20261004.json').read_text())
declaration=json.loads((code_dir/'retrain_declaration.json').read_text())
roots=[p.parent for p in INPUT.rglob('development_foundation_results.json') if (p.parent/'split_manifest.json').exists()]
foundation=unique(roots,'completed frozen foundation output')
if json.loads((foundation/'run_status.json').read_text())['stage']!='M1_foundation_complete':raise ValueError('Foundation incomplete')
original=json.loads((foundation/'protocol_np_pairtail_20261004.json').read_text())
for key in original:
    if original[key]!=protocol[key]:raise ValueError('Frozen scientific protocol differs: '+key)
cache_roots=[p.parent.parent for p in INPUT.rglob('layout.json') if p.parent.name=='train_cache' and (p.parent.parent/'data_manifest.json').exists()]
cached=unique(cache_roots,'failed Version 2 complete view caches')
manifest=json.loads((cached/'data_manifest.json').read_text())
if manifest['embedded_source_sha256']!=declaration['failed_embedded_source_sha256']:raise ValueError('Wrong failed source version mounted')
old_code=cached.parent/'chemistry_code/casmi_ml/np_pairtail_fpnet.py'
if sha256(old_code)!=declaration['failed_fpnet_source_sha256']:raise ValueError('Failed source binary hash mismatch')
if set(manifest['foundation_file_sha256'])!={'train.parquet','development.parquet','split_manifest.json','protocol_np_pairtail_20261004.json'}:raise ValueError('Unexpected foundation manifest scope')
for name,expected in manifest['foundation_file_sha256'].items():
    if name=='acceptance.parquet':raise ValueError('Acceptance must remain sealed')
    if sha256(foundation/name)!=expected:raise ValueError('Foundation file differs: '+name)
pool=cached.parent/'base_structure_pool.parquet'
if sha256(pool)!=declaration['expected_pool_sha256'] or manifest['candidate_pool_sha256']!=declaration['expected_pool_sha256']:raise ValueError('Frozen development pool differs')
root=WORKING/'np_pairtail_fpnet'
root.mkdir(exist_ok=True)
(root/'retrain_declaration.json').write_text(json.dumps(declaration,indent=2))
(root/'data_manifest.json').write_text(json.dumps(manifest | {'embedded_source_sha256':CODE_SHA256,'cache_origin_source_sha256':manifest['embedded_source_sha256'],'cache_input_path':str(cached),'old_weights_reused':False},indent=2))
run(foundation,root,protocol,pool,preparation_seconds=time.monotonic()-started,
    cache_root=cached,experiment_prior_seconds=declaration['prior_failed_mount_budget_charge_seconds'],historical_worker_seconds=declaration['historical_worker_seconds'])
print('M2 finished. Acceptance and competition release still require downstream frozen gates.',flush=True)
'''
    cells=[]
    for kind,content in [('markdown','# NP-PairTail cached retraining 20261005\n\nFresh three seeds; frozen science unchanged. Reuse verified failed V2 view/pool caches, never old weights. New four-hour experiment budget; historical 10528.1 seconds preserved. Internet OFF, T4 x2. Acceptance remains sealed.\n'),('code',bootstrap),('code',runner)]:
        c={'cell_type':kind,'metadata':{},'source':content.splitlines(keepends=True)}
        if kind=='code':compile(content,'retrain_notebook','exec');c.update(execution_count=None,outputs=[])
        cells.append(c)
    (DEST/'np_cache_retrain.ipynb').write_text(json.dumps({'cells':cells,'metadata':original['metadata'],'nbformat':4,'nbformat_minor':5},indent=2)+'\n')
    (DEST/'retrain_declaration.json').write_text(json.dumps(recovery,indent=2)+'\n')
    (DEST/'build_manifest.json').write_text(json.dumps({'stage':'prepared_not_executed','embedded_source_sha256':digest,'source_files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},'scientific_protocol_sha256':hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()},indent=2)+'\n')

if __name__=='__main__':build()
