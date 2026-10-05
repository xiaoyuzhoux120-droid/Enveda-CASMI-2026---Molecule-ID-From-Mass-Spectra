"""Freeze M3 separately: development only, no acceptance/test predictions."""
import base64,hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DEST=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from kaggle_release_np_pairtail.build_foundation import SOURCES
EXTRA=('np_pairtail_fpnet','np_pairtail_fusion','np_pairtail_structures','np_pairtail_ranker',
       'np_pairtail_learning_runtime','np_pairtail_fp_inference','np_pairtail_policy','np_pairtail_m3','np_pairtail_acceptance','np_pairtail_deployment')

def build():
    names=(*SOURCES,*(f'casmi_ml/{n}.py' for n in EXTRA))
    protocol=json.loads((DEST/'protocol_ranker_20261005.json').read_text())
    recovery=json.loads((ROOT/'kaggle_release_np_recovery/build_manifest.json').read_text())
    assets=json.loads((DEST/'runtime_asset_manifest.json').read_text())
    payload_io=io.BytesIO()
    with zipfile.ZipFile(payload_io,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            b=(ROOT/name).read_bytes();compile(b,name,'exec');z.writestr(name,b)
        for filename,obj in [('protocol_ranker_20261005.json',protocol),('expected_m2_recovery.json',recovery),
                             ('runtime_asset_manifest.json',assets)]:z.writestr(filename,json.dumps(obj))
        z.writestr('protocol_np_pairtail_20261004.json',(ROOT/'kaggle_release_np_pairtail/protocol_np_pairtail_20261004.json').read_bytes())
    payload=payload_io.getvalue();digest=hashlib.sha256(payload).hexdigest()
    template=json.loads((ROOT/'kaggle_release_np_pairtail/notebook/np_pairtail_foundation.ipynb').read_text())
    bootstrap=''.join(template['cells'][1]['source'])
    bootstrap=bootstrap[bootstrap.index('SUMS_SHA256 = '):]
    bootstrap=bootstrap[:bootstrap.index('from casmi_ml.np_pairtail_foundation import run')]
    bootstrap='import time\nM3_PREPARATION_STARTED=time.monotonic()\n'+bootstrap
    bootstrap=f'CODE_BASE64 = {base64.b64encode(payload).decode()!r}\nCODE_SHA256 = {digest!r}\n'+bootstrap
    bootstrap+='''\n# LOTUS/LightGBM assets must match full query-independent prepared universe.
assets=json.loads((code_dir/'runtime_asset_manifest.json').read_text())
lotus_files=list(INPUT.rglob('lotus_official_pool.parquet'))
if not lotus_files:
    zipped=unique(INPUT.rglob('np_lotus_ranker_assets_20261005.zip'),'frozen LOTUS/ranker archive')
    if sha256(zipped)!='09e08596c8a340a662209f4056723b2d98b3aeb1cbf3a683e2e1fce5d2cb4fce':raise ValueError('LOTUS/ranker archive SHA mismatch')
    with zipfile.ZipFile(zipped) as archive:extract_safely(archive,WORKING/'np_ranker_assets')
    lotus_files=list((WORKING/'np_ranker_assets').rglob('lotus_official_pool.parquet'))
lotus=unique(lotus_files,'full official LOTUS pool');asset_root=lotus.parent
for name,expected in assets['files_sha256'].items():
    if sha256(asset_root/name)!=expected:raise ValueError('Ranker asset hash mismatch: '+name)
lgbwheel=unique((asset_root/'wheels').glob('lightgbm-4.7.0-*.whl'),'official pinned LightGBM')
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(lgbwheel)],check=True)
# Avoid loading native CPU OpenMP and CUDA runtimes into a single process.
subprocess.run([sys.executable,'-c',"import lightgbm; assert lightgbm.__version__=='4.7.0'; print('LightGBM',lightgbm.__version__)"],check=True)
'''
    runner='''import shutil
import pandas as pd
from casmi_ml.np_pairtail_m3 import run
foundation=unique([p.parent for p in INPUT.rglob('development_foundation_results.json') if (p.parent/'split_manifest.json').exists()],'completed foundation')
models=unique([p.parent for p in INPUT.rglob('M2_training_summary.json') if (p.parent/'M2_development_candidate_gate.json').exists()],'completed three-seed FPNet')
expected=json.loads((code_dir/'expected_m2_recovery.json').read_text())
m2manifest=json.loads((models/'data_manifest.json').read_text())
if m2manifest['embedded_source_sha256']!=expected['embedded_source_sha256']:raise ValueError('Expected completed FPNet Version 4 immutable source')
for name,value in expected['source_files'].items():
    if sha256(models.parent/'chemistry_code'/name)!=value:raise ValueError('FPNet source differs: '+name)
if json.loads((models/'run_status.json').read_text())['stage']!='M2_complete':raise ValueError('M2 failed: do not proceed to ranker')
for name,value in m2manifest['foundation_file_sha256'].items():
    if sha256(foundation/name)!=value:raise ValueError('FPNet/foundation data identity mismatch')
cache=unique([p.parent.parent for p in INPUT.rglob('layout.json') if p.parent.name=='train_cache'],'verified view cache')
original=json.loads((code_dir/'protocol_np_pairtail_20261004.json').read_text())
mapping_file=unique([p for p in INPUT.rglob('structure_identity_map.json') if sha256(p)==original['canonical_cache_sha256']],'frozen original canonical identity cache')
mapping=json.loads(mapping_file.read_text())
# Cache alias→identity values from the audited M2 pool, never predictions/truth.
oldpool=unique([p for p in INPUT.rglob('base_structure_pool.parquet') if sha256(p)==m2manifest['candidate_pool_sha256']],'verified development mass pool')
for row in pd.read_parquet(oldpool,columns=['normalized_smiles','identity']).itertuples(index=False):
    if row.normalized_smiles in mapping and mapping[row.normalized_smiles]!=row.identity:raise ValueError('Identity cache collision')
    mapping[row.normalized_smiles]=row.identity
root=WORKING/'np_pairtail_ranker';root.mkdir(exist_ok=True)
# Preserve immutable inference weights so the next self-output version has all
# required runtime artifacts without mounting two versions of one notebook.
frozen_models=root/'frozen_fpnet';frozen_models.mkdir(exist_ok=True)
for name in ['M2_training_summary.json','M2_development_candidate_gate.json','run_status.json',
             'train_fitted_statistics.json','data_manifest.json']+[f'fpnet_seed_{seed}.pt' for seed in (20261004,20261005,20261006)]:
    shutil.copyfile(models/name,frozen_models/name)

reference_mapping=root/'reference_identity_map.json';shutil.copyfile(mapping_file,reference_mapping)
merged_mapping=root/'structure_identity_map.json';merged_mapping.write_text(json.dumps(mapping,sort_keys=True))
(root/'data_manifest.json').write_text(json.dumps({'embedded_source_sha256':CODE_SHA256,'M2_data_manifest_sha256':sha256(models/'data_manifest.json'),
    'foundation_data_sha256':m2manifest['foundation_file_sha256'],'original_identity_cache_sha256':sha256(mapping_file),
    'canonical_alias_cache_pool_sha256':sha256(oldpool),
    'unneeded_draft_startup':{'budget_allowance_seconds':300.,'actual_worker_seconds':None,'code_executed':False,'session_stopped':True},
    'prior_M3_failure':{'version':5,'script_version_id':355588043,'worker_seconds':151.5,
        'reason':'unsupported training query adduct','budget_charge_seconds':151.5},
    'acceptance_opened':False},indent=2))
protocol=json.loads((code_dir/'protocol_ranker_20261005.json').read_text())
run(foundation,models,cache,TRAIN_PATH,COCONUT_PATH,CATALOG_PATH,DICTIONARY_PATH,merged_mapping,
    lotus,lotus.with_suffix('.manifest.json'),protocol,root/'base_structure_pool.parquet',root,
    preparation_seconds=time.monotonic()-M3_PREPARATION_STARTED+300.+151.5,reference_mapping_path=reference_mapping)
print('M3 finished. One sealed acceptance and inference-only release remain mandatory.',flush=True)
'''
    cells=[]
    for kind,content in [('markdown','# NP-PairTail M3 — DEVELOPMENT ONLY\n\nFrozen full-source mass candidates, four-seed LambdaRank, LOTUS increment and PairTail. Acceptance sealed. Internet OFF, T4 x2, June 30 Python312 image. Actual completed M2 Version4 required; no retraining in this stage.\n'),('code',bootstrap),('code',runner)]:
        cell={'cell_type':kind,'metadata':{},'source':content.splitlines(keepends=True)}
        if kind=='code':compile(content,'m3_notebook','exec');cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    (DEST/'np_ranker_development.ipynb').write_text(json.dumps(template|{'cells':cells},indent=2)+'\n')
    manifest={'stage':'prepared_not_executed','embedded_source_sha256':digest,'source_files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
              'ranker_protocol_sha256':hashlib.sha256((DEST/'protocol_ranker_20261005.json').read_bytes()).hexdigest(),
              'acceptance_opened':False,'competition_submission_allowed':False}
    (DEST/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':build()
