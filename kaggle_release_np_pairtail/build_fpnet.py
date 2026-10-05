"""Freeze the M2 GPU stage separately from the CPU foundation audit."""
import base64
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from kaggle_release_np_pairtail.build_foundation import SOURCES


def build():
    protocol = json.loads((DEST/'protocol_np_pairtail_20261004.json').read_text())
    protocol['M2_predeclared_details'] = {
        'max_peaks':64, 'pos_weight_cap':10, 'bit_selection':'positive and negative identities both present in train',
        'sampling':'one uniformly sampled identity-view per identity/epoch; half single, half merged',
        'merged_peak_bin_da':.001, 'mixed_adduct_instrument_metadata':'missing',
        'merged_losses':'preserve strongest original peak precursor-minus-fragment; never synthetic average precursor',
        'pool':'C0 structure sources only for FR-11; LOTUS is a separate M3/C4 increment',
        'training_microbatch':'64 split32/card if two GPUs; 32 accumulate2 if one GPU',
        'acceptance_parquet_read':False, 'cumulative_foundation_budget_carried':True,
        'checkpoint_selection':'development fused weighted BCE',
        'random_ranking_seed':20261004, 'conditional_candidate_bootstrap':10000,
        'original_M0_checkpoint_script_version':355247134,
        'completed_foundation_required_script_version':355258706,
        'original_M0_checkpoint_source_sha256':'e094ea8ab49a65fb2fb81527f089cc161b7ee703389ec64c830edd560747fad8',
        'completed_foundation_embedded_source_sha256':'279cdac18675a3b0c51690af188cacde3d5fbe0f5b7ae3d0fa92e87ec049a1f1',
        'upstream_incomplete_or_failed':'refuse M2, retain failure',
        'resume':'complete epoch checkpoint retained, no half-epoch weight used',
    }
    path=DEST/'protocol_fpnet_20261004.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('M2 protocol already frozen; use separate experiment for scientific changes')
    path.write_text(json.dumps(protocol,indent=2)+'\n')
    names=(*SOURCES,'casmi_ml/np_pairtail_fpnet.py')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            content=(ROOT/name).read_bytes();compile(content,name,'exec');archive.writestr(name,content)
        archive.writestr('protocol_fpnet_20261004.json',json.dumps(protocol))
    payload=buffer.getvalue();sha=hashlib.sha256(payload).hexdigest()
    original=json.loads((DEST/'notebook/np_pairtail_foundation.ipynb').read_text())
    bootstrap=''.join(original['cells'][1]['source'])
    bootstrap=bootstrap[bootstrap.index('SUMS_SHA256 = '):]
    bootstrap=bootstrap.replace('from casmi_ml.np_pairtail_foundation import run',
        'from casmi_ml.np_pairtail_fpnet import run, prepare_base_structure_pool, sha256')
    bootstrap=f'CODE_BASE64 = {base64.b64encode(payload).decode()!r}\nCODE_SHA256 = {sha!r}\n'+bootstrap
    frozen_sources=json.loads((DEST/'build_manifest.json').read_text())['source_files']
    runner='''import time
preparation_started=time.monotonic()
protocol=json.loads((code_dir/'protocol_fpnet_20261004.json').read_text())
roots=[p.parent for p in Path('/kaggle/input').rglob('development_foundation_results.json') if (p.parent/'split_manifest.json').exists()]
if len(roots)!=1:raise ValueError('Mount exactly the completed frozen Cached Foundation Version 2 output')
foundation=roots[0]
frozen_sources=FROZEN_FOUNDATION_SOURCE_SHA256
for name,expected in frozen_sources.items():
    p=foundation.parent/'chemistry_code'/name
    if not p.exists() or sha256(p)!=expected:raise ValueError('Mounted foundation source hash mismatch: '+name)
if json.loads((foundation/'run_status.json').read_text())['stage']!='M1_foundation_complete':
    raise ValueError('M1 did not finish; GPU training refused')
if json.loads((foundation/'protocol_np_pairtail_20261004.json').read_text())['prd_sha256']!=protocol['prd_sha256']:
    raise ValueError('Foundation PRD hash differs')
mapping_paths=[p for p in Path('/kaggle/input').rglob('structure_identity_map.json') if hashlib.sha256(p.read_bytes()).hexdigest()==protocol['canonical_cache_sha256']]
if not mapping_paths:raise ValueError('Mount frozen identity cache from the prior V1 research output')
mapping=json.loads(mapping_paths[0].read_text())
pool=WORKING/'base_structure_pool.parquet'
prior_seconds=json.loads((foundation/'runtime_profile.json').read_text())[-1]['elapsed_seconds']
deadline=preparation_started+protocol['total_research_budget_seconds']-prior_seconds
prepare_base_structure_pool(TRAIN_PATH,COCONUT_PATH,CATALOG_PATH,mapping,pool,deadline=deadline)
root=WORKING/'np_pairtail_fpnet'
root.mkdir(exist_ok=True)
data_manifest={'train_source_sha256':sha256(TRAIN_PATH),'foundation_file_sha256':{
    name:sha256(foundation/name) for name in ['train.parquet','development.parquet','split_manifest.json','protocol_np_pairtail_20261004.json']},
    'acceptance_file_read':False,'candidate_pool_sha256':sha256(pool),'embedded_source_sha256':CODE_SHA256}
(root/'data_manifest.json').write_text(json.dumps(data_manifest,indent=2))
run(foundation,root,protocol,pool,preparation_seconds=time.monotonic()-preparation_started)
print('M2 only; ranker, forward, frozen acceptance and competition release remain gated.',flush=True)
'''
    runner=runner.replace('FROZEN_FOUNDATION_SOURCE_SHA256',repr(frozen_sources))
    cells=[]
    for kind,content in [('markdown','# NP-PairTail M2 FPNet — RESEARCH ONLY\n\nGPU T4 x2, Internet OFF. Mount frozen Cached Foundation V2 and prior V1 Routing Research output. No acceptance predictions or competition submission.\n'),('code',bootstrap),('code',runner)]:
        cell={'cell_type':kind,'metadata':{},'source':content.splitlines(keepends=True)}
        if kind=='code':compile(content,'fpnet_notebook','exec');cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    notebook=DEST/'notebook/np_pairtail_fpnet.ipynb'
    notebook.write_text(json.dumps({'cells':cells,'metadata':original['metadata'],'nbformat':4,'nbformat_minor':5},indent=2)+'\n')
    (DEST/'fpnet_build_manifest.json').write_text(json.dumps({'stage':'M2 not executed',
        'embedded_source_sha256':sha,'notebook_bytes':notebook.stat().st_size,
        'source_files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        'no_old_submission_or_fixed_test_ids':True},indent=2)+'\n')


if __name__=='__main__':build()
