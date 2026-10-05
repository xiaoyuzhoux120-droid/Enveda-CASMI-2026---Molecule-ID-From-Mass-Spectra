"""Prepare single sealed acceptance and inference-only release, both gated."""
import hashlib,json
from pathlib import Path
DEST=Path(__file__).resolve().parent

def build():
    original=json.loads((DEST/'np_ranker_development.ipynb').read_text())
    declaration=json.loads((DEST/'build_manifest.json').read_text())
    common='''import shutil
ranker=unique([p.parent for p in INPUT.rglob('candidate_freeze.json') if (p.parent/'run_status.json').exists()],'frozen development candidate')
if json.loads((ranker/'data_manifest.json').read_text())['embedded_source_sha256']!=CODE_SHA256:
    raise ValueError('Candidate was not produced by this exact frozen M3 implementation')
models=ranker/'frozen_fpnet'
if not (models/'M2_training_summary.json').exists():raise ValueError('Frozen three-seed inference models missing')
'''
    acceptance=common+'''from casmi_ml.np_pairtail_acceptance import run
foundation=unique([p.parent for p in INPUT.rglob('development_foundation_results.json') if (p.parent/'split_manifest.json').exists()],'completed sealed foundation')
root=WORKING/'np_pairtail_acceptance';root.mkdir(exist_ok=True)
freeze=json.loads((ranker/'candidate_freeze.json').read_text())
# Carry only immutable inference assets, never training feature/label arrays.
carry=root/'frozen_candidate';carry.mkdir(exist_ok=True)
for name in ['candidate_freeze.json','data_manifest.json','run_status.json','protocol_ranker_20261005.json',
             'structure_identity_map.json','reference_identity_map.json','base_structure_pool.parquet']:
    shutil.copyfile(ranker/name,carry/name)
shutil.copytree(models,carry/'frozen_fpnet',dirs_exist_ok=True)
(carry/freeze['ranker']).mkdir(exist_ok=True)
for name in ['ranker_manifest.json']+list(freeze['selected_boosters_sha256']):
    shutil.copyfile(ranker/freeze['ranker']/name,carry/freeze['ranker']/name)
(root/'release_source_manifest.json').write_text(json.dumps({'embedded_source_sha256':CODE_SHA256,
    'candidate_freeze_sha256':sha256(ranker/'candidate_freeze.json'),'acceptance_repeated':False}))
run(foundation,models,ranker,TRAIN_PATH,COCONUT_PATH,CATALOG_PATH,DICTIONARY_PATH,
    lotus,lotus.with_suffix('.manifest.json'),root,preparation_seconds=time.monotonic()-M3_PREPARATION_STARTED)
print('Actual sealed gate must pass before an inference-only release can run.',flush=True)
'''
    inference=common+'''from casmi_ml.np_pairtail_deployment import run
receipt_roots=[p.parent for p in INPUT.rglob('acceptance_gate.json') if (p.parent/'acceptance_summary.json').exists()]
acceptance=unique(receipt_roots,'actual one-time acceptance output')
root=WORKING/'np_pairtail_inference'
run(models,ranker,acceptance,TRAIN_PATH,TEST_PATH,COCONUT_PATH,CATALOG_PATH,DICTIONARY_PATH,
    lotus,lotus.with_suffix('.manifest.json'),root,preparation_seconds=time.monotonic()-M3_PREPARATION_STARTED)
shutil.copyfile(root/'submission.csv',WORKING/'submission.csv')
print('Inference-only CSV generated; actual Kaggle submission and scoring still required.',flush=True)
'''
    for stage,runner in [('acceptance',acceptance),('inference',inference)]:
        compile(runner,stage+'_runner','exec')
        header={'cell_type':'markdown','metadata':{},'source':[f'# NP-PairTail frozen {stage}\n',
            'Internet OFF, T4 x2, original June 30 Python312 image. Hash-qualified M3 source and candidate required.\n',
            'Acceptance runs once after development freeze. Inference never fits models or evaluates held-out truths.\n']}
        notebook=original|{'cells':[header,original['cells'][1],{'cell_type':'code','metadata':{},
            'source':runner.splitlines(keepends=True),'outputs':[],'execution_count':None}]}
        path=DEST/f'np_gated_{stage}.ipynb';path.write_text(json.dumps(notebook,indent=2)+'\n')
    (DEST/'gated_release_build_manifest.json').write_text(json.dumps({'stage':'prepared_not_executed',
        'embedded_source_sha256':declaration['embedded_source_sha256'],
        'notebooks_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.glob('np_gated_*.ipynb')},
        'acceptance_opened':False,'competition_submitted':False},indent=2)+'\n')
if __name__=='__main__':build()
