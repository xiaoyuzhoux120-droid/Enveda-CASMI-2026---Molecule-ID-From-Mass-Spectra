"""Engineering budget recovery; retain this experiment's audited full seed."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DEST=Path(__file__).resolve().parent

def build():
    template=json.loads((ROOT/'kaggle_release_np_retrain/np_cache_retrain.ipynb').read_text())
    old_manifest=json.loads((ROOT/'kaggle_release_np_retrain/build_manifest.json').read_text())
    recovery=json.loads((ROOT/'kaggle_release_np_retrain/completed_seed_recovery.json').read_text())
    declaration=json.loads((ROOT/'kaggle_release_np_retrain/retrain_declaration.json').read_text())
    declaration.update(prior_failed_mount_budget_charge_seconds=recovery['prior_budget_charge_seconds'],
        completed_seed_reused=True,old_scientific_experiment_weights_reused=False,
        recovery_basis='this fresh experiment Version 3 selected completed-epoch binary; actual GPU replay required',
        all_three_seeds_from_scratch=False,cache_dataset_version=1)
    protocol=json.loads((ROOT/'kaggle_release_np_pairtail/protocol_fpnet_20261004.json').read_text())
    names=tuple(old_manifest['source_files'])+('casmi_ml/np_pairtail_seed_recovery.py',)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            value=(ROOT/name).read_bytes();compile(value,name,'exec');z.writestr(name,value)
        for name,value in [('protocol_fpnet_20261004.json',protocol),('retrain_declaration.json',declaration),
                           ('completed_seed_recovery.json',recovery)]:z.writestr(name,json.dumps(value))
    payload=buf.getvalue();digest=hashlib.sha256(payload).hexdigest()
    code=''.join(template['cells'][1]['source']);code=code[code.index('SUMS_SHA256 = '):]
    code=f'CODE_BASE64 = {base64.b64encode(payload).decode()!r}\nCODE_SHA256 = {digest!r}\n'+code
    runner=''.join(template['cells'][2]['source'])
    before="root=WORKING/'np_pairtail_fpnet'"
    insert='''recovery=json.loads((code_dir/'completed_seed_recovery.json').read_text())
seed_sources=[p.parent for p in INPUT.rglob('seed_20261004_history.json') if sha256(p)==recovery['files_sha256']['seed_20261004_history.json']]
completed=unique(seed_sources,'declared Version 3 complete seed')
previous_code=completed.parent/'chemistry_code/casmi_ml/np_pairtail_fpnet.py'
if sha256(previous_code)!=recovery['source_module_sha256']:raise ValueError('Recovered seed original training source mismatch')
if json.loads((completed/'data_manifest.json').read_text())['embedded_source_sha256']!=recovery['embedded_source_sha256']:raise ValueError('Recovered seed embedded source mismatch')
for name,expected in manifest['foundation_file_sha256'].items():
    if json.loads((completed/'data_manifest.json').read_text())['foundation_file_sha256'][name]!=expected:raise ValueError('Recovered seed foundation identity differs')
'''
    runner=runner.replace(before,insert+before)
    runner=runner.replace("historical_worker_seconds=declaration['historical_worker_seconds'])", "historical_worker_seconds=declaration['historical_worker_seconds'],\n    completed_seed_root=completed,completed_seed_declaration=recovery)")
    runner=runner.replace("'old_weights_reused':False", "'old_scientific_experiment_weights_reused':False,'this_experiment_completed_seed_reused':True")
    cells=[]
    for kind,content in [('markdown','# NP-PairTail budget recovery 20261005\n\nRestore the SHA-bound Version 3 completed seed after actual binary/logit GPU replay; discard partial epoch 20. Seeds 20261005 and 20261006 train fresh. Frozen protocol unchanged; reserve one observed epoch before the 45-minute cap. Prior experiment charge 3456.8 seconds; old worker cost 10528.1 separately preserved. Acceptance sealed. Internet OFF, T4 x2, June 30 Python312 container.\n'),('code',code),('code',runner)]:
        cell={'cell_type':kind,'metadata':{},'source':content.splitlines(keepends=True)}
        if kind=='code':compile(content,'recovery_notebook','exec');cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    (DEST/'np_budget_recovery.ipynb').write_text(json.dumps(template|{'cells':cells},indent=2)+'\n')
    (DEST/'build_manifest.json').write_text(json.dumps({'embedded_source_sha256':digest,'source_files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},'completed_seed_recovery':recovery,'stage':'prepared_not_executed'},indent=2)+'\n')
if __name__=='__main__':build()
