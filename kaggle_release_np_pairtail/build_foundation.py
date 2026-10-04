"""Package the M0/M1 research stage; assets are mounted, never answer lookups."""
import base64
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
SOURCES=('baseline.py','hybrid.py','casmi_ml/__init__.py','casmi_ml/data.py',
         'casmi_ml/chemical_priors.py','casmi_ml/hybrid_chemistry.py',
         'casmi_ml/v1_routing.py','casmi_ml/np_pairtail_foundation.py')


def build():
    cache=ROOT/'kaggle_release_v1opt/run_output/research_output/structure_identity_map.json'
    prior=ROOT/'kaggle_release_v1opt/run_output/chemistry_code/prior_seen_identities.json'
    cohort=ROOT/'kaggle_release_v1opt/run_output/research_output/cohort_manifest.json'
    spec=ROOT/'docs/PRD_FORWARD_NP_PAIRTAIL_RUN_20261004.md'
    protocol={'version':1,'prd_sha256':hashlib.sha256(spec.read_bytes()).hexdigest(),
              'salt':'casmi26-np-pairtail-20261004-v1:', 'rdkit_version':'2026.03.3',
              'identity_ratio':[.8,.1,.1], 'train_identity_cap':120000,
              'development_selected':2000,'acceptance_selected':3000,
              'minimum_development':1000,'minimum_acceptance':1500,
              'spectra_per_identity':{'train':8,'development':4,'acceptance':4},
              'afix_thresholds_ppm':[10,20,30], 'query_mass_window_unchanged':True,
              'afix_policy':'formula first, deterministic precursor fallback; recompute charge-1 adduct residual; exclude above threshold, preserve old fields; no adduct relabeling',
              'reference_representatives':['polarity/adduct/peaks/stable ID','polarity/adduct/instrument/peaks/stable ID'],
              'cross_split_spectrum_duplicates':'remove every owner of any exact or retrieval-bin signature shared across splits before freezing actual cohorts',
              'canonical_cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),
              'prior_seen_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),
              'prior_cohort_sha256':hashlib.sha256(cohort.read_bytes()).hexdigest(),
              'reuse':'canonical input map only, no old candidate ranks/query lookup/weights',
              'total_research_budget_seconds':14400,'data_audit_budget_seconds':1200,
              'afix_reference_budget_seconds':1500, 'pilot_queries':100,
              'stage':'M0 data preparation / M1 C0-C2 development only',
              'fpnet':{'layers':6,'hidden':512,'heads':8,'dropout':.1,
                       'seeds':[20261004,20261005,20261006],'minimum_epochs':6,'maximum_epochs':20,
                       'effective_batch':64,'seed_seconds':2700},
              'acceptance':{'repeats':10000,'seed':20261004,'unknown_delta_min':.001,
                            'unknown_ci_lower_gt':0,'known_ci_lower_min':-.001,
                            'known_top1_drop_max':.002,'known_recall25_drop_max':.002},
              'acceptance_predictions_allowed_in_foundation':False,
              'competition_submission_allowed_in_foundation':False,
              'forward_weights_domains_licenses':'pending separate verified asset manifests; no forward execution authorized by this package',
              'NPAtlas':{'enabled':False,'reason':'use eligibility not established'},'PubChem':{'enabled':False}}
    DEST.mkdir(parents=True,exist_ok=True)
    write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n')
    protocol_path=DEST/'protocol_np_pairtail_20261004.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Frozen protocol changed; create a separately named experiment')
    write(protocol_path,protocol)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name in SOURCES:
            value=(ROOT/name).read_bytes();compile(value,name,'exec');z.writestr(name,value)
        z.writestr('protocol_np_pairtail_20261004.json',json.dumps(protocol))
        if (DEST/'foundation_recovery.json').exists():
            z.writestr('foundation_recovery.json',(DEST/'foundation_recovery.json').read_bytes())
    payload=buf.getvalue();sha=hashlib.sha256(payload).hexdigest()
    original=json.loads((ROOT/'kaggle_release_chemistry/notebook/chemical_priors_hybrid.ipynb').read_text())
    bootstrap=''.join(original['cells'][1]['source']);bootstrap=bootstrap[bootstrap.index('from pathlib import Path\n'):]
    bootstrap=bootstrap.replace('from casmi_ml.hybrid_chemistry import predict, behavior_check','from casmi_ml.np_pairtail_foundation import run')
    # Research checkpoints also contain train.parquet; only a directory with
    # the competition train/test pair can supply raw competition data.
    bootstrap=bootstrap.replace('TRAIN_PATH = unique(INPUT.rglob("train.parquet"), "competition train.parquet")',
        'TRAIN_PATH = unique((p for p in INPUT.rglob("train.parquet") if (p.parent/"test.parquet").is_file()), "competition train/test pair")')
    bootstrap=bootstrap.replace('TEST_PATH = unique(INPUT.rglob("test.parquet"), "competition test.parquet")',
        'TEST_PATH = TRAIN_PATH.parent / "test.parquet"')
    code=f'CODE_BASE64 = {base64.b64encode(payload).decode()!r}\nCODE_SHA256 = {sha!r}\n'+"SUMS_SHA256 = '1563d3ed2c3a3529926c0507880c4ab39e3477265487978b496cf635ae0553fa'\nCOCONUT_SHA256 = '6d8bd9206fa576fecd2741c020ba64f5f4e60609f767bd8aad8a6628a5b87bbd'\n"+bootstrap
    runner='''protocol=json.loads((code_dir/'protocol_np_pairtail_20261004.json').read_text())
def checked_input(name,expected):
    candidates=list(Path('/kaggle/input').rglob(name))
    for path in candidates:
        if hashlib.sha256(path.read_bytes()).hexdigest()==expected:return path
    raise ValueError('Missing or hash-mismatched frozen input: '+name)
cached=checked_input('structure_identity_map.json',protocol['canonical_cache_sha256'])
prior=checked_input('prior_seen_identities.json',protocol['prior_seen_sha256'])
cohort=checked_input('cohort_manifest.json',protocol['prior_cohort_sha256'])
observed=set(json.loads(prior.read_text())['identities'])
previous=json.loads(cohort.read_text())
observed.update(x['identity'] for x in previous['selected'])
observed_path=WORKING/'observed_identity_exclusions.json'
observed_path.write_text(json.dumps(sorted(observed)))
recovery_path=code_dir/'foundation_recovery.json'
recovery=json.loads(recovery_path.read_text()) if recovery_path.exists() else {}
prior_seconds=float(recovery.get('prior_actual_run_seconds',0.))
prepared=None
if recovery:
    for p in Path('/kaggle/input').rglob('split_manifest.json'):
        previous_source=p.parent.parent/'chemistry_code/casmi_ml/np_pairtail_foundation.py'
        if previous_source.exists() and hashlib.sha256(previous_source.read_bytes()).hexdigest()==recovery['prior_foundation_module_sha256']:
            if prepared is not None:raise ValueError('More than one frozen M0 recovery checkpoint')
            prepared=p.parent
    if prepared is None:raise ValueError('Mount the canceled original Foundation V1 output; no silent data rebuild')
run(TRAIN_PATH,COCONUT_PATH,CATALOG_PATH,DICTIONARY_PATH,cached,observed_path,protocol,WORKING/'np_pairtail_foundation',prior_seconds=prior_seconds,prepared=prepared)
print('M0/M1 research only. Acceptance was not opened; FPNet is not trained; no competition submission.',flush=True)
'''
    cells=[]
    for kind,text in [('markdown','# NP-PairTail M0/M1 foundation — RESEARCH ONLY\n\nNew identity split, copy audit, AFIX and metadata representatives. Mount the completed CASMI V1 Routing Research 20261003 output as an additional input. No acceptance evaluation, FPNet training or competition submission in this stage.\n'),('code',code),('code',runner)]:
        cell={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':compile(text,'foundation_notebook','exec');cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    nb={'cells':cells,'metadata':original['metadata'],'nbformat':4,'nbformat_minor':5}
    path=DEST/'notebook/np_pairtail_foundation.ipynb';path.parent.mkdir(parents=True,exist_ok=True);write(path,nb)
    write(DEST/'build_manifest.json',{'embedded_source_sha256':sha,'notebook_bytes':path.stat().st_size,
            'source_files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES},
            'no_old_submission_or_fixed_test_ids':True,'stage':'M0/M1 research only'})


if __name__=='__main__':build()
