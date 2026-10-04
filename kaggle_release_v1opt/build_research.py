"""Package frozen V1 replay and routing research; never a competition entry."""
import base64
import hashlib
import io
import json
import re
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path(__file__).resolve().parent
SOURCES = ('baseline.py', 'hybrid.py', 'casmi_ml/__init__.py',
           'casmi_ml/chemical_priors.py', 'casmi_ml/hybrid_chemistry.py',
           'casmi_ml/v1_routing.py', 'casmi_ml/v1_research.py')


def build():
    from casmi_ml.v1_routing import CONFIGS
    seen = set()
    paths = list((ROOT / 'artifacts').rglob('*.json'))
    paths += [ROOT / 'kaggle_release_gan/run_output' / name for name in
              ['acceptance_cases.json', 'known_acceptance_cases.json', 'prepared_manifest.json']]
    for path in paths:
        if path.exists():
            seen.update(re.findall(r'\b[A-Z]{14}\b', path.read_text()))
    previous = json.loads((ROOT / 'kaggle_release_gan/run_output/prepared_manifest.json').read_text())
    prior = {'identities': sorted(seen), 'v2_excluded_identities': previous['excluded_identities'],
             'scope': 'Exclude all recoverable previously observed identity keys, all old raw train/holdout buckets, first 2000 old development raw keys, and reconstructed v2 dev/acceptance selections. No old neural model is used. Unretained historical query records cannot be exhaustively audited.'}
    protocol = {'version': 1, 'namespace': 'casmi-v1-routing-20261003-v1:',
                'configs': list(CONFIGS), 'development_selected': 500, 'acceptance_selected': 1000,
                'maximum_spectra_per_identity': 2, 'mass_range': [157, 1159],
                'quality': 'supported adduct; precursor error <=30ppm when known; nonempty vector',
                'research_budget_seconds': 7200, 'ranking_pilot': '50 queries; <=6 seconds/query',
                'development_selection': 'maximize unknown MRR subject to known MRR >=B0-.001 and top1 >=B0-.002; baseline/simpler config on ties',
                'acceptance': 'only frozen candidate vs B0; unknown delta>=.001 and CI95 lower>0; known CI95 lower>=-.001 and Top1 delta>=-.002',
                'bootstrap': {'repeats': 10000, 'seed': 20261003},
                'fallback': 'retain B0 if not accepted; no scientific upgrade submission',
                'prior_seen_count': len(seen),
                'prior_sha256': hashlib.sha256(json.dumps(prior, sort_keys=True).encode()).hexdigest(),
                'research_notebook_must_not_be_submitted': True}
    RELEASE.mkdir(parents=True, exist_ok=True)
    protocol_path = RELEASE / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Protocol changed; start a new frozen experiment')
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    original = json.loads((ROOT / 'kaggle_release_chemistry/notebook/chemical_priors_hybrid.ipynb').read_text())
    bootstrap = ''.join(original['cells'][1]['source'])
    bootstrap = bootstrap[bootstrap.index('from pathlib import Path\n'):]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            data = (ROOT / name).read_bytes()
            compile(data, name, 'exec')
            archive.writestr(name, data)
        archive.writestr('prior_seen_identities.json', json.dumps(prior, sort_keys=True))
        archive.writestr('protocol.json', json.dumps(protocol, sort_keys=True))
        with zipfile.ZipFile('/Users/jimmy/Downloads/results.zip') as old:
            archive.writestr('expected_v1.csv', old.read('submission.csv'))
    payload = buffer.getvalue()
    archive_sha = hashlib.sha256(payload).hexdigest()
    constants = (f'CODE_BASE64 = {base64.b64encode(payload).decode()!r}\n'
                 f'CODE_SHA256 = {archive_sha!r}\n'
                 "SUMS_SHA256 = '1563d3ed2c3a3529926c0507880c4ab39e3477265487978b496cf635ae0553fa'\n"
                 "COCONUT_SHA256 = '6d8bd9206fa576fecd2741c020ba64f5f4e60609f767bd8aad8a6628a5b87bbd'\n")
    replay = '''import pandas as pd, time
start = time.monotonic()
baseline, baseline_report = predict(DATA_DIR,COCONUT_PATH,CATALOG_PATH,DICTIONARY_PATH,
                                  WORKING/'baseline_replay.csv')
expected = pd.read_csv(code_dir/'expected_v1.csv').set_index('molecule_id').sort_index()
actual = baseline.set_index('molecule_id').sort_index()
if not actual.equals(expected):
    raise AssertionError('Visible V1 replay differs from the original full submission')
replay_report = {'matched_original_all_rows':True,'molecules':len(actual),
                 'seconds':time.monotonic()-start,'source_sha256':CODE_SHA256}
(WORKING/'baseline_replay_report.json').write_text(json.dumps(replay_report,indent=2))
print('BASELINE_REPLAY',json.dumps(replay_report),flush=True)
'''
    research = '''from casmi_ml.v1_research import run
prior = json.loads((code_dir/'prior_seen_identities.json').read_text())
protocol = json.loads((code_dir/'protocol.json').read_text())
result = run(TRAIN_PATH,COCONUT_PATH,CATALOG_PATH,DICTIONARY_PATH,
             WORKING/'research_output',prior,protocol)
print('V1_ROUTING_ACCEPTANCE',json.dumps(result),flush=True)
print('Research complete. This notebook must not be submitted to the competition.',flush=True)
'''
    def cell(kind, source):
        value = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            compile(source, 'v1_research_notebook', 'exec')
            value.update(execution_count=None, outputs=[])
        return value
    document = {'cells': [cell('markdown', '# V1 routing research — NOT a competition submission\n\nReplays all original visible V1 answers before independent, predeclared routing controls. No GAN, neural training or generated graph. Acceptance never selects parameters. Historical exclusion reconstruction has documented limits. A separate inference-only notebook is required for any new competition submission.\n'),
                         cell('code', constants + bootstrap), cell('code', replay), cell('code', research)],
                'metadata': original['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
    output = RELEASE / 'notebook/v1_routing_research.ipynb'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=1) + '\n')
    manifest = {'embedded_source_sha256': archive_sha, 'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                'notebook_bytes': output.stat().st_size, 'prior_seen_count': len(seen),
                'source_files': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}}
    (RELEASE / 'build_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    build()
