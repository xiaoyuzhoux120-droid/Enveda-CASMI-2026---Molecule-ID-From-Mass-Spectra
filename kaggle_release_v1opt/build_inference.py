"""Build a small inference-only notebook from the frozen acceptance decision."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from casmi_ml.v1_routing import CONFIGS

ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path(__file__).resolve().parent
SOURCES = ('baseline.py', 'hybrid.py', 'casmi_ml/__init__.py',
           'casmi_ml/chemical_priors.py', 'casmi_ml/hybrid_chemistry.py',
           'casmi_ml/v1_routing.py')


def build(report_path):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    config = report['release_config']
    if config not in CONFIGS:
        raise ValueError('Acceptance returned an unknown frozen configuration')
    if not report['accepted'] and config != CONFIGS[0]:
        raise ValueError('Unaccepted scientific change cannot be released')
    if report['accepted'] and (config != report['selected'] or config == CONFIGS[0]):
        raise ValueError('Release is not the frozen accepted candidate')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            data = (ROOT / name).read_bytes()
            compile(data, name, 'exec')
            archive.writestr(name, data)
    payload = buffer.getvalue()
    archive_sha = hashlib.sha256(payload).hexdigest()
    original = json.loads((ROOT / 'kaggle_release_chemistry/notebook/chemical_priors_hybrid.ipynb').read_text())
    bootstrap = ''.join(original['cells'][1]['source'])
    bootstrap = bootstrap[bootstrap.index('from pathlib import Path\n'):]
    bootstrap = bootstrap.replace('from casmi_ml.hybrid_chemistry import predict, behavior_check',
                                  'from casmi_ml.v1_routing import inference')
    constants = (f'CODE_BASE64 = {base64.b64encode(payload).decode()!r}\n'
                 f'CODE_SHA256 = {archive_sha!r}\n'
                 "SUMS_SHA256 = '1563d3ed2c3a3529926c0507880c4ab39e3477265487978b496cf635ae0553fa'\n"
                 "COCONUT_SHA256 = '6d8bd9206fa576fecd2741c020ba64f5f4e60609f767bd8aad8a6628a5b87bbd'\n"
                 f'CONFIG = {config!r}\n')
    prediction = '''submission, report = inference(DATA_DIR,COCONUT_PATH,CATALOG_PATH,DICTIONARY_PATH,
                               WORKING/'submission.csv',CONFIG)
print('V1_INFERENCE_RESULT',json.dumps(report),flush=True)
display(submission.head())
'''
    def cell(kind, source):
        value = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            compile(source, 'v1_inference_notebook', 'exec')
            value.update(execution_count=None, outputs=[])
        return value
    decision_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    text = ('# CASMI V1 optimized routing — inference only\n\n'
            f'Frozen routing policy: `{config["name"]}`. Research acceptance passed: `{report["accepted"]}`. '
            f'Acceptance artifact SHA256: `{decision_sha}`. '
            'This notebook loads actual mounted test spectra and performs retrieval and chemical ranking only. '
            'No GAN, neural training, graph generation, development selection or acceptance is run during scoring. '
            'It contains neither visible predictions nor query IDs. '
            'If the acceptance decision retained B0, this is an engineering release of the original V1 policy, not a scientifically improved model.\n')
    document = {'cells': [cell('markdown', text), cell('code', constants + bootstrap), cell('code', prediction)],
                'metadata': original['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
    output = RELEASE / 'notebook/v1_optimized_inference.ipynb'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=1) + '\n')
    manifest = {'embedded_source_sha256': archive_sha, 'acceptance_sha256': decision_sha,
                'config': config, 'scientific_upgrade_accepted': report['accepted'],
                'research_source_commit': '5586d3e', 'research_script_version_id': 355064584,
                'notebook_bytes': output.stat().st_size,
                'source_files': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},
                'research_or_visible_answers_included': False}
    (RELEASE / 'inference_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('acceptance_report', type=Path)
    build(parser.parse_args().acceptance_report)
