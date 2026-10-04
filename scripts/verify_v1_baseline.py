"""Verify the submitted v1 source archive without executing its notebook."""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '06e12ee720b19af2ebb562811d074d1caa91b0ec'
SHA256 = '27cdd79e735f5ce2e0c9b43eacaad507b6757ac8f1a6dfb9c292704bc8c0d7e9'
FILES = {'baseline.py', 'hybrid.py', 'casmi_ml/__init__.py',
         'casmi_ml/chemical_priors.py', 'casmi_ml/hybrid_chemistry.py'}


def verify():
    notebook = ROOT / 'kaggle_release_chemistry/notebook/chemical_priors_hybrid.ipynb'
    document = json.loads(notebook.read_text())
    source = '\n'.join(''.join(c['source']) for c in document['cells']
                       if c['cell_type'] == 'code')
    constants = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {'CODE_BASE64', 'CODE_SHA256'}):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    payload = base64.b64decode(constants['CODE_BASE64'], validate=True)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != SHA256 or constants['CODE_SHA256'] != SHA256:
        raise ValueError('Notebook archive differs from submitted v1')
    rows = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if set(archive.namelist()) != FILES or len(archive.namelist()) != len(FILES):
            raise ValueError('Unexpected v1 archive members')
        for name in sorted(FILES):
            embedded = archive.read(name)
            frozen = subprocess.check_output(['git', 'show', f'{COMMIT}:{name}'], cwd=ROOT)
            if embedded != frozen:
                raise ValueError(f'Embedded file differs from frozen commit: {name}')
            current = ROOT / name
            rows.append({'path': name, 'sha256': hashlib.sha256(embedded).hexdigest(),
                         'embedded_matches_frozen_commit': True,
                         'current_matches_frozen_commit': current.exists()
                         and current.read_bytes() == frozen})
    return {'status': 'source_provenance_verified', 'source_commit': COMMIT,
            'archive_sha256': actual, 'files': rows,
            'scope': 'Static source verification only; no full-data rerun or new Kaggle score.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    result = verify()
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded)
    print(encoded, end='')
