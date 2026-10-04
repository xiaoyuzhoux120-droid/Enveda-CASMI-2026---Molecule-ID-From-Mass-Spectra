import ast
import base64
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest


def builder(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'kaggle_release_v1opt/build_inference.py'
    spec = importlib.util.spec_from_file_location('v1_release_builder_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RELEASE = tmp_path
    return module


def test_unaccepted_candidate_cannot_be_packaged(tmp_path):
    module = builder(tmp_path)
    path = tmp_path / 'synthetic_decision.json'
    path.write_text(json.dumps({'accepted': False, 'selected': module.CONFIGS[1],
                                'release_config': module.CONFIGS[1]}))
    with pytest.raises(ValueError, match='Unaccepted scientific change'):
        module.build(path)


def test_release_contains_no_research_data_or_visible_answer_lookup(tmp_path):
    module = builder(tmp_path)
    path = tmp_path / 'synthetic_decision.json'
    path.write_text(json.dumps({'accepted': False, 'selected': module.CONFIGS[0],
                                'release_config': module.CONFIGS[0]}))
    module.build(path)
    document = json.loads((tmp_path / 'notebook/v1_optimized_inference.ipynb').read_text())
    assert len(document['cells']) == 3
    source = ''.join(document['cells'][1]['source'])
    payload = None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'CODE_BASE64':
            payload = base64.b64decode(ast.literal_eval(node.value))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        assert 'casmi_ml/v1_research.py' not in names
        assert not any(n.endswith(('.csv', '.parquet', '.pt')) for n in names)
        assert 'prior_seen_identities.json' not in names
        for name in names:
            if name.endswith('.py'):
                for node in ast.walk(ast.parse(archive.read(name).decode())):
                    if isinstance(node, ast.Import):
                        assert all(n.name.split('.')[0] != 'torch' for n in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        assert (node.module or '').split('.')[0] != 'torch'
    prediction = ''.join(document['cells'][2]['source'])
    assert 'inference(' in prediction and 'run(' not in prediction
