"""Regression for raw data versus identically named research checkpoints."""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_paths(input_root):
    notebook = json.loads((ROOT/'kaggle_release_np_pairtail/notebook/np_pairtail_foundation.ipynb').read_text())
    tree = ast.parse(''.join(notebook['cells'][1]['source']))
    statements = [node for node in tree.body if
        isinstance(node, ast.FunctionDef) and node.name == 'unique' or
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'TRAIN_PATH','TEST_PATH'} for t in node.targets)]
    namespace = {'INPUT':input_root, 'Path':Path}
    exec(compile(ast.Module(body=statements, type_ignores=[]),'frozen_bootstrap','exec'), namespace)
    return namespace['TRAIN_PATH'], namespace['TEST_PATH']


def test_bootstrap_ignores_checkpoint_named_train(tmp_path):
    raw = tmp_path/'competitions'/'casmi'; raw.mkdir(parents=True)
    for name in ['train.parquet','test.parquet']:
        (raw/name).touch()
    checkpoint = tmp_path/'notebooks'/'M0'; checkpoint.mkdir(parents=True)
    (checkpoint/'train.parquet').touch()
    assert bootstrap_paths(tmp_path) == (raw/'train.parquet',raw/'test.parquet')


def test_bootstrap_refuses_multiple_raw_pairs(tmp_path):
    for name in ['competition_a','competition_b']:
        folder=tmp_path/name;folder.mkdir()
        (folder/'train.parquet').touch();(folder/'test.parquet').touch()
    with pytest.raises(FileNotFoundError,match='Expected one'):
        bootstrap_paths(tmp_path)


def test_bootstrap_refuses_checkpoint_without_raw_data(tmp_path):
    (tmp_path/'train.parquet').touch()
    with pytest.raises(FileNotFoundError,match='Expected one'):
        bootstrap_paths(tmp_path)
