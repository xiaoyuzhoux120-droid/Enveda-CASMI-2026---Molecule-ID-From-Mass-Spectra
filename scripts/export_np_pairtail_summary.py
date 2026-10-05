"""Export actual run metadata without downloading multi-GB spectrum caches.

The resulting ZIP is a metadata subset, never a claim of a full run archive.
Call on a completed/failed Kaggle output mounted read-only in a helper notebook.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


EXACT = {
    'run_status.json', 'runtime_profile.json', 'data_manifest.json',
    'protocol_np_pairtail_20261004.json', 'protocol_fpnet_20261004.json',
    'development_foundation_results.json', 'M1_afix_selection.json',
    'reference_mass_summary.json', 'M0_restored_checkpoint.json',
    'split_manifest.json', 'M2_training_summary.json',
    'M2_development_candidate_gate.json', 'train_fitted_statistics.json',
    'development_identity_order.json', 'base_structure_pool.manifest.json',
    'fpnet_recovery.json',
}


def export_summary(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    selected = []
    for path in sorted(root.rglob('*.json')):
        if (path.name in EXACT or (path.name.startswith('seed_') and path.name.endswith('_history.json'))):
            if not path.resolve().is_relative_to(root):
                raise ValueError('Metadata symlink escapes the actual run directory')
            selected.append(path)
    if not any(path.name == 'run_status.json' for path in selected):
        raise ValueError('No actual run status; cannot export an unverifiable run')
    files = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            data = path.read_bytes()
            json.loads(data)  # Fail on corrupt/truncated summaries.
            relative = str(path.relative_to(root))
            files.append({'path':relative, 'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()})
            archive.writestr(relative, data)
        manifest = {'archive_scope':'actual run metadata subset only',
                    'full_run_archived':False, 'models_included':False,
                    'raw_spectra_included':False, 'candidate_case_rankings_included':False,
                    'files':files}
        archive.writestr('summary_export_manifest.json', json.dumps(manifest, indent=2)+'\n')
    return manifest | {'zip_bytes':destination.stat().st_size,
                       'zip_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('output_zip', type=Path)
    args = parser.parse_args()
    print(json.dumps(export_summary(args.run_directory, args.output_zip), indent=2))
