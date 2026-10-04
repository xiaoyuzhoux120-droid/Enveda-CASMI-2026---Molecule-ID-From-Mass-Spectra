"""Independently verify saved routing cases, frozen gate, and output deduplication."""
import argparse
import json
import math
from pathlib import Path
from casmi_ml.v1_routing import acceptance, metrics


def audit(root):
    root = Path(root)
    selected = json.loads((root / 'selection.json').read_text())['selected']
    saved = json.loads((root / 'acceptance_report.json').read_text())
    unknown = json.loads((root / 'acceptance_unknown_cases.json').read_text())
    known = json.loads((root / 'acceptance_known_cases.json').read_text())
    identity_sets = []
    summary = {}
    for label in ('dev_unknown', 'dev_known', 'acceptance_unknown', 'acceptance_known'):
        rows = json.loads((root / f'{label}_cases.json').read_text())
        if label in ('dev_unknown', 'acceptance_unknown'):
            ids = [r['identity'] for r in rows]
            if len(set(ids)) != len(ids): raise ValueError('Repeated query identity')
            identity_sets.append(set(ids))
        clean = []
        removed = 0
        for row in rows:
            rr = {}
            for name, ranking in row['rankings'].items():
                raw_ids = [p['identity'] for p in ranking]
                value = 1/(raw_ids.index(row['identity'])+1) if row['identity'] in raw_ids else 0.
                if not math.isclose(value, row['rr'][name], abs_tol=1e-14):
                    raise ValueError('Stored reciprocal rank disagrees with candidate identities')
                if any(not isinstance(i, str) or len(i)!=14 for i in raw_ids):
                    raise ValueError('Invalid candidate identity; needs engineering diagnosis')
                ids = list(dict.fromkeys(raw_ids))
                rr[name] = 1/(ids.index(row['identity'])+1) if row['identity'] in ids else 0.
                if rr[name] < value: raise ValueError('Deduplication worsened truth rank')
                removed += len(raw_ids)-len(ids)
            clean.append({'rr':rr})
        names = list(rows[0]['rr'])
        summary[label] = {'molecules':len(rows), 'duplicate_slots_removed_across_arms':removed,
                          'raw':{n:metrics(rows,n) for n in names},
                          'normalized':{n:metrics(clean,n) for n in names}}
    if identity_sets[0] & identity_sets[1]: raise ValueError('Development/acceptance identity leakage')
    rebuilt = acceptance(unknown, known, selected)
    for field in ('accepted','selected','release_config','known_top1_difference','unknown','known'):
        if rebuilt[field] != saved[field]: raise ValueError(f'Frozen gate mismatch: {field}')
    result = {'status':'independently_verified','frozen_gate':rebuilt,
              'development_acceptance_identity_overlap':0,'cohorts':summary,
              'format_policy':'default RDKit tautomer identity; retain first; no refill',
              'limitations':['Verifies retained cases and identity lists, not all original training rows.',
                             'Deduplication analysis does not change frozen raw-routing acceptance.']}
    (root/'independent_recheck.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'accepted':rebuilt['accepted'],
                      'release_config':rebuilt['release_config']}))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    audit(parser.parse_args().root)
