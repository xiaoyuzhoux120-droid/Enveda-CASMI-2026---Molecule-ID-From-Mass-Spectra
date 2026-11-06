"""One-time numerical oracle from the unchanged original Python baseline."""
import importlib.util
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
baseline_path = root.parent/'casmi26/baseline.py'
if not baseline_path.exists():
    baseline_path = root/'baseline.py'
spec = importlib.util.spec_from_file_location('casmi_baseline', baseline_path)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
data = json.loads((root/'dist/data/library.json').read_text())
expected=[]
for query in data['examples']:
    q=baseline.vectorize(*zip(*query['peaks']))
    for compound in data['library']:
        scores=[]
        for record in compound['spectra']:
            r=baseline.vectorize(*zip(*record['peaks']))
            scores.append(sum(v*r.get(k,0) for k,v in q.items()))
        expected.append(dict(query=query['id'],compound=compound['id'],score=max(scores)))
rounding_peaks=[[40.05,1],[40.15,.8],[100.25,.5],[100.35,.7]]
fixture=dict(source='Unmodified project baseline.py vectorize(), computed with Python/numpy',
             scores=expected,roundingPeaks=rounding_peaks,
             roundingVector=list(baseline.vectorize(*zip(*rounding_peaks)).items()))
(root/'tests/python-oracle.json').write_text(json.dumps(fixture,indent=2)+'\n')
print('Saved 16 cross-spectrum cosine scores and numpy rounding oracle.')
