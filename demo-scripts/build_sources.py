import html
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root/'dist/data/library.json').read_text())
rows = []
for role, records in [('Reference', [s for c in data['library'] for s in c['spectra']]), ('Example query', data['examples'])]:
    for r in records:
        rows.append('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in [role,r['name'],r['energy']]) +
                    f'<td><a href="{html.escape(r["source"])}">{html.escape(r["id"])}</a><br><a href="{r["raw"]}">Original record</a></td><td>CC0</td></tr>')
page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Sources & credits — Spectra</title><link rel="stylesheet" href="styles.css"><link rel="icon" href="assets/favicon.svg"></head><body><main class="sources-page"><a class="sources-back" href="index.html">← Back to the search lab</a><h1>Open spectra. Traceable sources.</h1>
<p>All twelve spectra below come from MassBank's official open-data repository and explicitly declare CC0. Records are pinned to commit <code>c1c272f44837881d2f310a8171739a6c2223408f</code>. The eight library spectra and four example queries are different acquisitions of the same four compounds; this separation is not a validation benchmark.</p>
<p><strong>Authors:</strong> Tobias Schulze and Martin Krauss, Helmholtz Centre for Environmental Research (UFZ), Leipzig, Germany. NaToxAq records are dated February 2020; the UFZ records are dated August 2023. Full authorship, instrument, collision-energy and processing metadata remain in the original records.</p>
<p><strong>Processing:</strong> Peak m/z values and intensities were extracted; intensities were normalized to their spectrum maximum. Chemical structures were rendered locally with RDKit from the original SMILES. Neutral masses were computed from those structures. No CASMI competition data are distributed.</p>
<table class="sources-table"><caption>Included MassBank records</caption><thead><tr><th>Role</th><th>Compound</th><th>Collision energy</th><th>Record & source</th><th>License</th></tr></thead><tbody>''' + ''.join(rows) + '''</tbody></table>
<p><strong>Method credit:</strong> The spectral vectorization and retrieval method is adapted from the existing team project's <a class="inline-link" href="https://github.com/xiaoyuzhoux120-droid/Enveda-CASMI-2026---Molecule-ID-From-Mass-Spectra/blob/main/baseline.py">baseline.py</a>, originally forked from billylu24's project. This browser companion and user interface were newly developed with AI assistance for Zhenyu Wang. The page's research contribution statement reflects Zhenyu's previously documented role; it does not claim sole authorship of the team's code or results.</p>
<p>Neutral-loss annotations compare measured precursor-to-fragment differences with exact masses of H₂O (18.010565 Da), CO (27.994915 Da), CO₂ (43.989829 Da) and C₆H₁₀O₅ (162.052823 Da). A mass difference is not a unique chemical assignment and does not modify retrieval scores. Only peaks with at least 1% normalized intensity are annotated; tolerance sums separate precursor and fragment measurement bounds of max(10 ppm, 0.002 Da).</p>
<p><a class="inline-link" href="https://github.com/MassBank/MassBank-data">MassBank data repository ↗</a> &nbsp; <a class="inline-link" href="https://creativecommons.org/publicdomain/zero/1.0/">CC0 1.0 Universal ↗</a></p></main></body></html>'''
(root/'dist/sources.html').write_text(page)
