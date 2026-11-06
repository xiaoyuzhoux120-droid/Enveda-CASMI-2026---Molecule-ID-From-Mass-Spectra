# Spectra — Molecular Search Lab

A browser-only companion to the [Enveda CASMI 2026 project](https://github.com/xiaoyuzhoux120-droid/Enveda-CASMI-2026---Molecule-ID-From-Mass-Spectra), created for Zhenyu Wang's ACM Industry application, with AI assistance.

## Run

No install or build is required. From this directory:

```sh
python3 -m http.server 8765 --directory dist
```

Open http://localhost:8765. Run numerical and input checks with `node --test tests/*.test.mjs`.

## Functionality

- Four real CC0 MassBank query spectra, separate from eight reference spectra spanning four compounds.
- Editable precursor and positive singly charged adduct, peak text entry, local CSV/TSV import, mass tolerance and filter controls.
- Real browser-computed weighted cosine scores, candidate selection, molecular structures, mirror spectrum, neutral-loss annotations, and CSV export.
- No account, API key, analytics, server-side processing, or model download. Query spectra remain on the device. Google Fonts is optional presentation only; system fallbacks work if it is blocked.

The algorithm adapts `baseline.py` in the team project: 0.1 Da bins, maximum weight for duplicate bins, square-root intensity and m/z weighting, up to 100 peaks with normalized intensity ≥0.01 and m/z in [40,1250), L2 normalization, maximum cosine across reference spectra per compound. Mass filtering uses max(neutral mass × ppm × 1e-6, 0.006 Da). The browser deliberately returns no candidate when none pass, rather than the original baseline's nearest-mass fallback. The selected adduct is used for query neutral mass; spectral references are all [M+H]+ and other-adduct comparisons are exploratory.

This is a small educational retrieval demonstration, not the full competition inference pipeline, a general molecule identifier, a benchmark, or a de novo generator. Separate example records are not molecule-disjoint validation. Similarity is not identification probability. Neutral losses are mass-difference annotations only and do not affect candidate ranking or assert structural motifs.

## Provenance and credit

- Existing team baseline: [source and original history](https://github.com/xiaoyuzhoux120-droid/Enveda-CASMI-2026---Molecule-ID-From-Mass-Spectra/blob/main/baseline.py); original project forked from billylu24. Team contributions are not all attributed to Zhenyu Wang.
- Zhenyu's previously documented research contribution: competition-material research; investigating diagnostic fragment-ion and neutral-loss resources; proposing chemistry-informed approaches and specialized Transformer architectures; sharing findings in team meetings. The webpage is a new companion developed with AI assistance.
- Data: [MassBank/MassBank-data](https://github.com/MassBank/MassBank-data), pinned at `c1c272f44837881d2f310a8171739a6c2223408f`. All 12 included records explicitly declare **CC0**. Authors: Tobias Schulze and Martin Krauss, UFZ. Original records are retained unchanged under `dist/data/records/`; normalized arrays, formula/SMILES metadata and exact RDKit formula masses are in `dist/data/library.json`.
- Structures are locally rendered with RDKit from source SMILES. No competition data or model weights are redistributed.
- Record-level links, processing changes, license information, and authors are on `dist/sources.html`.

## GitHub Pages

The web application is published under `dist/` on `codex/spectra-pages`. A small root entry page redirects visitors to `dist/`. Development scripts are published under `demo-scripts/`; tests retain their `tests/` paths. This file is published as `DEMO_README.md`, preserving the research README. GitHub Pages serves the branch root; `.nojekyll` bypasses Jekyll. The existing research branch and local competition experiments are preserved.
