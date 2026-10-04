# V1 optimization execution — 2026-10-03

The active specification is [PRD_V1_OPTIMIZATION_20261003.md](../docs/PRD_V1_OPTIMIZATION_20261003.md).

The separate [research notebook](https://www.kaggle.com/code/xiaoyuzhoux120/casmi-v1-routing-research-20261003?scriptVersionId=355064584) runs CPU/offline with frozen source `5586d3e` and archive SHA256 `17575695c61d0b653b2b843e6ef7f7b16d852d24f3343ca34f7d1c5e7a305061`. **Do not submit the research notebook.** Its old visible CSV is solely a replay reference and is excluded from the later inference package.

The real replay matched all 400 original V1 rows, including complete candidate order, in 225.709936224 seconds. This is prediction time, not full research runtime or a new public score. The subsequent research canonically isolates fresh cohorts, compares six predeclared routing controls on development, freezes one choice, and evaluates that choice against B0 on separate acceptance. The protocol permits 7,200 seconds for research and stops if the first 50 ranking queries average over six seconds. Acceptance/public results are not used to tune parameters.

`build_inference.py` accepts the actual acceptance decision and builds a small dynamic-test retrieval notebook. It contains no query-ID lookup, reference CSV, research cohort, neural training or graph generation. Unaccepted scientific routing changes cannot be packaged. A B0 fallback is explicitly labeled an engineering release, not a scientific improvement. Research completed successfully in 3558.3 seconds; independent verification rejected the selected 0.95 guard. The retained B0 inference package is 32,112 bytes. Formal CPU inference succeeded in 364.6 seconds (333.115 seconds inside the inference entry), all 400 rows independently matched normalized V1, and Version 3 was submitted. The competition hidden rerun is Notebook Running; its score remains pending; see [status.json](status.json), [validation](validation_summary.json), and [actual result analysis](../docs/V1_OPTIMIZATION_RESULTS_20261003.md).

## Output identity correction, declared before acceptance

An independent audit of the previously downloaded V1 CSV found 21 of 400 rows with duplicate default-RDKit-tautomer InChIKey14 identities, among 9,281 distinct SMILES. See [v1_identity_format_audit.json](v1_identity_format_audit.json). This audit uses prior visible output, not hidden labels or new acceptance results.

The inference entry now keeps the first ranked representative for each official identity and does not refill removed slots. Raw routing order, sources, scores, chemical weight and frozen research gates are unchanged. Under the same identity definition, removing later duplicates cannot decrease reciprocal rank for any truth: a first matching occurrence remains, and removal before it only improves its position. This correction is reported separately from routing effectiveness. The original full replay retains duplicates to establish exact V1 reproduction; its 400-row equality therefore applies to raw rankings before output normalization. No public-score gain is guaranteed. Canonicalization in final inference is limited to output candidates and cached by SMILES, not the full training structure universe.

Validation: 28 targeted routing, chemistry and release packaging tests passed. These engineering checks do not establish chemical generalization. Final inference timing, actual output and competition score must be recorded after execution.

[Public ID-mapping caveat](../docs/LEADERBOARD_ID_CAVEAT_20261003.md): a separately inspected public notebook uses a fixed 400-ID top-1 map. Our inference excludes such lookup data and passes an arbitrary query-ID remapping regression. This does not establish hidden ID reuse.

Final-entry instrumentation records reference/index loading, retrieval, analog ranking, chemistry, candidate identity normalization and output-writing time. Optional reference-load progress/deadline arguments leave legacy default scoring unchanged; the original frozen replay archive remains authoritative.
