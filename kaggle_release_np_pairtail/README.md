# NP-PairTail research execution

The active specification is [PRD_FORWARD_NP_PAIRTAIL_RUN_20261004.md](../docs/PRD_FORWARD_NP_PAIRTAIL_RUN_20261004.md). The first executable stage implements M0 data preparation and M1 C0/C1/C2 development. It is not the full trained system, not a competition submission, and does not open acceptance predictions.

The stage mounts the prior actual canonical identity map, verifies its SHA and RDKit version, constructs the new 80/10/10 identity split, samples at most 8/4/4 spectra, excludes cross-split exact and retrieval-bin copies, and saves actual identity lists. Acceptance identities exclude all recovered previously observed identities plus the prior V1 selection. Historical unretained query identities remain a documented audit limit.

AFIX recomputes formula/precursor-adduct mass consistency without relabeling adducts; original error fields are preserved. It compares the three frozen 10/20/30 ppm thresholds on development and compares instrument-aware against instrument-independent metadata representatives. It retains the original V1 C0 logic and checks 100-query parity.

FPNet, ranker and forward assets have not yet been trained or executed. Forward weights/domain/license manifests and LOTUS import remain work to complete before the corresponding stages; NPAtlas and full PubChem are disabled. This package alone cannot authorize publication.

Actual execution state is recorded in `run_status.json`; no score estimate is a competition score. Complete arrays, identity lists and cases stay under ignored run_output directories.
