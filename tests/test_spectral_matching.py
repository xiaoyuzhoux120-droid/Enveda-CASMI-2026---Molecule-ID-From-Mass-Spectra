import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research' / 'spectral_matching'))
smlib = pytest.importorskip('smlib')
import pv_modified_kernel as pk  # noqa: E402


def _spec(rng, n, prec):
    mz = np.sort(rng.uniform(50, prec + 1, n)); it = rng.uniform(0.001, 1, n)
    return mz, it


def test_identical_spectra_score_one():
    m, i = smlib.preprocess([100, 150, 200], [1, .5, .2], 300, metric=1)
    for metric in (0, 1):
        mm, ii = smlib.preprocess([100, 150, 200], [1, .5, .2], 300, metric=metric)
        s, n = smlib.similarity(mm, ii, 300., mm, ii, 300., 0.01, 0, metric, 0)
        assert abs(s - 1) < 1e-5 and n == 3


def test_modified_matches_neutral_losses():
    a = smlib.preprocess([100, 150, 250], [1, .5, .3], 300, metric=1)
    b = smlib.preprocess([100, 164, 264], [1, .5, .3], 314, metric=1)  # +14 Da on the two heavy fragments
    plain, _ = smlib.similarity(a[0], a[1], 300., b[0], b[1], 314., 0.01, 0, 1, 0)
    hybrid, n = smlib.similarity(a[0], a[1], 300., b[0], b[1], 314., 0.01, 0, 1, 2)
    assert hybrid > 0.99 and n == 3 and plain < 0.8


def test_ppm_tolerance():
    a = smlib.preprocess([500.0], [1.0], 600, metric=1)
    b = smlib.preprocess([500.008], [1.0], 600, metric=1)  # 16 ppm; tol(500) = 0.002 + 0.005 Da
    assert smlib.similarity(a[0], a[1], 600., b[0], b[1], 600., 0.002, 10, 1, 0)[0] == 0
    assert smlib.similarity(a[0], a[1], 600., b[0], b[1], 600., 0.002, 20, 1, 0)[0] > 0.99


def test_dropin_kernel_equals_study_kernel():
    rng = np.random.default_rng(3)
    for _ in range(50):
        pq, pr = rng.uniform(200, 600, 2)
        qm, qi = _spec(rng, 40, pq); rm, ri = _spec(rng, 40, pr)
        a = smlib.preprocess(qm, qi, pq, metric=1, ent_weight=True, rm_prec=True)
        b = smlib.preprocess(rm, ri, pr, metric=1, ent_weight=True, rm_prec=True)
        ref, _ = smlib.similarity(a[0], a[1], pq, b[0], b[1], pr, 0.002, 10, 1, 2)
        got = pk.modified_entropy(a[0], a[1], b[0], b[1], 0.002, 10.0, pq - pr)
        assert abs(ref - got) < 1e-6
        roff = np.array([0, len(b[0])], np.int64)
        out = pk.search_modified_pre(a[0], a[1], 0, 1, roff, b[0], b[1], 0.002, 10.0, np.array([pq - pr], np.float32))
        assert abs(out[0] - got) < 1e-4
