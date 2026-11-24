"""Drop-in replacement for the 0.495 pipeline's analog-search kernel (pv.search_shift_pre).

Change measured in research/spectral_matching (Task B/C):
  * one-to-one *modified* (hybrid) entropy similarity: fragment matches at shift 0 and neutral-loss
    matches at shift = (query mass - reference mass) compete in ONE greedy assignment, instead of
    max(unshifted, shifted) computed separately;
  * peaks at or above precursor - 1.5 are removed from query and references before cleaning;
  * tolerance tol_da + tol_ppm * 1e-6 * m (study setting: 0.002 Da + 10 ppm) instead of a flat 0.01 Da.
Self-contained (numpy + numba only) so it can be embedded in a notebook cell.

IMPORTANT: the pipeline's GBM rankers were fitted on features computed with the old kernel
(fit_rankers / fit_pv_rankers read precomputed matrices). Swapping the kernel only at inference shifts the
analog-feature distribution; regenerate the ranker training features with this kernel, or use it in an
untrained post-hoc layer (see REPORT.md section 6) before any leaderboard use.
"""
import numpy as np
from numba import njit, prange

_LN4 = np.log(4.0)


def strip_precursor(mz, it, prec, margin=1.5):
    mz = np.asarray(mz, np.float32); it = np.asarray(it, np.float32)
    k = mz < prec - margin
    return mz[k], it[k]


@njit(cache=True)
def _ent_pair(a, b):
    s = a + b
    return (s * np.log(s) - a * np.log(a) - b * np.log(b)) / _LN4


@njit(cache=True)
def _collect(qm, qp, rm, rp, shift, tol_da, tol_ppm, bs, bi, bj, nb):
    m = len(rm); j0 = 0
    for i in range(len(qm)):
        t = qm[i] - shift
        tol = tol_da + tol_ppm * 1e-6 * qm[i]
        while j0 < m and rm[j0] < t - tol:
            j0 += 1
        j = j0
        while j < m and rm[j] <= t + tol:
            if nb < len(bs) and qp[i] > 0 and rp[j] > 0:
                bs[nb] = _ent_pair(qp[i], rp[j]); bi[nb] = i; bj[nb] = j; nb += 1
            j += 1
    return nb


@njit(cache=True)
def modified_entropy(qm, qp, rm, rp, tol_da, tol_ppm, shift):
    """qp, rp: cleaned intensities summing to 1 (pv._clean output). Returns similarity in [0, 1]."""
    cap = 4 * (len(qm) + len(rm)) + 64
    bs = np.empty(cap, np.float64); bi = np.empty(cap, np.int64); bj = np.empty(cap, np.int64)
    nb = _collect(qm, qp, rm, rp, 0.0, tol_da, tol_ppm, bs, bi, bj, 0)
    if abs(shift) >= 1e-3:
        nb = _collect(qm, qp, rm, rp, shift, tol_da, tol_ppm, bs, bi, bj, nb)
    if nb == 0:
        return 0.0
    order = np.argsort(-bs[:nb])
    uq = np.zeros(len(qm), np.bool_); ur = np.zeros(len(rm), np.bool_); tot = 0.0
    for k in order:
        i = bi[k]; j = bj[k]
        if uq[i] or ur[j]:
            continue
        uq[i] = True; ur[j] = True; tot += bs[k]
    return tot


@njit(cache=True, parallel=True)
def search_modified_pre(qmz, qp, lo, hi, roff, rmz, rit, tol_da, tol_ppm, shift):
    """Same contract as pv.search_shift_pre (shift[k] = query mass - reference k mass)."""
    out = np.zeros(hi - lo, np.float32)
    for k in prange(hi - lo):
        a = roff[lo + k]; b = roff[lo + k + 1]
        if b <= a:
            continue
        out[k] = modified_entropy(qmz, qp, rmz[a:b], rit[a:b], tol_da, tol_ppm, shift[k])
    return out
