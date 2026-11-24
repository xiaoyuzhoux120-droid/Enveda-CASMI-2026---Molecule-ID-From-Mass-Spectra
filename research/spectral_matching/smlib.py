"""Spectral similarity kernels for the matching study (numba).

All kernels take peak lists that are already preprocessed (sorted m/z, transformed intensities) and a
mass tolerance tol(m) = tol_da + tol_ppm * 1e-6 * m. Matching is one-to-one: candidate pairs are
collected, sorted by their pair score and assigned greedily (as matchms CosineGreedy / ModifiedCosine).

metric codes: 0 cosine (dot of L2-normalised vectors)
              1 entropy (Li et al. 2021, intensities normalised to sum 1; weighting done in preprocessing)
shift mode:   0 none, 1 max(unshifted, shifted) as the 0.495 pipeline, 2 hybrid/modified (one assignment
              over shift 0 and shift = prec_q - prec_r together, as matchms ModifiedCosine)
"""
import numpy as np
from numba import njit, prange

LN4 = np.log(4.0)


def preprocess(mz, it, prec, *, floor=0.0, topk=0, rm_prec=False, power=1.0, ent_weight=False,
               metric=0, min_mz=0.0, centroid_da=0.0):
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz >= min_mz)
    if rm_prec and np.isfinite(prec):
        ok &= mz < prec - 1.5
    mz, it = mz[ok], it[ok]
    if len(mz) == 0:
        return mz.astype(np.float32), it.astype(np.float32)
    o = np.argsort(mz); mz, it = mz[o], it[o]
    if centroid_da > 0 and len(mz) > 1:  # merge peaks closer than centroid_da (intensity-weighted m/z)
        grp = np.r_[0, np.cumsum(np.diff(mz) > centroid_da)]
        s = np.bincount(grp, it)
        mz = np.bincount(grp, it * mz) / s; it = s
    if floor > 0:
        k = it >= floor * it.max(); mz, it = mz[k], it[k]
    if topk and len(mz) > topk:
        k = np.sort(np.argsort(-it, kind='stable')[:topk]); mz, it = mz[k], it[k]
    it = it ** power
    if metric == 0:
        it = it / np.linalg.norm(it)
    else:
        it = it / it.sum()
        if ent_weight:
            S = -np.sum(it * np.log(it))
            if S < 3.0:
                it = it ** (0.25 + 0.25 * S); it = it / it.sum()
    return mz.astype(np.float32), it.astype(np.float32)


@njit(cache=True)
def _pair_score(a, b, metric):
    if metric == 0:
        return a * b
    s = a + b
    return (s * np.log(s) - a * np.log(a) - b * np.log(b)) / LN4


@njit(cache=True)
def _collect(qm, qi, rm, ri, shift, tol_da, tol_ppm, metric, buf_s, buf_i, buf_j, nb):
    m = len(rm)
    j0 = 0
    for i in range(len(qm)):
        t = qm[i] - shift
        tol = tol_da + tol_ppm * 1e-6 * qm[i]
        while j0 < m and rm[j0] < t - tol:
            j0 += 1
        j = j0
        while j < m and rm[j] <= t + tol:
            if nb < len(buf_s):
                buf_s[nb] = _pair_score(qi[i], ri[j], metric); buf_i[nb] = i; buf_j[nb] = j; nb += 1
            j += 1
    return nb


@njit(cache=True)
def _assign(buf_s, buf_i, buf_j, nb, nq, nr):
    if nb == 0:
        return 0.0, 0
    order = np.argsort(-buf_s[:nb])
    uq = np.zeros(nq, np.bool_); ur = np.zeros(nr, np.bool_)
    tot = 0.0; n = 0
    for k in order:
        i = buf_i[k]; j = buf_j[k]
        if uq[i] or ur[j]:
            continue
        uq[i] = True; ur[j] = True; tot += buf_s[k]; n += 1
    return tot, n


@njit(cache=True)
def similarity(qm, qi, qp, rm, ri, rp, tol_da, tol_ppm, metric, shift_mode):
    cap = 4 * (len(qm) + len(rm)) + 64
    bs = np.empty(cap, np.float64); bi = np.empty(cap, np.int64); bj = np.empty(cap, np.int64)
    nb = _collect(qm, qi, rm, ri, 0.0, tol_da, tol_ppm, metric, bs, bi, bj, 0)
    d = qp - rp
    if shift_mode == 0 or abs(d) < 1e-3:
        return _assign(bs, bi, bj, nb, len(qm), len(rm))
    if shift_mode == 1:
        a, na = _assign(bs, bi, bj, nb, len(qm), len(rm))
        nb2 = _collect(qm, qi, rm, ri, d, tol_da, tol_ppm, metric, bs, bi, bj, 0)
        b, nbm = _assign(bs, bi, bj, nb2, len(qm), len(rm))
        if b > a:
            return b, nbm
        return a, na
    nb = _collect(qm, qi, rm, ri, d, tol_da, tol_ppm, metric, bs, bi, bj, nb)
    return _assign(bs, bi, bj, nb, len(qm), len(rm))


@njit(cache=True, parallel=True)
def score_pairs(pq, pr, qoff, qmz, qit, qprec, roff, rmz, rit, rprec, tol_da, tol_ppm, metric, shift_mode):
    n = len(pq)
    out = np.zeros(n, np.float32); nm = np.zeros(n, np.int32)
    for k in prange(n):
        a = pq[k]; b = pr[k]
        s, c = similarity(qmz[qoff[a]:qoff[a + 1]], qit[qoff[a]:qoff[a + 1]], qprec[a],
                          rmz[roff[b]:roff[b + 1]], rit[roff[b]:roff[b + 1]], rprec[b],
                          tol_da, tol_ppm, metric, shift_mode)
        out[k] = s; nm[k] = c
    return out, nm


def pack(spectra):
    """spectra: list of (mz, it) -> offsets, flat mz, flat it"""
    lens = np.array([len(s[0]) for s in spectra], np.int64)
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    mz = np.concatenate([s[0] for s in spectra]) if len(spectra) else np.empty(0, np.float32)
    it = np.concatenate([s[1] for s in spectra]) if len(spectra) else np.empty(0, np.float32)
    return off, mz.astype(np.float32), it.astype(np.float32)


@njit(cache=True)
def nl_transform(mz, it, prec):
    """Neutral-loss spectrum: prec - mz, re-sorted."""
    nl = prec - mz[::-1]
    return nl.astype(np.float32), it[::-1].copy()
