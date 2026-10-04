"""
detection.py -- exact-binomial detection and the reference receivers.

Fixed-threshold detector (Section II-B): the number of bound receptors n_B is compared with N_R / 2, and the detector
decides 1 if n_B > N_R/2, 0 if n_B < N_R/2, and at random if n_B = N_R/2.  For even N_R, this rule equals the strict
majority rule of N_R - 1 receptors, and for odd N_R, no tie occurs.  With equal priors,
    BEP   = 1/2 P_e|0 + 1/2 P_e|1,
    P_e|0 = P[n_B > m | theta0] + 1/2 P[n_B = m | theta0],
    P_e|1 = P[n_B < m | theta1] + 1/2 P[n_B = m | theta1],     m = N_R / 2.
At K_D = sqrt(c0 c1), this rule is the maximum-likelihood rule, and K_D = sqrt(c0 c1) is the unique global minimizer
of the BEP (Proposition 1).  All reported error probabilities are exact binomial tails averaged over the simulated
modulator fluctuations.

Mixed arrays: n_B = Bin(N_t, theta_t(y)) + Bin(N_f, theta_f), evaluated exactly by convolution.

Reference receivers, with the same Hill coefficient as the adaptive receiver they are compared with:
    best non-adaptive receiver ("static" in the code): fixed K_D and decision threshold, both optimized over the
        channel ensemble with full knowledge of the channel statistics;
    genie-aided receiver ("oracle" in the code): K_D = sqrt(c0 c1) at every instant, without noise.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from model import N_RECEPTORS, W_WEIGHT, K_FIX, theta_B, kd_of_y


def _err_terms(t0, t1, n_r, thr):
    """Conditional error probabilities for threshold thr (integer: randomized tie-breaking;
    half-integer: deterministic).  Arrays broadcast."""
    if abs(thr - round(thr)) < 1e-9:
        m = int(round(thr))
        e0 = binom.sf(m, n_r, t0) + 0.5 * binom.pmf(m, n_r, t0)
        e1 = binom.cdf(m - 1, n_r, t1) + 0.5 * binom.pmf(m, n_r, t1)
    else:
        m = int(np.floor(thr))
        e0 = binom.sf(m, n_r, t0)
        e1 = binom.cdf(m, n_r, t1)
    return e0, e1


def bep_exact(theta0, theta1, n_r=N_RECEPTORS, thr=None):
    """Exact BEP of the homogeneous binomial array; thr defaults to the midpoint."""
    thr = n_r / 2.0 if thr is None else thr
    t0 = np.clip(np.asarray(theta0, float), 1e-12, 1 - 1e-12)
    t1 = np.clip(np.asarray(theta1, float), 1e-12, 1 - 1e-12)
    e0, e1 = _err_terms(t0, t1, n_r, thr)
    return 0.5 * e0 + 0.5 * e1


def hill(c, kd, n):
    """Hill occupancy c^n / (c^n + kd^n)."""
    cn = np.power(c, n)
    return cn / (cn + np.power(kd, n))


# --------------------------------------------------------------------------- #
#  Mixed array: exact two-cohort count distribution
# --------------------------------------------------------------------------- #
def _mixed_tail(theta_t, theta_f, n_t, n_f, m):
    """P[n_B > m], P[n_B = m] for n_B = Bin(n_t, theta_t) + Bin(n_f, theta_f).
    theta_t: array (any shape); theta_f scalar."""
    theta_t = np.asarray(theta_t, float)
    ks = np.arange(0, n_t + 1)
    pk = binom.pmf(ks[None, :], n_t, theta_t.ravel()[:, None])          # (n, n_t+1)
    sf_f = binom.sf(m - ks, n_f, theta_f)                                 # P[X_f > m - k]
    pmf_f = binom.pmf(m - ks, n_f, theta_f)                               # P[X_f = m - k]
    p_gt = pk @ sf_f
    p_eq = pk @ pmf_f
    return p_gt.reshape(theta_t.shape), p_eq.reshape(theta_t.shape)


def bep_mixed_exact(alpha, y, c0, c1, n_r=N_RECEPTORS, w=W_WEIGHT, k_fix=K_FIX):
    """Exact BEP of the mixed array (tunable fraction w at K_D(y), fixed fraction at
    k_fix) with the randomized midpoint rule.  y array, c0 and c1 scalars."""
    y = np.asarray(y, float)
    n_t = int(round(w * n_r)); n_f = n_r - n_t; m = n_r // 2
    kd = kd_of_y(y, alpha)
    eps = 1e-9
    tt0, tt1 = np.clip(c0 / (c0 + kd), eps, 1 - eps), np.clip(c1 / (c1 + kd), eps, 1 - eps)
    tf0, tf1 = float(np.clip(c0 / (c0 + k_fix), eps, 1 - eps)), float(np.clip(c1 / (c1 + k_fix), eps, 1 - eps))
    gt0, eq0 = _mixed_tail(tt0, tf0, n_t, n_f, m)
    gt1, eq1 = _mixed_tail(tt1, tf1, n_t, n_f, m)
    e0 = gt0 + 0.5 * eq0
    e1 = (1.0 - gt1 - eq1) + 0.5 * eq1
    return 0.5 * e0 + 0.5 * e1


def bep_from_record(scheme, alpha, y_rec, c0_rec, c1_rec, n_r=N_RECEPTORS):
    """
    Exact BEP of the adaptive receiver along a recorded ensemble.
    y_rec: (n_rec, n_traj); c0_rec, c1_rec: (n_rec,).
    Returns (mean over everything, per-trajectory time-averaged BEP array).
    """
    c0 = np.asarray(c0_rec, float)[:, None]
    c1 = np.asarray(c1_rec, float)[:, None]
    if scheme == "weighted":
        if np.ptp(c0_rec) > 1e-12 or np.ptp(c1_rec) > 1e-12:
            raise ValueError("mixed-array exact BEP implemented for a static channel only")
        # evaluate on the set of distinct (rounded) y values, then map back
        yq = np.round(np.asarray(y_rec, float), 3)
        uniq, inv = np.unique(yq, return_inverse=True)
        bu = bep_mixed_exact(alpha, uniq, float(c0_rec[0]), float(c1_rec[0]), n_r)
        b = bu[inv].reshape(yq.shape)
    else:
        th0 = theta_B(scheme, c0, y_rec, alpha)
        th1 = theta_B(scheme, c1, y_rec, alpha)
        b = bep_exact(th0, th1, n_r)
    per_traj = b.mean(axis=0)
    return float(per_traj.mean()), per_traj


def occupancy_record(scheme, alpha, y_rec, c0_rec, c1_rec):
    c0 = np.asarray(c0_rec, float)[:, None]
    c1 = np.asarray(c1_rec, float)[:, None]
    return theta_B(scheme, c0, y_rec, alpha), theta_B(scheme, c1, y_rec, alpha)


# --------------------------------------------------------------------------- #
#  Reference receivers
# --------------------------------------------------------------------------- #
def best_static_receiver(c0_grid, c1_grid, n_hill=1.0, n_r=N_RECEPTORS,
                         n_kd=300, kd_lo=1e-3, kd_hi=1e2):
    """
    Best non-adaptive receiver with the Hill coefficient n_hill: the fixed K_D and the decision threshold
    (integer thresholds with randomized tie-breaking and half-integer deterministic thresholds) jointly minimize
    the mean BEP over the channel ensemble (c0_grid, c1_grid), i.e., with full knowledge of the channel
    statistics.  Returns dict(kd, thr, bep, per_point).
    """
    c0 = np.asarray(c0_grid, float)[None, :]
    c1 = np.asarray(c1_grid, float)[None, :]
    kd = np.geomspace(kd_lo, kd_hi, n_kd)[:, None]
    t0 = np.clip(hill(c0, kd, n_hill), 1e-12, 1 - 1e-12)
    t1 = np.clip(hill(c1, kd, n_hill), 1e-12, 1 - 1e-12)
    best = dict(kd=None, thr=None, bep=np.inf, per_point=None)
    for thr in np.arange(0.0, n_r, 0.5):
        e0, e1 = _err_terms(t0, t1, n_r, thr)
        b = 0.5 * e0 + 0.5 * e1
        m = b.mean(axis=1)
        i = int(np.argmin(m))
        if m[i] < best["bep"]:
            best = dict(kd=float(kd[i, 0]), thr=float(thr), bep=float(m[i]), per_point=b[i].copy())
    return best


def static_receiver_bep(kd, thr, c0, c1, n_hill=1.0, n_r=N_RECEPTORS):
    """Exact BEP of a non-adaptive receiver with the fixed affinity kd and threshold thr."""
    return bep_exact(hill(np.asarray(c0, float), kd, n_hill),
                     hill(np.asarray(c1, float), kd, n_hill), n_r, thr)


def oracle_bep(c0, c1, n_hill=1.0, n_r=N_RECEPTORS):
    """Exact BEP of the genie-aided receiver, i.e., with K_D = sqrt(c0 c1) at every instant."""
    c0 = np.asarray(c0, float)
    c1 = np.asarray(c1, float)
    kd = np.sqrt(c0 * c1)
    return bep_exact(hill(c0, kd, n_hill), hill(c1, kd, n_hill), n_r)


def bootstrap_ci(per_traj, n_boot=4000, seed=0, level=0.95):
    """Percentile bootstrap over trajectories (the independent unit)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(per_traj, float)
    idx = rng.integers(0, x.size, (n_boot, x.size))
    m = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.percentile(m, 100 * a)), float(np.percentile(m, 100 * (1 - a)))
