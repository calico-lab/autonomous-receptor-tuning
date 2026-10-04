"""
engine.py -- stochastic simulation of the closed-loop receiver.

Reactions (molecule counts, system size Omega):
    r1  0        -> Z1        mu Omega                          (reference)
    r2  0        -> Z2        theta theta_B(c(t); Y) Omega      (sensing)
    r3  Z1       -> Z1 + Y    k Z1                              (actuation)
    r4  Y        -> 0         gamma_p Y                         (modulator turnover)
    r5  Z1 + Z2  -> 0         (eta / Omega) Z1 Z2               (sequestration)
    r6  Z1 -> 0, Z2 -> 0      gamma_c                           (leaky integration)
The cascaded variant "CAIF", which is not reported in the paper, produces Z1 at the rate eps mu Omega and adds
    r7  0        -> Z3        (1 - eps) mu Omega
    r8  Z2 + Z3  -> 0         (eta1 / Omega) Z2 Z3

Ligand binding is at quasi-steady state, i.e., the occupancy is an algebraic function of Y, evaluated with the full
nonlinear Hill map at the current received concentration.  Nothing is linearized in the simulation.

Drive modes (what the controller senses):
    "stream"         an actual random symbol stream, independent for every trajectory.  levels[tr, j] is the level
                     of symbol j relative to c0 (1 for bit 0, c1/c0 for bit 1, or a value that includes ISI), held
                     for t_sym, and c(t) = levels[tr, j] * c0_fn(t), such that a slow common attenuation is carried
                     by c0_fn.  Used for every reported performance result.
    "bitavg"         the controller senses the symbol-averaged occupancy directly, i.e., without self-noise
                     (averaged-input reference).  Also used for the stability sweep.
    "instantaneous"  c0_fn(t) returns the actual received concentration (a scalar or one value per trajectory),
                     as needed for the CIR-based model.
The option linear_zeta replaces the receptor map by the logistic sensing function 1/(1 + exp(-4 zeta (y - y*))),
which has the slope zeta at y* and stays in (0, 1).  It is used only for the stability sweep of Section III-D.

Integrators:
    tauleap()    vectorized Poisson tau-leaping (high copy numbers)
    exact_ssa()  Gillespie direct method (low copy numbers and cross-checks)
Both return the modulator concentration y = Y / Omega on a recording grid.
"""
from __future__ import annotations

import numpy as np

import model as M
from model import (OMEGA_HC, T_SYM, CL0_NOM, CL1_NOM, sensed, theta_B, steady_state, y_equilibrium, stable_seed)


# --------------------------------------------------------------------------- #
#  Symbol streams
# --------------------------------------------------------------------------- #
def make_levels(rng, n_traj, n_sym, r=CL1_NOM / CL0_NOM, kind="iid", isi_taps=None):
    """
    Per-symbol level multipliers (relative to c0), shape (n_traj, n_sym).
    kind = "iid" : independent equiprobable bits, independent per trajectory
           "alt" : alternating 0101... (no fluctuation of the fraction of ones; reference only)
    isi_taps     : optional normalized channel taps w[0..], w[0] = 1; the level of
                   symbol j becomes sum_m w[m] level[j-m] (free-diffusion tail).
    """
    if kind == "iid":
        bits = rng.integers(0, 2, (n_traj, n_sym))
    elif kind == "alt":
        bits = np.tile(np.arange(n_sym) % 2, (n_traj, 1))
    else:
        raise ValueError(kind)
    lv = np.where(bits == 1, r, 1.0).astype(float)
    if isi_taps is not None:
        w = np.asarray(isi_taps, float)
        lv = np.array([np.convolve(row, w)[:n_sym] for row in lv])
    return lv, bits


def n_symbols(t_end, t_sym=T_SYM):
    return int(np.ceil(t_end / t_sym)) + 2


def _const(v):
    return (lambda t: v) if not callable(v) else v


def _init_counts(scheme, alpha, motif, omega, c0, c1, gc):
    ss = steady_state(scheme, alpha, motif, gc, c0, c1)
    return (max(ss["y"], 0.0) * omega, max(ss["z1"], 0.0) * omega,
            max(ss["z2"], 0.0) * omega, max(ss["z3"], 0.0) * omega)


# --------------------------------------------------------------------------- #
#  Tau-leaping
# --------------------------------------------------------------------------- #
def tauleap(scheme, alpha, c0_fn, c1_fn, *, motif="AIF", omega=OMEGA_HC, gc=0.0,
            n_traj=64, t_end=600.0, dt=0.01, t_burn=200.0, dt_rec=1.0, seed=0,
            linear_zeta=None, y_init=None, drive="stream", init_levels=None,
            levels=None, t_sym=T_SYM, stream_kind="iid", isi_taps=None):
    """
    Returns t_rec (n_rec,), y_rec (n_rec, n_traj), c0_rec, c1_rec, clip_fraction.
    In "stream" mode the levels matrix is generated here (seeded) unless given.
    """
    MU, THETA, K, ETA, GP, EPS, ETA1 = M.MU, M.THETA, M.K, M.ETA, M.GP, M.EPS, M.ETA1
    rng = np.random.default_rng(seed)
    c0_fn, c1_fn = _const(c0_fn), _const(c1_fn)
    l0, l1 = (c0_fn(0.0), c1_fn(0.0)) if init_levels is None else init_levels
    Y0, Z10, Z20, Z30 = _init_counts(scheme, alpha, motif, omega, l0, l1, gc)
    if y_init is not None:
        Y0 = y_init * omega
    ys_lin = y_equilibrium(scheme, alpha, l0, l1) if linear_zeta is not None else None
    if drive == "stream" and levels is None:
        levels, _ = make_levels(rng, n_traj, n_symbols(t_end, t_sym), r=l1 / l0,
                                kind=stream_kind, isi_taps=isi_taps)
    if drive == "stream":
        levels = np.asarray(levels, float)
        assert levels.shape[0] == n_traj, "levels must be (n_traj, n_sym)"
        n_sym = levels.shape[1]

    Y = np.full(n_traj, Y0, float)
    Z1 = np.full(n_traj, Z10, float)
    Z2 = np.full(n_traj, Z20, float)
    Z3 = np.full(n_traj, Z30, float)
    prod_z1 = (EPS * MU if motif == "CAIF" else MU) * omega
    prod_z3 = (1.0 - EPS) * MU * omega
    n_steps = int(round(t_end / dt))
    rec_every = max(1, int(round(dt_rec / dt)))
    P = rng.poisson
    t_rec, y_rec, c0_rec, c1_rec = [], [], [], []
    n_clip = 0

    for it in range(n_steps + 1):
        t = it * dt
        c0, c1 = c0_fn(t), c1_fn(t)
        y = Y / omega
        if linear_zeta is not None:
            # logistic sensing with slope linear_zeta at y*: bounded in (0, 1), no clipping needed
            g = 1.0 / (1.0 + np.exp(-4.0 * linear_zeta * (y - ys_lin)))
        elif drive == "stream":
            j = min(int(t / t_sym), n_sym - 1)
            g = np.clip(theta_B(scheme, levels[:, j] * c0, y, alpha), 0.0, 1.0)
        elif drive == "instantaneous":
            g = np.clip(theta_B(scheme, c0, y, alpha), 0.0, 1.0)
        else:
            g = np.clip(sensed(scheme, c0, c1, y, alpha), 0.0, 1.0)

        n_pz1 = P(prod_z1 * dt, n_traj)
        n_pz2 = P(THETA * g * omega * dt)
        n_cat = P(K * Z1 * dt)
        n_dY = P(GP * Y * dt)
        n_sq = P(np.clip(ETA / omega * Z1 * Z2, 0.0, None) * dt)
        Y = Y + n_cat - n_dY
        Z1 = Z1 + n_pz1 - n_sq
        Z2 = Z2 + n_pz2 - n_sq
        if gc > 0:
            Z1 = Z1 - P(gc * np.clip(Z1, 0.0, None) * dt)
            Z2 = Z2 - P(gc * np.clip(Z2, 0.0, None) * dt)
        if motif == "CAIF":
            n_pz3 = P(prod_z3 * dt, n_traj)
            n_sq2 = P(np.clip(ETA1 / omega * Z2 * Z3, 0.0, None) * dt)
            Z2 = Z2 - n_sq2
            Z3 = Z3 + n_pz3 - n_sq2
            if gc > 0:
                Z3 = Z3 - P(gc * np.clip(Z3, 0.0, None) * dt)
            n_clip += int(np.count_nonzero(Z3 < 0))
            np.clip(Z3, 0.0, None, out=Z3)
        n_clip += int(np.count_nonzero(Y < 0) + np.count_nonzero(Z1 < 0)
                      + np.count_nonzero(Z2 < 0))
        np.clip(Y, 0.0, None, out=Y)
        np.clip(Z1, 0.0, None, out=Z1)
        np.clip(Z2, 0.0, None, out=Z2)
        if t >= t_burn - 1e-9 and (it % rec_every == 0):
            t_rec.append(t)
            y_rec.append(Y / omega)
            c0_rec.append(c0 if np.ndim(c0) == 0 else float(np.mean(c0)))
            c1_rec.append(c1 if np.ndim(c1) == 0 else float(np.mean(c1)))

    frac = n_clip / max(1, n_steps * n_traj)
    return (np.asarray(t_rec), np.asarray(y_rec), np.asarray(c0_rec, float),
            np.asarray(c1_rec, float), frac)


# --------------------------------------------------------------------------- #
#  Exact SSA
# --------------------------------------------------------------------------- #
def exact_ssa(scheme, alpha, c0_fn, c1_fn, *, motif="AIF", omega=20.0, gc=0.0,
              n_traj=32, t_end=600.0, t_burn=200.0, dt_rec=1.0, seed=0,
              linear_zeta=None, drive="stream", levels=None, t_sym=T_SYM,
              stream_kind="iid", isi_taps=None):
    """
    Gillespie direct method.  Propensities are piecewise constant between segment
    boundaries (symbol boundaries in "stream" mode, recording instants otherwise),
    which is exact for a piecewise-constant symbol stream on a static channel and a first-order
    approximation for a slowly varying attenuation.  Same return signature as
    tauleap().  In "stream" mode dt_rec must be a multiple of t_sym.
    """
    MU, THETA, K, ETA, GP, EPS, ETA1 = M.MU, M.THETA, M.K, M.ETA, M.GP, M.EPS, M.ETA1
    rng = np.random.default_rng(seed)
    c0_fn, c1_fn = _const(c0_fn), _const(c1_fn)
    l0, l1 = c0_fn(0.0), c1_fn(0.0)
    Y0, Z10, Z20, Z30 = _init_counts(scheme, alpha, motif, omega, l0, l1, gc)
    ys_lin = y_equilibrium(scheme, alpha, l0, l1) if linear_zeta is not None else None
    prod_z1 = (EPS * MU if motif == "CAIF" else MU) * omega
    prod_z3 = (1.0 - EPS) * MU * omega
    seg = t_sym if drive == "stream" else dt_rec
    grid = np.arange(0.0, t_end + 1e-9, seg)
    rec_every = max(1, int(round(dt_rec / seg)))
    rec_idx = np.array([i for i in range(len(grid)) if grid[i] >= t_burn - 1e-9 and i % rec_every == 0])
    rec_set = set(rec_idx.tolist())
    if drive == "stream" and levels is None:
        levels, _ = make_levels(rng, n_traj, len(grid) + 1, r=l1 / l0, kind=stream_kind, isi_taps=isi_taps)
    y_all = np.zeros((len(rec_idx), n_traj))
    c0_rec = np.array([c0_fn(t) for t in grid[rec_idx]])
    c1_rec = np.array([c1_fn(t) for t in grid[rec_idx]])

    for tr in range(n_traj):
        Y, Z1, Z2, Z3 = (int(round(Y0)), int(round(Z10)), int(round(Z20)), int(round(Z30)))
        t = 0.0
        k_rec = 0
        for i_g in range(len(grid) - 1):
            t_next = grid[i_g + 1]
            c0, c1 = c0_fn(grid[i_g]), c1_fn(grid[i_g])
            c_now = levels[tr, min(i_g, levels.shape[1] - 1)] * c0 if drive == "stream" else None
            while True:
                y = Y / omega
                if linear_zeta is not None:
                    g = 1.0 / (1.0 + np.exp(-4.0 * linear_zeta * (y - ys_lin)))
                elif drive == "stream":
                    g = min(max(float(theta_B(scheme, c_now, y, alpha)), 0.0), 1.0)
                else:
                    g = min(max(float(sensed(scheme, c0, c1, y, alpha)), 0.0), 1.0)
                a_sens = THETA * g * omega
                a_cat = K * Z1
                a_dY = GP * Y
                a_sq = ETA / omega * Z1 * Z2
                a_g1 = gc * Z1
                a_g2 = gc * Z2
                a_sq2 = ETA1 / omega * Z2 * Z3 if motif == "CAIF" else 0.0
                a_g3 = gc * Z3 if motif == "CAIF" else 0.0
                a_p3 = prod_z3 if motif == "CAIF" else 0.0
                a0 = prod_z1 + a_sens + a_cat + a_dY + a_sq + a_g1 + a_g2 + a_p3 + a_sq2 + a_g3
                tau = -np.log(rng.random()) / a0
                if t + tau >= t_next:
                    t = t_next
                    break
                t += tau
                r = rng.random() * a0
                if r < prod_z1:
                    Z1 += 1
                elif r < prod_z1 + a_sens:
                    Z2 += 1
                elif r < prod_z1 + a_sens + a_cat:
                    Y += 1
                elif r < prod_z1 + a_sens + a_cat + a_dY:
                    Y -= 1
                elif r < prod_z1 + a_sens + a_cat + a_dY + a_sq:
                    Z1 -= 1; Z2 -= 1
                elif r < prod_z1 + a_sens + a_cat + a_dY + a_sq + a_g1:
                    Z1 -= 1
                elif r < prod_z1 + a_sens + a_cat + a_dY + a_sq + a_g1 + a_g2:
                    Z2 -= 1
                elif r < prod_z1 + a_sens + a_cat + a_dY + a_sq + a_g1 + a_g2 + a_p3:
                    Z3 += 1
                elif r < prod_z1 + a_sens + a_cat + a_dY + a_sq + a_g1 + a_g2 + a_p3 + a_sq2:
                    Z2 -= 1; Z3 -= 1
                else:
                    Z3 -= 1
                Y = max(Y, 0); Z1 = max(Z1, 0); Z2 = max(Z2, 0); Z3 = max(Z3, 0)
            if (i_g + 1) in rec_set:
                y_all[k_rec, tr] = Y / omega
                k_rec += 1
    return grid[rec_idx], y_all, c0_rec, c1_rec, 0.0


def run(scheme, alpha, c0_fn, c1_fn, *, method="tauleap", **kw):
    if method == "tauleap":
        return tauleap(scheme, alpha, c0_fn, c1_fn, **kw)
    kw.pop("dt", None)
    return exact_ssa(scheme, alpha, c0_fn, c1_fn, **kw)
