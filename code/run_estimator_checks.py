"""
run_estimator_checks.py -- sampling convention and convergence of the BEP estimator (Section IV).

The modulator concentration is recorded at the symbol boundaries, where it is independent of the bit of the symbol
that starts there, such that the average of the two conditional binomial tails is exact for a detector that samples
at the start of a symbol.  A detector that samples later in the symbol observes a modulator that has already begun to
respond to the current bit.  For every receiver, this script compares the marginal estimate with the conditional
estimates at the start and at the end of the symbol and at intermediate detection phases.  It also runs the loop with
linearized sensing driven by the exact piecewise-constant symbol switching, which isolates the nonlinear part of the
deviation of the modulator variance from the linear-noise prediction (Section IV-C1).

Output: results/estimator.json
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from engine import tauleap, make_levels, n_symbols

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
DP = json.load(open(os.path.join(OUT, "design_points.json")))["design"]
CONFIGS = [("allosteric", "allosteric", DP["allosteric"]["alpha"], M.K),
           ("cooperative", "cooperative", DP["cooperative"]["alpha"], M.K),
           ("coregulated", "coregulated", DP["coregulated"]["alpha"], M.K),
           ("coregulated_slow", "coregulated", 1.0, M.K_COREG_SLOW),
           ("weighted", "weighted", DP["weighted"]["alpha"], M.K),
           ("cooperative_a1", "cooperative", 1.0, M.K)]


def conditional_check(scheme, alpha, n_traj, t_burn=200.0, t_win=300.0, dt=0.01, seed=31):
    t_end = t_burn + t_win
    T = M.T_SYM
    rng = np.random.default_rng(seed)
    levels, bits = make_levels(rng, n_traj, n_symbols(t_end, T))
    t, y, _, _, clip = tauleap(scheme, alpha, M.CL0_NOM, M.CL1_NOM, omega=M.OMEGA_HC, n_traj=n_traj, t_end=t_end,
                               dt=dt, t_burn=t_burn, dt_rec=T, seed=seed, drive="stream", levels=levels, t_sym=T)
    j = (t / T + 1e-9).astype(int)
    b_new = bits[:, np.minimum(j, bits.shape[1] - 1)].T
    b_prev = bits[:, np.maximum(j - 1, 0)].T
    m = M.N_RECEPTORS // 2
    if scheme == "weighted":
        from detection import bep_mixed_exact
        # conditional terms for the mixed array: evaluate both conditional errors through the exact two-cohort tails
        from detection import _mixed_tail
        n_t = int(round(M.W_WEIGHT * M.N_RECEPTORS)); n_f = M.N_RECEPTORS - n_t
        kd = M.kd_of_y(y, alpha)
        tt0, tt1 = M.CL0_NOM / (M.CL0_NOM + kd), M.CL1_NOM / (M.CL1_NOM + kd)
        tf0, tf1 = M.CL0_NOM / (M.CL0_NOM + M.K_FIX), M.CL1_NOM / (M.CL1_NOM + M.K_FIX)
        gt0, eq0 = _mixed_tail(tt0, tf0, n_t, n_f, m); gt1, eq1 = _mixed_tail(tt1, tf1, n_t, n_f, m)
        e0 = gt0 + 0.5 * eq0; e1 = (1.0 - gt1 - eq1) + 0.5 * eq1
    else:
        th0 = np.clip(M.theta_B(scheme, M.CL0_NOM, y, alpha), 1e-12, 1 - 1e-12)
        th1 = np.clip(M.theta_B(scheme, M.CL1_NOM, y, alpha), 1e-12, 1 - 1e-12)
        e0 = binom.sf(m, M.N_RECEPTORS, th0) + 0.5 * binom.pmf(m, M.N_RECEPTORS, th0)
        e1 = binom.cdf(m - 1, M.N_RECEPTORS, th1) + 0.5 * binom.pmf(m, M.N_RECEPTORS, th1)
    marg = 0.5 * (e0 + e1)
    cond_start = np.where(b_new == 0, e0, e1)
    cond_end = np.where(b_prev == 0, e0, e1)
    return dict(marginal=float(marg.mean()), cond_start=float(cond_start.mean()), cond_end=float(cond_end.mean()),
                ratio_start=float(cond_start.mean() / marg.mean()), ratio_end=float(cond_end.mean() / marg.mean()),
                n_traj=n_traj, clip=clip)


def phase_sweep(scheme, alpha, n_traj, t_burn=200.0, t_win=300.0, dt=0.01, seed=37, n_phase=5):
    """Conditional BEP at detection phases 0, 1/5, ..., 4/5 of the symbol (and 1, the next boundary),
    relative to the marginal estimate at the symbol boundary."""
    t_end = t_burn + t_win
    T = M.T_SYM; dt_rec = T / n_phase
    rng = np.random.default_rng(seed)
    levels, bits = make_levels(rng, n_traj, n_symbols(t_end, T))
    t, y, _, _, clip = tauleap(scheme, alpha, M.CL0_NOM, M.CL1_NOM, omega=M.OMEGA_HC, n_traj=n_traj, t_end=t_end,
                               dt=dt, t_burn=t_burn, dt_rec=dt_rec, seed=seed, drive="stream", levels=levels, t_sym=T)
    j = np.floor(t / T + 1e-9).astype(int)
    ph = np.round((t - j * T) / T * n_phase).astype(int)          # 0 .. n_phase-1
    b_cur = bits[:, np.minimum(j, bits.shape[1] - 1)].T
    b_prev = bits[:, np.maximum(j - 1, 0)].T
    m = M.N_RECEPTORS // 2
    th0 = np.clip(M.theta_B(scheme, M.CL0_NOM, y, alpha), 1e-12, 1 - 1e-12)
    th1 = np.clip(M.theta_B(scheme, M.CL1_NOM, y, alpha), 1e-12, 1 - 1e-12)
    e0 = binom.sf(m, M.N_RECEPTORS, th0) + 0.5 * binom.pmf(m, M.N_RECEPTORS, th0)
    e1 = binom.cdf(m - 1, M.N_RECEPTORS, th1) + 0.5 * binom.pmf(m, M.N_RECEPTORS, th1)
    marg0 = float((0.5 * (e0 + e1))[ph == 0].mean())
    out = {}
    for k in range(n_phase):
        sel = ph == k
        cond = np.where(b_cur[sel] == 0, e0[sel], e1[sel])
        out[f"{k / n_phase:g}"] = float(cond.mean() / marg0)
    sel = ph == 0
    cond_end = np.where(b_prev[sel] == 0, e0[sel], e1[sel])
    out["1"] = float(cond_end.mean() / marg0)
    return dict(marginal=marg0, ratios=out, max_dev_pct=float(100 * max(abs(v - 1) for v in out.values())), clip=clip)


def linear_stream(scheme, alpha, n_traj, t_end=600.0, t_burn=200.0, dt=0.01, seed=5):
    """Linearized sensing (slope zeta* at y*) driven by the exact piecewise-constant symbol switching."""
    rng = np.random.default_rng(seed)
    ss = M.steady_state(scheme, alpha); ys = ss["y"]; z = M.zeta_star(scheme, alpha)
    th0, th1 = M.occ_levels(scheme, alpha, ys); dth = th1 - th0
    n_sym = int(np.ceil(t_end / M.T_SYM)) + 2
    bits = rng.integers(0, 2, (n_traj, n_sym))
    om = M.OMEGA_HC
    Y = np.full(n_traj, ys * om); Z1 = np.full(n_traj, ss["z1"] * om); Z2 = np.full(n_traj, ss["z2"] * om)
    P = rng.poisson; rec = []
    for it in range(int(round(t_end / dt)) + 1):
        t = it * dt; j = min(int(t / M.T_SYM), n_sym - 1); y = Y / om
        g = np.clip(0.5 + z * (y - ys) + dth * (bits[:, j] - 0.5), 0.0, 1.0)
        n_pz1 = P(M.MU * om * dt, n_traj); n_pz2 = P(M.THETA * g * om * dt)
        n_cat = P(M.K * Z1 * dt); n_dY = P(M.GP * Y * dt); n_sq = P(np.clip(M.ETA / om * Z1 * Z2, 0, None) * dt)
        Y = np.clip(Y + n_cat - n_dY, 0, None); Z1 = np.clip(Z1 + n_pz1 - n_sq, 0, None); Z2 = np.clip(Z2 + n_pz2 - n_sq, 0, None)
        if t >= t_burn - 1e-9 and it % 100 == 0:
            rec.append(Y / om)
    y = np.array(rec)
    return float(y.var() * om)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n_traj = 200 if quick else 2000
    res = {}
    t0 = time.time()
    print(f"{'config':18s}{'marginal':>11}{'cond start':>12}{'ratio':>7}{'cond end':>11}{'ratio':>7}{'Var linear':>13}{'LNA+SN':>10}")
    for label, scheme, alpha, kk in CONFIGS:
        M.set_rates(K=kk)
        r = conditional_check(scheme, alpha, n_traj)
        r["var_linear_stream"] = linear_stream(scheme, alpha, min(n_traj, 1000))
        if label in ("allosteric", "cooperative", "coregulated", "coregulated_slow"):
            r["phase"] = phase_sweep(scheme, alpha, min(n_traj, 1000))
            print(f"   phase sweep {label}: " + "  ".join(f"{k}:{v:.3f}" for k, v in r["phase"]["ratios"].items()) +
                  f"   max dev {r['phase']['max_dev_pct']:.1f} %", flush=True)
        sur = M.bep_surrogate(scheme, alpha)
        r["var_lna_bits"] = sur["var_dy"]
        M.set_rates()
        res[label] = r
        print(f"{label:18s}{r['marginal']:>11.3e}{r['cond_start']:>12.3e}{r['ratio_start']:>7.3f}{r['cond_end']:>11.3e}"
              f"{r['ratio_end']:>7.3f}{r['var_linear_stream']:>13.1f}{r['var_lna_bits']:>10.1f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "estimator.json"), "w"), indent=1, default=float)
    print(f"\nwrote results/estimator.json  ({time.time() - t0:.0f} s)")
