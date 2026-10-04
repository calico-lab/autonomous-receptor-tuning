"""
run_leaky_stochastic.py -- leak tolerance of the noisy receiver by stochastic simulation (Section III-F).

run_leaky.py gives the deterministic equilibrium offset and a Gaussian estimate of the leak penalty.  This script
measures the penalty directly: the receiver is simulated with random symbol streams at Omega = 200 for a set of
controller dilution rates gamma_c, with the same symbol streams and seeds for every gamma_c (common random numbers),
and the ratio P_e(gamma_c) / P_e(0) is estimated with a paired bootstrap over trajectories.  The mean affinity, the
mean sensed occupancy, and the modulator variance are recorded, together with the deterministic and Gaussian
predictions of run_leaky.py.

Output: results/leaky_stochastic.json
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from engine import tauleap, make_levels, n_symbols
from run_leaky import row as leaky_row

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
DP = json.load(open(os.path.join(OUT, "design_points.json")))["design"]
GCS = {"allosteric": (0.0, 1e-3, 2e-3, 5e-3, 1e-2, 1.5e-2, 2e-2, 3e-2),
       "cooperative": (0.0, 5e-4, 1.3e-3, 3e-3, 5e-3, 1e-2, 2e-2)}


def per_traj_bep(scheme, alpha, y):
    m = M.N_RECEPTORS // 2
    th0 = np.clip(M.theta_B(scheme, M.CL0_NOM, y, alpha), 1e-12, 1 - 1e-12)
    th1 = np.clip(M.theta_B(scheme, M.CL1_NOM, y, alpha), 1e-12, 1 - 1e-12)
    e0 = binom.sf(m, M.N_RECEPTORS, th0) + 0.5 * binom.pmf(m, M.N_RECEPTORS, th0)
    e1 = binom.cdf(m - 1, M.N_RECEPTORS, th1) + 0.5 * binom.pmf(m, M.N_RECEPTORS, th1)
    return (0.5 * (e0 + e1)).mean(axis=0)          # (n_traj,)


def paired_bootstrap(p_ref, p, n_boot=4000, seed=1):
    rng = np.random.default_rng(seed)
    n = len(p); idx = rng.integers(0, n, (n_boot, n))
    r = p[idx].mean(axis=1) / p_ref[idx].mean(axis=1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def bootstrap(p, n_boot=4000, seed=2):
    rng = np.random.default_rng(seed)
    n = len(p); idx = rng.integers(0, n, (n_boot, n))
    b = p[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def sweep(scheme, alpha, gcs, n_traj, t_end=600.0, t_burn=200.0, dt=0.01, seed=41):
    T = M.T_SYM
    rng = np.random.default_rng(seed)
    levels, _ = make_levels(rng, n_traj, n_symbols(t_end, T))
    out = {}; p_ref = None
    for gc in gcs:
        t0 = time.time()
        t, y, _, _, clip = tauleap(scheme, alpha, M.CL0_NOM, M.CL1_NOM, omega=M.OMEGA_HC, gc=gc * M.GP, n_traj=n_traj,
                                   t_end=t_end, dt=dt, t_burn=t_burn, dt_rec=1.0, seed=seed, drive="stream",
                                   levels=levels, t_sym=T)
        p = per_traj_bep(scheme, alpha, y)
        if p_ref is None:
            p_ref = p
        lo, hi = bootstrap(p)
        rlo, rhi = paired_bootstrap(p_ref, p)
        det = leaky_row(scheme, alpha, "AIF", gc * M.GP)
        rec = dict(gc_over_gp=gc, bep=float(p.mean()), lo=lo, hi=hi, ratio=float(p.mean() / p_ref.mean()),
                   ratio_lo=rlo, ratio_hi=rhi, mean_kd=float(M.kd_of_y(y, alpha).mean()),
                   mean_g=float(M.sensed(scheme, M.CL0_NOM, M.CL1_NOM, y, alpha).mean()),
                   var_dy_omega=float(y.var() * M.OMEGA_HC), clip=clip, n_traj=n_traj,
                   det_bep=det.get("bep_det"), det_kd=det.get("kd"), gauss_bep=det.get("bep_gauss"))
        out[f"{gc:g}"] = rec
        print(f"  {scheme:12s} gamma_c/gamma_p={gc:<7g} BEP={rec['bep']:.3e} [{lo:.2e},{hi:.2e}]  "
              f"R={rec['ratio']:.3f} [{rlo:.3f},{rhi:.3f}]  <K_D>={rec['mean_kd']:.3f}  <g>={rec['mean_g']:.4f}  "
              f"Var(y)Om={rec['var_dy_omega']:.1f}  det R={det.get('bep_det', float('nan')) / out[f'{gcs[0]:g}']['det_bep']:.3f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    return out


def tolerance_from_sweep(rows, penalty=1.1):
    """gamma_c/gamma_p at which the measured ratio first reaches `penalty` (log interpolation)."""
    pts = sorted((r["gc_over_gp"], r["ratio"]) for r in rows.values() if r["gc_over_gp"] > 0)
    prev = None
    for gc, r in pts:
        if r >= penalty and prev is not None and prev[1] < penalty:
            g0, r0 = prev
            return float(np.exp(np.log(g0) + (penalty - r0) / (r - r0) * (np.log(gc) - np.log(g0))))
        prev = (gc, r)
    return None


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    schemes = [a for a in sys.argv[1:] if not a.startswith("--")] or ["allosteric", "cooperative"]
    n_traj = 200 if quick else 2000
    path = os.path.join(OUT, "leaky_stochastic.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    for scheme in schemes:
        alpha = DP[scheme]["alpha"]
        print(f"{scheme}: alpha={alpha:.3f}, Omega={M.OMEGA_HC:g}, {n_traj} trajectories, common streams")
        rows = sweep(scheme, alpha, GCS[scheme], n_traj)
        res[scheme] = dict(alpha=alpha, rows=rows, tol_ten_pct=tolerance_from_sweep(rows, 1.1),
                           tol_twofold=tolerance_from_sweep(rows, 2.0))
        print(f"  measured tolerance (10 %): {res[scheme]['tol_ten_pct']}   (twofold): {res[scheme]['tol_twofold']}")
        json.dump(res, open(path, "w"), indent=1, default=float)
    print(f"\nwrote results/leaky_stochastic.json  ({time.time() - t0:.0f} s)")
