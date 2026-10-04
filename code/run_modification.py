"""
run_modification.py -- explicit receptor modification level (Section IV-B).

The exponential actuation law K_D = K_base exp(-alpha y) treats the modulator concentration y as the quantity that
shifts the binding free energy.  Physically, the free energy is shifted by the modification level m of the receptor,
e.g., the number of methylated residues, which follows the modulator through
    dm/dt = k_m y (M - m) - k_d m,        K_D = K_0 exp(-a m),
with M modification sites, the modification rate k_m, and the removal rate k_d.  For m << M and a removal rate that is
fast compared with the loop, m ~ (k_m M / k_d) y, and the law reduces to the exponential law with
alpha = a k_m M / k_d (Section II-D).  This script simulates the loop with m as an additional state (deterministic per
trajectory, since the receptor population is large) for the removal rates k_d = 5 gamma_p, gamma_p, and 0.2 gamma_p,
and for a saturating case with M = 10 sites, calibrated to the actuator gain and y* of the non-cooperative design.
It reports the equilibrium affinity, the variances of the modulator and of the modification level, the BEP on the
static channel, and the BEP reduction factor at S = 5 and T_c = 240.

Output: results/modification.json
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from engine import make_levels, n_symbols
from detection import bep_exact, bootstrap_ci, best_static_receiver
from run_headline import gain_fn

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
DP = json.load(open(os.path.join(OUT, "design_points.json")))["design"]
ALPHA = DP["allosteric"]["alpha"]
M_SITES = 100.0


def simulate(kd_rate, n_traj, t_end, t_burn, dt, seed, S=None, T=None, M_SITES=M_SITES):
    """Tau-leap loop with the modification state; returns recorded (y, m, c0, c1)."""
    rng = np.random.default_rng(seed)
    km = kd_rate / M_SITES                      # k_m M/k_d = 1 -> m* = y* in the linear regime
    a = ALPHA                                   # K_D = K_base exp(-a m)
    ss = M.steady_state("allosteric", ALPHA); ys = ss["y"]
    om = M.OMEGA_HC
    Y = np.full(n_traj, ys * om); Z1 = np.full(n_traj, ss["z1"] * om); Z2 = np.full(n_traj, ss["z2"] * om)
    m = np.full(n_traj, km * M_SITES * ys / (kd_rate + km * ys))
    levels, _ = make_levels(rng, n_traj, n_symbols(t_end))
    g = gain_fn(S, T) if S else (lambda t: 1.0)
    P = rng.poisson
    rec_y, rec_m, rec_c0, rec_c1 = [], [], [], []
    n_sym = levels.shape[1]
    for it in range(int(round(t_end / dt)) + 1):
        t = it * dt; gt = g(t)
        j = min(int(t / M.T_SYM), n_sym - 1)
        kd = M.KBASE * np.exp(-a * m)
        c = levels[:, j] * M.CL0_NOM * gt
        occ = c / (c + kd)
        y = Y / om
        n_pz1 = P(M.MU * om * dt, n_traj); n_pz2 = P(M.THETA * occ * om * dt)
        n_cat = P(M.K * Z1 * dt); n_dY = P(M.GP * Y * dt); n_sq = P(np.clip(M.ETA / om * Z1 * Z2, 0, None) * dt)
        Y = np.clip(Y + n_cat - n_dY, 0, None); Z1 = np.clip(Z1 + n_pz1 - n_sq, 0, None); Z2 = np.clip(Z2 + n_pz2 - n_sq, 0, None)
        m = m + dt * (km * y * (M_SITES - m) - kd_rate * m)
        if t >= t_burn - 1e-9 and it % 100 == 0:
            rec_y.append(Y / om); rec_m.append(m.copy()); rec_c0.append(M.CL0_NOM * gt); rec_c1.append(M.CL1_NOM * gt)
    return np.array(rec_y), np.array(rec_m), np.array(rec_c0), np.array(rec_c1)


def score(mrec, c0r, c1r):
    kd = M.KBASE * np.exp(-ALPHA * mrec)
    th0 = c0r[:, None] / (c0r[:, None] + kd); th1 = c1r[:, None] / (c1r[:, None] + kd)
    b = bep_exact(th0, th1)
    per = b.mean(axis=0)
    return float(per.mean()), bootstrap_ci(per, seed=1), kd


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n_nom, n_sw = (200, 16) if quick else (1000, 48)
    res = dict(alpha=ALPHA, M_sites=M_SITES, cases={})
    t0 = time.time()
    for kd_rate, Msites in ((5 * M.GP, 100.0), (M.GP, 100.0), (0.2 * M.GP, 100.0), (5 * M.GP, 10.0)):
        y, mrec, c0r, c1r = simulate(kd_rate, n_nom, 600.0, 200.0, 0.01, seed=3, M_SITES=Msites)
        bep, (lo, hi), kd = score(mrec, c0r, c1r)
        y2, m2, c0r2, c1r2 = simulate(kd_rate, n_sw, 200.0 + 2 * 240.0, 200.0, 0.01, seed=4, S=5.0, T=240.0, M_SITES=Msites)
        bep_sw, (lo2, hi2), kd2 = score(m2, c0r2, c1r2)
        st = best_static_receiver(c0r2, c1r2, n_hill=1.0)
        kd_opt = np.sqrt(c0r2 * c1r2)[:, None]
        trk = float(np.mean(np.abs(np.log(kd2.mean(axis=1) / kd_opt[:, 0]))))
        # exact quasi-steady law at the mean modulator concentration: K_D = K_base exp[-alpha y/(1 + alpha y/(a M))] with a = alpha
        y_mean = float(y.mean()); kd_exact = float(M.KBASE * np.exp(-ALPHA * y_mean / (1.0 + ALPHA * y_mean / (ALPHA * Msites))))
        case = dict(kd_rate=kd_rate, kd_over_gp=kd_rate / M.GP, M_sites=Msites, a=ALPHA, km=kd_rate / Msites, aM=ALPHA * Msites,
                    m_over_M_mean=float(mrec.mean() / Msites), m_over_M_max=float(mrec.max() / Msites),
                    m_over_M_max_swing=float(m2.max() / Msites), y_mean=y_mean, kd_exact_law=kd_exact,
                    kd_mean=float(kd.mean()), var_y_omega=float(y.var() * M.OMEGA_HC),
                    var_m_omega=float(mrec.var() * M.OMEGA_HC), bep=bep, lo=lo, hi=hi, adaptive_S5=bep_sw, lo_S5=lo2, hi_S5=hi2,
                    static_S5=st["bep"], gain_S5=st["bep"] / bep_sw, track_err_ln=trk)
        key = f"{kd_rate / M.GP:g}" if Msites == 100.0 else f"{kd_rate / M.GP:g}|M{Msites:g}"
        res["cases"][key] = case
        print(f"k_d = {kd_rate / M.GP:g} gamma_p, M = {Msites:g}: mean K_D={kd.mean():.3f} (exact law {kd_exact:.3f})  m/M mean {case['m_over_M_mean']:.3f} max {case['m_over_M_max']:.3f} (swing {case['m_over_M_max_swing']:.3f})  Var(y)Om={y.var() * M.OMEGA_HC:.1f}  Var(m)Om={mrec.var() * M.OMEGA_HC:.1f}  "
              f"BEP={bep:.2e} [{lo:.1e},{hi:.1e}]  S=5,T=240: adaptive {bep_sw:.2e} static {st['bep']:.2e} gain {st['bep'] / bep_sw:.3g} "
              f"tracking err {100 * (np.exp(trk) - 1):.0f} %", flush=True)
    json.dump(res, open(os.path.join(OUT, "modification.json"), "w"), indent=1, default=float)
    print(f"\nwrote results/modification.json  ({time.time() - t0:.0f} s)")
