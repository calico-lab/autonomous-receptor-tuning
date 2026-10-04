"""
run_bounded.py -- the bounded actuation law (Section IV-B).

An effector that binds an allosteric site of the receptor gives the bounded actuation law of Section II-D,
    K_D(y) = K_base (1 + y/K_Y) / (1 + chi y/K_Y),   K_base/chi <= K_D <= K_base,
with the local actuator gain beta(y) = (chi - 1) K_Y / [(K_Y + y)(K_Y + chi y)].  This script calibrates the bounded
law to the loop gain of the non-cooperative design (y* = 1.5, since beta* y* must stay below
(sqrt(chi) - 1)/(sqrt(chi) + 1)), verifies the equilibrium, the modulator noise, and the BEP on the static channel, and
sweeps the variation depth at T_c = 240, which shows that the decreasing actuator gain limits the tracking.

Output: results/bounded.json
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from engine import tauleap
from detection import bep_from_record, bootstrap_ci, best_static_receiver, oracle_bep
from run_headline import gain_fn

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
DP = json.load(open(os.path.join(OUT, "design_points.json")))["design"]
CHI = 100.0
Y_STAR = 1.5


def nominal(alpha, n_traj, seed=5, t_burn=200.0, t_win=400.0, dt=0.01):
    t, y, c0, c1, clip = tauleap("allosteric", alpha, M.CL0_NOM, M.CL1_NOM, omega=M.OMEGA_HC, n_traj=n_traj,
                                 t_end=t_burn + t_win, dt=dt, t_burn=t_burn, dt_rec=1.0, seed=seed)
    bep, per = bep_from_record("allosteric", alpha, y, c0, c1)
    lo, hi = bootstrap_ci(per, seed=seed)
    kd = M.kd_of_y(y, alpha)
    sur = M.bep_surrogate("allosteric", alpha)
    return dict(kd_mean=float(kd.mean()), y_mean=float(y.mean()), var_dy_omega=float(y.var() * M.OMEGA_HC),
                var_dy_analytic=sur["var_dy"], zeta=sur["zeta"], beta=sur["beta"], wc=sur["wc"], tau=sur["tau"],
                rho=sur["rho"], y_star=sur["ys"], bep=bep, lo=lo, hi=hi, clip=clip)


def swing(alpha, S, T, n_traj, seed=3, n_periods=2, t_burn=200.0, dt=0.01):
    g = gain_fn(S, T)
    c0 = lambda t: M.CL0_NOM * g(t)
    c1 = lambda t: M.CL1_NOM * g(t)
    t, y, c0r, c1r, clip = tauleap("allosteric", alpha, c0, c1, omega=M.OMEGA_HC, n_traj=n_traj,
                                   t_end=t_burn + n_periods * T, dt=dt, t_burn=t_burn, dt_rec=1.0,
                                   seed=M.stable_seed("bd", S, T, M.ACT["law"], base=seed))
    bep, per = bep_from_record("allosteric", alpha, y, c0r, c1r)
    lo, hi = bootstrap_ci(per, seed=seed)
    st = best_static_receiver(c0r, c1r, n_hill=1.0)
    kd = M.kd_of_y(y, alpha)
    kd_opt = np.sqrt(c0r * c1r)[:, None]
    err = float(np.mean(np.abs(np.log(kd.mean(axis=1) / kd_opt[:, 0]))))
    return dict(S=S, T=T, adaptive=bep, lo=lo, hi=hi, static_best=st["bep"], gain=st["bep"] / bep,
                track_err_mean_ln=err, min_target=float(2.0 / S), clip=clip)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n_nom, n_sw = (200, 16) if quick else (1000, 48)
    swings = [1.0, 5.0, 20.0, 100.0] if quick else [1.0, 2.0, 5.0, 20.0, 50.0, 100.0]
    T = 240.0
    t0 = time.time()
    a_exp = DP["allosteric"]["alpha"]
    # bounded law calibrated to the same loop gain as the exponential design
    ss_exp = M.steady_state("allosteric", a_exp)
    loop_gain = M.K * M.rho_of(ss_exp) * M.THETA * M.zeta_star("allosteric", a_exp)
    z1, z2 = M.GP * Y_STAR / M.K, M.MU / (M.ETA * M.GP * Y_STAR / M.K)
    rho_b = z1 / (z1 + z2)
    zeta_b = loop_gain / (M.K * rho_b * M.THETA)
    th0 = 1.0 / (1.0 + (M.CL1_NOM / M.CL0_NOM) ** 0.5)
    beta_b = zeta_b / (th0 * (1 - th0))
    cal = M.atcm_calibrate(Y_STAR, beta_b, CHI)
    print(f"bounded law: chi={CHI:g}, y*={Y_STAR}, beta*={beta_b:.4f}, K_base={cal['kbase']:.3f}, K_Y={cal['ky']:.2f}, "
          f"range [{cal['kd_min']:.3f}, {cal['kd_max']:.3f}]")
    res = dict(chi=CHI, y_star=Y_STAR, beta_star=beta_b, calibration=cal, T=T, laws={})
    for law in ("exp", "atcm"):
        if law == "exp":
            M.set_actuator("exp", kbase=M.KBASE); alpha = a_exp
        else:
            M.set_actuator("atcm", kbase=cal["kbase"], ky=cal["ky"], chi=CHI); alpha = beta_b
        nom = nominal(alpha, n_nom)
        print(f"[{law}] nominal: K_D={nom['kd_mean']:.3f} y*={nom['y_star']:.3f} (<y>={nom['y_mean']:.3f}) zeta*={nom['zeta']:.4f} "
              f"beta*={nom['beta']:.3f} w_c={nom['wc']:.4f} tau={nom['tau']:.1f} Var(dy)Om={nom['var_dy_omega']:.2f} "
              f"(LNA {nom['var_dy_analytic']:.2f}) BEP={nom['bep']:.2e} [{nom['lo']:.1e},{nom['hi']:.1e}]", flush=True)
        rows = []
        for S in swings:
            r = swing(alpha, S, T, n_sw)
            rows.append(r)
            print(f"   S={S:5.1f}: adaptive {r['adaptive']:.2e} [{r['lo']:.1e},{r['hi']:.1e}]  static {r['static_best']:.2e}  "
                  f"gain {r['gain']:.3g}  tracking err {100 * (np.exp(r['track_err_mean_ln']) - 1):.0f} %  "
                  f"min target {r['min_target']:.3f}" + ("  [below range]" if law == "atcm" and r["min_target"] < cal["kd_min"] else ""),
                  flush=True)
        res["laws"][law] = dict(alpha=alpha, nominal=nom, swings=rows)
    M.set_actuator("exp", kbase=M.KBASE)
    json.dump(res, open(os.path.join(OUT, "bounded.json"), "w"), indent=1, default=float)
    print(f"\nwrote results/bounded.json  ({time.time() - t0:.0f} s)")
