"""
run_zeta_crit.py -- validation of the exact stability boundary (Section III-D, Fig. 6(c)).

Part A: at the calibration alpha = 1, y* = 1, the receptor gain zeta is varied independently of the receptor map with
the logistic sensing function 1/(1 + exp(-4 zeta (y - y*))), which has the slope zeta at y*.  For Omega = 200 and 400,
the script records the excess kurtosis of the modulator fluctuation (0 for a Gaussian fluctuation and -1.5 for a
sinusoidal limit cycle) and the Omega-scaled modulator variance propagated to the occupancy, which is independent of
Omega for a stationary fluctuation.  The onset of oscillation is compared with
zeta_crit = gamma_p (gamma_p + eta z_Sigma) / (k theta rho).
Part B: the actuator gain alpha is swept for the non-cooperative and cooperative receivers with the full receptor map,
to check that no loop below zeta_crit(alpha) oscillates.
The position of the normalized two-stage reference S_f = 1 of Section III-D is stored for Fig. 6(c).

Outputs: results/zeta_crit.json, results/zeta_crit_sweep.csv, and results/zeta_crit_nonlinear.csv
"""
from __future__ import annotations

import json, os, sys
import numpy as np
from scipy.stats import kurtosis

sys.path.insert(0, os.path.dirname(__file__))
from model import (steady_state, rho_of, zeta_crit_closed_form, zeta_star,
                   sf_reference, sensed, stable_seed, CL0_NOM, CL1_NOM, GP, K, THETA)
from engine import tauleap

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def sweep_linear(alpha=1.0, omegas=(200.0, 400.0), n_traj=48, t_burn=250.0,
                 t_win=3000.0, dt=0.02, dt_rec=0.5, seed=17):
    ss = steady_state("allosteric", alpha)
    zc = zeta_crit_closed_form(ss)
    rho = rho_of(ss)
    zetas = zc * np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.2, 1.5, 2.0])
    rows = []
    for z in zetas:
        rec = {}
        for om in omegas:
            t, y, _, _, clip = tauleap("allosteric", alpha, CL0_NOM, CL1_NOM, omega=om,
                                       n_traj=n_traj, t_end=t_burn + t_win, dt=dt,
                                       t_burn=t_burn, dt_rec=dt_rec,
                                       seed=stable_seed("zc", z, om, base=seed),
                                       linear_zeta=z, drive="bitavg")
            g = 0.5 + z * (y - ss["y"])                 # linearized output statistic zeta (y - y*) + 1/2
            g_log = 1.0 / (1.0 + np.exp(-4.0 * z * (y - ss["y"])))   # the sensed occupancy actually simulated (bounded)
            rec[om] = dict(var=float(g.var()), kurt=float(kurtosis(g.ravel(), fisher=True)),
                           var_log=float(g_log.var()), g_min=float(g_log.min()), g_max=float(g_log.max()),
                           clip=clip)
        rows.append(dict(zeta=float(z), zeta_over_crit=float(z / zc),
                         sf_reference=sf_reference(z, rho),
                         var200=rec[200.0]["var"], var400=rec[400.0]["var"],
                         varOmega200=rec[200.0]["var"] * 200, varOmega400=rec[400.0]["var"] * 400,
                         varLogOmega200=rec[200.0]["var_log"] * 200, varLogOmega400=rec[400.0]["var_log"] * 400,
                         varLog200=rec[200.0]["var_log"], varLog400=rec[400.0]["var_log"],
                         gmin200=rec[200.0]["g_min"], gmax200=rec[200.0]["g_max"],
                         gmin400=rec[400.0]["g_min"], gmax400=rec[400.0]["g_max"],
                         kurt200=rec[200.0]["kurt"], kurt400=rec[400.0]["kurt"]))
        r = rows[-1]
        print(f"  zeta={z:7.3f} (z/zc={r['zeta_over_crit']:.2f}, S_f={r['sf_reference']:.2f})  "
              f"zeta^2 Var(y)*Om: {r['varOmega200']:9.3g} {r['varOmega400']:9.3g}  Var(g_log): {r['varLog200']:.3f} {r['varLog400']:.3f}  "
              f"kurt: {r['kurt200']:+.2f} {r['kurt400']:+.2f}")
    return zc, rho, rows


def sweep_nonlinear(scheme, alphas, omega=200.0, n_traj=32, t_burn=250.0,
                    t_win=1500.0, dt=0.02, dt_rec=0.5, seed=23):
    rows = []
    for a in alphas:
        ss = steady_state(scheme, a)
        z = zeta_star(scheme, a)
        zc = zeta_crit_closed_form(ss)
        t, y, _, _, _ = tauleap(scheme, a, CL0_NOM, CL1_NOM, omega=omega, n_traj=n_traj,
                                t_end=t_burn + t_win, dt=dt, t_burn=t_burn, dt_rec=dt_rec,
                                seed=stable_seed("zcnl", scheme, a, base=seed), drive="bitavg")
        g = sensed(scheme, CL0_NOM, CL1_NOM, y, a)
        rows.append(dict(scheme=scheme, alpha=float(a), zeta=float(z), zeta_crit=float(zc),
                         margin=float(zc / z), varOmega=float(g.var() * omega),
                         kurt=float(kurtosis(g.ravel(), fisher=True)),
                         mean_g=float(g.mean())))
        r = rows[-1]
        print(f"  {scheme:12s} alpha={a:5.2f} zeta*={z:.3f} zeta_crit={zc:.3f} "
              f"margin={r['margin']:6.2f}  Var*Om={r['varOmega']:.3g}  kurt={r['kurt']:+.2f}  <g>={r['mean_g']:.4f}")
    return rows


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    kw = dict(n_traj=16, t_win=800.0) if quick else {}
    print("Part A: linearized-sensing zeta sweep at alpha = 1 (y* = 1)")
    zc, rho, rows = sweep_linear(**kw)
    zeta_sf1 = 2.0 * GP ** 3 / (K * THETA * rho)      # normalized two-stage reference S_f = 1: k rho theta zeta / 2 = gp^3 (Olsman et al., 2019)
    print(f"  zeta_crit = {zc:.4f}  (rho = {rho:.4f});  the normalized two-stage reference S_f = 1 lies at "
          f"zeta = {zeta_sf1:.3f}, i.e., zeta/zeta_crit = {zeta_sf1 / zc:.3f}")
    with open(os.path.join(OUT, "zeta_crit_sweep.csv"), "w") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r.values()) + "\n")

    print("\nPart B: full nonlinear receiver, alpha sweep")
    nl = []
    kw2 = dict(n_traj=12, t_win=600.0) if quick else {}
    nl += sweep_nonlinear("allosteric", [0.3, 0.6, 1.0, 1.6, 3.0, 6.0, 12.0], **kw2)
    nl += sweep_nonlinear("cooperative", [0.3, 0.6, 1.0, 1.6, 3.0, 6.0], **kw2)
    with open(os.path.join(OUT, "zeta_crit_nonlinear.csv"), "w") as f:
        f.write(",".join(nl[0].keys()) + "\n")
        for r in nl:
            f.write(",".join(str(v) if isinstance(v, str) else f"{v:.6g}" for v in r.values()) + "\n")

    json.dump(dict(alpha_ref=1.0, zeta_crit=zc, rho=rho,
                   zeta_at_sf1=zeta_sf1, ratio_sf1=zeta_sf1 / zc, linear=rows, nonlinear=nl),
              open(os.path.join(OUT, "zeta_crit.json"), "w"), indent=1)
    print("\nwrote results/zeta_crit_sweep.csv, zeta_crit_nonlinear.csv, zeta_crit.json")
