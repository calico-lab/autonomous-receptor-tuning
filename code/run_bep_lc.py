"""
run_bep_lc.py -- error probability as a function of the system size Omega (Section IV-C2, Fig. 6(b)).

Each receiver at its design point on the static channel, driven by a random symbol stream (T_sym = 0.5), for Omega
from 12 to 800.  Exact stochastic simulation (Gillespie direct method) is used for Omega <= 200 and tau-leaping
(dt = 0.005) above, and both methods are run at Omega = 45 and 200 as a cross-check.  The cooperative receptor at the
actuator gain of the co-regulated receiver (alpha = 1) is included as a control.  The cascaded variant (CAIF) is run
at Omega = 20 only and is not reported in the paper.

Output: results/bep_lc.json
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from engine import tauleap, exact_ssa
from detection import bep_from_record, bootstrap_ci, occupancy_record

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def ci(r):
    return "[%.2e,%.2e]" % (r["lo"], r["hi"])

DP = json.load(open(os.path.join(OUT, "design_points.json")))["design"]

OMEGAS = [12, 16, 22, 30, 45, 65, 90, 130, 200, 400, 800]
EXACT_MAX = 200
XCHECK = (45, 200)
CONFIGS = [("allosteric", "allosteric", DP["allosteric"]["alpha"], M.K),
           ("cooperative", "cooperative", DP["cooperative"]["alpha"], M.K),
           ("coregulated", "coregulated", DP["coregulated"]["alpha"], M.K),
           ("coregulated_slow", "coregulated", 1.0, M.K_COREG_SLOW),
           ("weighted", "weighted", DP["weighted"]["alpha"], M.K),
           ("cooperative_a1", "cooperative", 1.0, M.K)]


def measure(scheme, alpha, motif, omega, method, n_traj, t_burn=150.0, t_win=300.0,
            dt=0.005, dt_rec=1.0, seed=11):
    sd = M.stable_seed("lc", scheme, alpha, motif, omega, method, base=seed)
    if method == "exact":
        t, y, c0, c1, clip = exact_ssa(scheme, alpha, M.CL0_NOM, M.CL1_NOM, motif=motif,
                                       omega=omega, n_traj=n_traj, t_end=t_burn + t_win,
                                       t_burn=t_burn, dt_rec=dt_rec, seed=sd)
    else:
        t, y, c0, c1, clip = tauleap(scheme, alpha, M.CL0_NOM, M.CL1_NOM, motif=motif,
                                     omega=omega, n_traj=n_traj, t_end=t_burn + t_win, dt=dt,
                                     t_burn=t_burn, dt_rec=dt_rec, seed=sd)
    th0, th1 = occupancy_record(scheme, alpha, y, c0, c1)
    bep, per = bep_from_record(scheme, alpha, y, c0, c1)
    lo, hi = bootstrap_ci(per, seed=seed)
    sur = M.bep_surrogate(scheme, alpha, omega=omega)
    return dict(bep=bep, lo=lo, hi=hi, mean_g=float(0.5 * (th0 + th1).mean()),
                var_dy_omega=float(y.var() * omega), var_dy_analytic=sur["var_dy"],
                var_dy_analytic_chem=sur["var_dy_chem"], bep_surr=sur["bep"],
                mu0=float(th0.mean()), mu1=float(th1.mean()),
                sigma_ctrl0=float(th0.std()), sigma_ctrl1=float(th1.std()), clip=clip,
                method=method, n_traj=n_traj, omega=omega)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    omegas = [12, 22, 45, 90, 200, 800] if quick else OMEGAS
    n_exact, n_tau = (12, 200) if quick else (40, 1000)
    t_win = 120.0 if quick else 300.0
    res = {}
    t0 = time.time()
    for label, scheme, alpha, kk in CONFIGS:
        M.set_rates(K=kk)
        print(f"\n{label} (alpha={alpha:.3f}, k={kk:.3f})")
        print(f"{'Omega':>6}{'method':>8}{'<g>':>8}{'Var(dy)Om':>10}{'LNA+SN':>9}{'BEP':>11}{'95% CI':>24}{'s':>6}")
        for om in omegas:
            methods = ["exact"] if om <= EXACT_MAX else ["tauleap"]
            if om in XCHECK:
                methods = ["exact", "tauleap"]
            for meth in methods:
                ts = time.time()
                r = measure(scheme, alpha, "AIF", float(om), meth,
                            n_exact if meth == "exact" else n_tau, t_win=t_win)
                r["sec"] = time.time() - ts
                res[f"{label}|AIF|{om}|{meth}"] = r
                print(f"{om:>6}{meth:>8}{r['mean_g']:>8.4f}{r['var_dy_omega']:>10.3f}{r['var_dy_analytic']:>9.2f}"
                      f"{r['bep']:>11.3e}{ci(r):>24}{r['sec']:>6.0f}", flush=True)
        if label not in ("cooperative_a1", "coregulated_slow"):
            r = measure(scheme, alpha, "CAIF", 20.0, "exact", n_exact, t_win=t_win)
            res[f"{label}|CAIF|20|exact"] = r
            print(f"{20:>6}{'exact':>8}{r['mean_g']:>8.4f}{r['var_dy_omega']:>10.3f}{'':>9}{r['bep']:>11.3e}"
                  f"{ci(r):>24}   (CAIF)")
        if 20 not in omegas:
            r = measure(scheme, alpha, "AIF", 20.0, "exact", n_exact, t_win=t_win)
            res[f"{label}|AIF|20|exact"] = r
            print(f"{20:>6}{'exact':>8}{r['mean_g']:>8.4f}{r['var_dy_omega']:>10.3f}{'':>9}{r['bep']:>11.3e}"
                  f"{ci(r):>24}   (AIF, ranking point)", flush=True)
        M.set_rates()
    json.dump(dict(results=res, omegas=omegas, design_points=DP, t_sym=M.T_SYM),
              open(os.path.join(OUT, "bep_lc.json"), "w"), indent=1, default=float)
    print(f"\nwrote results/bep_lc.json  ({time.time() - t0:.0f} s)")
