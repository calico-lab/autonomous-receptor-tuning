"""
run_leaky.py -- equilibrium offset under leaky integration (Section III-F).

With controller dilution or degradation at the rate gamma_c > 0, the integrator is leaky, and the sensed occupancy
settles at
    g* = mu/theta + gamma_c (z2* - z1*) / theta
(for the cascaded variant CAIF, z1* is replaced by z1* + z3*), such that K_D(y*) leaves the geometric mean of the
received levels.  For every receiver at its design point and a sweep of gamma_c, this script reports the offset of the
sensed occupancy, the detuning of K_D, the increase in the noiseless error probability, and the Gaussian
approximation of the BEP including the controller noise and the self-noise.  It also evaluates the small-leak
formula delta ln K_D = alpha gamma_c (z1* - z2*) / (theta zeta*) and the leak tolerances for a ten-percent and a
twofold increase in the error probability.

Output: results/leaky.json
"""
from __future__ import annotations

import json, os, sys
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from detection import bep_exact

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
DP = json.load(open(os.path.join(OUT, "design_points.json")))["design"]
GCS = [0.0, 0.0005, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5]


def row(scheme, alpha, motif, gc):
    ss = M.steady_state(scheme, alpha, motif, gc)
    ys, g = ss["y"], ss["g"]
    kd = float(M.kd_of_y(ys, alpha))
    th0 = float(M.theta_B(scheme, M.CL0_NOM, ys, alpha))
    th1 = float(M.theta_B(scheme, M.CL1_NOM, ys, alpha))
    formula = M.MU / M.THETA + gc * (ss["z2"] - ss["z1"] - ss["z3"]) / M.THETA
    bep_det = float(bep_exact(th0, th1))
    s0, s1 = M.level_slopes(scheme, alpha, ys)
    z = 0.5 * (s0 + s1)
    rho = M.rho_of(ss)
    db = M.THETA ** 2 * (th1 - th0) ** 2 * M.T_SYM / 4.0
    vdy = (M.var_dy_full(z, ss, gc, M.OMEGA_HC, db) if motif == "AIF"
           else M.var_dy(z, rho, ys, gc, M.OMEGA_HC, db)) / M.OMEGA_HC
    sig0 = np.sqrt(s0 ** 2 * vdy + M.var_bind(th0))
    sig1 = np.sqrt(s1 ** 2 * vdy + M.var_bind(th1))
    bep_g = float(0.5 * M.Q((0.5 - th0) / sig0) + 0.5 * M.Q((th1 - 0.5) / sig1))
    # small-leak formula from the zero-leak state
    ss0 = M.steady_state(scheme, alpha, motif, 0.0)
    z0 = M.zeta_star(scheme, alpha)
    dlnk = float(M.beta_local(ss0["y"], alpha)) * gc * (ss0["z1"] - ss0["z2"] - ss0["z3"]) / (M.THETA * z0)
    return dict(scheme=scheme, motif=motif, gc=gc, gc_over_gp=gc / M.GP, alpha=alpha, g_star=float(g),
                g_formula=float(formula), y_star=float(ys), kd=kd, kd_ratio=kd / M.KD_OPT,
                kd_ratio_smallleak=float(np.exp(dlnk)),
                mu0=th0, mu1=th1, bep_det=bep_det, bep_gauss=bep_g,
                z1=ss["z1"], z2=ss["z2"], z3=ss["z3"], zeta=z,
                zeta_crit=M.zeta_crit_closed_form(ss, gc) if motif == "AIF" else float("nan"))


def tolerance(scheme, alpha, motif, penalty, gcs=(1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1), key="bep_det"):
    """gamma_c/gamma_p at which the BEP penalty (deterministic, or the Gaussian approximation with
    the reaction noise and the self-noise for key = 'bep_gauss') reaches `penalty` (log-log
    interpolation on a fine leak grid; steady states that fail to converge are skipped)."""
    b0 = row(scheme, alpha, motif, 0.0)[key]
    xs, ys = [], []
    for gc in gcs:
        try:
            ys.append(row(scheme, alpha, motif, gc)[key] / b0); xs.append(gc)
        except RuntimeError:
            continue
    xs, ys = np.array(xs), np.array(ys)
    if len(ys) < 2 or not np.any(ys >= penalty):
        return float("nan")
    i = int(np.argmax(ys >= penalty))
    if i == 0:
        return float("nan")
    lx = np.interp(np.log(penalty), np.log(ys[i - 1:i + 1]), np.log(xs[i - 1:i + 1]))
    return float(np.exp(lx) / M.GP)


if __name__ == "__main__":
    rows = []
    for scheme in M.SCHEMES:
        alpha = DP[scheme]["alpha"]
        print(f"\n{scheme} (alpha = {alpha:.3f}); ideal setpoint g* = {M.MU / M.THETA}")
        print(f"{'gc/gp':>7}{'motif':>6}{'g*':>8}{'formula':>9}{'K_D/K_opt':>10}{'small-leak':>11}{'mu0':>7}{'mu1':>7}"
              f"{'BEP det':>10}{'BEP gauss':>11}")
        for gc in GCS:
            for motif in M.MOTIFS:
                r = row(scheme, alpha, motif, gc)
                rows.append(r)
                print(f"{gc / M.GP:>7.3f}{motif:>6}{r['g_star']:>8.4f}{r['g_formula']:>9.4f}{r['kd_ratio']:>10.4f}"
                      f"{r['kd_ratio_smallleak']:>11.4f}{r['mu0']:>7.4f}{r['mu1']:>7.4f}{r['bep_det']:>10.2e}{r['bep_gauss']:>11.2e}")
    summary = {}
    for scheme in ("allosteric", "cooperative"):
        for motif in M.MOTIFS:
            r = [x for x in rows if x["scheme"] == scheme and x["motif"] == motif and abs(x["gc"] - 0.15) < 1e-9][0]
            summary[f"{scheme}|{motif}|0.15"] = dict(g_star=r["g_star"], kd_ratio=r["kd_ratio"],
                                                    bep_det=r["bep_det"], bep_gauss=r["bep_gauss"])
    tol = {}
    print("\nLeak tolerance gamma_c/gamma_p (deterministic BEP penalty 1.1 and 2):")
    for scheme in M.SCHEMES:
        alpha = DP[scheme]["alpha"]
        for motif in M.MOTIFS:
            t10, t2 = tolerance(scheme, alpha, motif, 1.1), tolerance(scheme, alpha, motif, 2.0)
            n10, n2 = tolerance(scheme, alpha, motif, 1.1, key="bep_gauss"), tolerance(scheme, alpha, motif, 2.0, key="bep_gauss")
            tol[f"{scheme}|{motif}"] = dict(ten_pct=t10, twofold=t2, ten_pct_noisy=n10, twofold_noisy=n2)
            print(f"  {scheme:12s} {motif}: deterministic 10 % at {t10:.2e}, twofold at {t2:.2e};  Gaussian approximation 10 % at {n10:.2e}, twofold at {n2:.2e}")
    json.dump(dict(rows=rows, summary=summary, tolerance=tol, gcs=GCS),
              open(os.path.join(OUT, "leaky.json"), "w"), indent=1, default=float)
    print("\nwrote results/leaky.json")
