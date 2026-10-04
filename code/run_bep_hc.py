"""
run_bep_hc.py -- error probabilities on the static channel at a high copy number (Section IV-C1, Table II, Fig. 6(a)).

Every receiver of Table II is simulated at its design point with the full nonlinear receptor map at Omega = 200,
driven by a random stream of independent and equiprobable bits (T_sym = 0.5), which is independent for every
trajectory.  The BEP is the exact binomial error probability of the fixed-threshold detector, averaged over the
recorded modulator ensemble, with the exact two-population statistics for the mixed array.  The script also runs the
averaged-input references, in which the controller senses the symbol-averaged occupancy directly (no self-noise),
evaluates the non-adaptive receptor with its affinity fixed at the optimum, and records the measured occupancy
moments and modulator variances.

Output: results/bep_hc.json
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from engine import tauleap
from detection import (bep_from_record, occupancy_record, bootstrap_ci, bep_exact,
                       oracle_bep, hill)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def ci(r):
    return "[%.2e,%.2e]" % (r["lo"], r["hi"])


DP = json.load(open(os.path.join(OUT, "design_points.json")))["design"]

CONFIGS = [  # (label, scheme, alpha, k)
    ("allosteric", "allosteric", DP["allosteric"]["alpha"], M.K),
    ("cooperative", "cooperative", DP["cooperative"]["alpha"], M.K),
    ("coregulated", "coregulated", DP["coregulated"]["alpha"], M.K),
    ("coregulated_slow", "coregulated", 1.0, M.K_COREG_SLOW),
    ("weighted", "weighted", DP["weighted"]["alpha"], M.K),
    ("cooperative_a1", "cooperative", 1.0, M.K),
]


def measure(scheme, alpha, motif, omega=M.OMEGA_HC, n_traj=2000, t_burn=200.0,
            t_win=400.0, dt=0.01, dt_rec=1.0, seed=7, n_boot=4000, drive="stream"):
    t, y, c0, c1, clip = tauleap(scheme, alpha, M.CL0_NOM, M.CL1_NOM, motif=motif,
                                 omega=omega, n_traj=n_traj, t_end=t_burn + t_win,
                                 dt=dt, t_burn=t_burn, dt_rec=dt_rec, drive=drive,
                                 seed=M.stable_seed("hc", scheme, alpha, motif, drive, base=seed))
    th0, th1 = occupancy_record(scheme, alpha, y, c0, c1)
    g = 0.5 * (th0 + th1)
    bep, per = bep_from_record(scheme, alpha, y, c0, c1)
    lo, hi = bootstrap_ci(per, n_boot=n_boot, seed=seed)
    ys = M.y_equilibrium(scheme, alpha)
    m0, m1 = float(th0.mean()), float(th1.mean())
    # standard deviation at the detector input per level: modulator-induced (incl. self-noise) + binomial sampling term
    s0 = float(np.sqrt(th0.var() + m0 * (1 - m0) / M.N_RECEPTORS))
    s1 = float(np.sqrt(th1.var() + m1 * (1 - m1) / M.N_RECEPTORS))
    bep_gauss = float(0.5 * M.Q((0.5 - m0) / s0) + 0.5 * M.Q((m1 - 0.5) / s1))
    kd = M.kd_of_y(y, alpha)
    # convergence diagnostics: BEP over the first n trajectories and the largest single-trajectory share
    conv = {str(n): float(per[:n].mean()) for n in (125, 250, 500, 1000, 2000) if n <= per.size}
    share = float(per.max() / per.sum())
    return dict(bep=bep, lo=lo, hi=hi, bep_gauss=bep_gauss, convergence=conv, max_traj_share=share,
                per_traj=[float(v) for v in per],
                mean_g=float(g.mean()), mean_y=float(y.mean()), y_star=ys,
                mean_kd=float(kd.mean()), var_dy_omega=float(y.var() * omega),
                mu0=m0, mu1=m1, dmu=m1 - m0, sigma0=s0, sigma1=s1,
                sigma_ctrl0=float(np.sqrt(th0.var())), sigma_ctrl1=float(np.sqrt(th1.var())),
                clip=clip, n_traj=n_traj, t_win=t_win, dt=dt, omega=omega, drive=drive)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    kw = dict(n_traj=200, t_win=150.0) if quick else {}
    res = {}
    print(f"{'config':22s}{'zeta*':>7}{'<g>':>7}{'Var(dy)Om':>10}{'LNA+SN':>9}{'LNA':>9}"
          f"{'sig0':>7}{'sig1':>7}{'dth':>6}{'BEP exact':>11}{'95% CI':>24}{'BEP gauss':>11}")
    for label, scheme, alpha, kk in CONFIGS:
        for motif in M.MOTIFS:
            t0 = time.time()
            M.set_rates(K=kk)
            r = measure(scheme, alpha, motif, **kw)
            sur = M.bep_surrogate(scheme, alpha)
            M.set_rates()
            r.update(scheme=scheme, alpha=alpha, motif=motif, k=kk, zeta=sur["zeta"], rho=sur["rho"],
                     var_dy_analytic=sur["var_dy"], var_dy_analytic_chem=sur["var_dy_chem"],
                     var_dy_reduced=sur["var_dy_reduced"], var_dy_reduced_chem=sur["var_dy_reduced_chem"],
                     d_bits=sur["d_bits"], wc=sur["wc"], tau=sur["tau"], tau_full=sur["tau_full"],
                     zeta_crit=sur["zeta_crit"], margin=sur["zeta_crit"] / sur["zeta"], n_h=sur["n_h"],
                     inventory=sur["inventory"], sig0_pred=sur["sig0"], sig1_pred=sur["sig1"],
                     bep_surr=sur["bep"], sec=time.time() - t0)
            res[f"{label}|{motif}"] = r
            print(f"{label + ' ' + motif:22s}{r['zeta']:>7.3f}{r['mean_g']:>7.4f}"
                  f"{r['var_dy_omega']:>10.3f}{r['var_dy_analytic']:>9.2f}{r['var_dy_analytic_chem']:>9.2f}"
                  f"{r['sigma0']:>7.4f}{r['sigma1']:>7.4f}{r['dmu']:>6.3f}{r['bep']:>11.2e}"
                  f"{ci(r):>24}{r['bep_gauss']:>11.2e}")

    # averaged-input references: the controller senses the symbol-averaged occupancy (no self-noise)
    bitavg = {}
    for label, scheme, alpha, kk in CONFIGS[:5]:
        t0 = time.time()
        M.set_rates(K=kk)
        r = measure(scheme, alpha, "AIF", drive="bitavg", **kw)
        sur = M.bep_surrogate(scheme, alpha, with_bits=False)
        M.set_rates()
        r.update(scheme=scheme, alpha=alpha, motif="AIF", zeta=sur["zeta"], var_dy_analytic=sur["var_dy"],
                 var_dy_reduced=sur["var_dy_reduced"], sec=time.time() - t0)
        bitavg[f"{label}|AIF"] = r
        print(f"{label + ' AIF (bitavg ref)':22s}{r['zeta']:>7.3f}{r['mean_g']:>7.4f}"
              f"{r['var_dy_omega']:>10.3f}{r['var_dy_analytic']:>9.2f}{'':>9}"
              f"{r['sigma0']:>7.4f}{r['sigma1']:>7.4f}{r['dmu']:>6.3f}{r['bep']:>11.2e}{ci(r):>24}")

    # non-adaptive reference: receptor with its affinity fixed at the optimum
    ref = {}
    for n in (1.0, M.N_COOP):
        b = float(oracle_bep(M.CL0_NOM, M.CL1_NOM, n))
        th0 = float(hill(M.CL0_NOM, M.KD_OPT, n)); th1 = float(hill(M.CL1_NOM, M.KD_OPT, n))
        ref[f"static_at_kdopt_n{n:g}"] = dict(bep=b, mu0=th0, mu1=th1, dmu=th1 - th0,
                                             sigma=float(np.sqrt(th0 * (1 - th0) / M.N_RECEPTORS)))
        print(f"{'fixed K_D=K_D,opt n=' + f'{n:g}':22s}{'--':>7}{'--':>7}{'--':>10}{'--':>9}{'--':>9}"
              f"{ref[f'static_at_kdopt_n{n:g}']['sigma']:>7.4f}{'':>7}{th1 - th0:>6.3f}{b:>11.2e}")
    json.dump(dict(configs=res, bitavg_reference=bitavg, references=ref, design_points=DP,
                   t_sym=M.T_SYM),
              open(os.path.join(OUT, "bep_hc.json"), "w"), indent=1, default=float)
    print("\nwrote results/bep_hc.json")
