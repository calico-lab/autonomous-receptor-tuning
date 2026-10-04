"""
run_design_rule.py -- test of the loop-speed design rule (Section IV-B, Fig. 5).

The noise at the detector input decreases with the tracking time constant tau_LF = gamma_p / b, whereas the tracking
error on a periodic channel grows with it (Section III-C).  model.surrogate_bep_varying() evaluates the Gaussian
approximation of the BEP on the periodic channel for a given actuator gain alpha, and model.design_rule_tau() gives
the leading-order optimum.  This script sweeps alpha for the non-cooperative receiver at S = 5 with T_c = 240 and
960, and at S = 20 with T_c = 240, simulates the full nonlinear receiver with random symbol streams at every alpha,
and records the measured BEP together with the approximation and the rule.

Output: results/design_rule.json
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from run_headline import one, OUT

ALPHAS = [0.15, 0.2, 0.25, 0.3, 0.384, 0.5, 0.65, 0.85, 1.1, 1.5]
CASES = [(5.0, 240.0), (5.0, 960.0), (20.0, 240.0)]


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    alphas = [0.2, 0.3, 0.384, 0.65, 1.1] if quick else ALPHAS
    cases = [(5.0, 240.0)] if quick else CASES
    kw = dict(n_traj=16, n_periods=1) if quick else {}
    res = dict(alphas=alphas, cases=[], surrogate=[], rule=[])
    t0 = time.time()
    for S, T in cases:
        tau_rule, A, B = M.design_rule_tau("allosteric", S, T)
        res["rule"].append(dict(S=S, T=T, tau_opt=tau_rule, A=A, B=B))
        print(f"\nS={S:g}, T_c={T:g}: rule tau_opt = {tau_rule:.2f} (A={A:.4f}, B={B:.2e})")
        print(f"{'alpha':>6}{'tau':>7}{'mismatch':>10}{'surrogate':>11}{'adaptive':>11}{'95% CI':>24}{'static':>10}{'gain':>8}")
        rows = []
        for a in alphas:
            sur = M.surrogate_bep_varying("allosteric", a, S, T)
            r = one("allosteric", a, 1.0, S, T, seed=5, **kw)
            rows.append(dict(alpha=a, tau=sur["tau"], mismatch=sur["mismatch"], surrogate=sur["bep"],
                             adaptive=r["adaptive"], lo=r["lo"], hi=r["hi"], static_best=r["static_best"],
                             gain=r["gain"], track_err_mean_ln=r["track_err_mean_ln"]))
            print(f"{a:>6.3f}{sur['tau']:>7.2f}{sur['mismatch']:>10.3f}{sur['bep']:>11.2e}{r['adaptive']:>11.2e}"
                  f"  [{r['lo']:.1e},{r['hi']:.1e}]{r['static_best']:>10.2e}{r['gain']:>8.3g}", flush=True)
        # optimum of the Gaussian approximation on a fine alpha grid
        grid = np.geomspace(0.1, 3.0, 120)
        sg = [M.surrogate_bep_varying("allosteric", a, S, T) for a in grid]
        i = int(np.argmin([g["bep"] for g in sg]))
        j = int(np.argmin([r["adaptive"] for r in rows]))
        res["cases"].append(dict(S=S, T=T, rows=rows, surrogate_alpha_opt=float(grid[i]), surrogate_tau_opt=sg[i]["tau"],
                                 surrogate_bep_min=sg[i]["bep"], measured_alpha_opt=rows[j]["alpha"],
                                 measured_tau_opt=rows[j]["tau"], measured_bep_min=rows[j]["adaptive"],
                                 rule_tau_opt=tau_rule))
        res["surrogate"].append(dict(S=S, T=T, alpha=[float(a) for a in grid], tau=[g["tau"] for g in sg],
                                     bep=[g["bep"] for g in sg]))
        print(f"  surrogate optimum: alpha={grid[i]:.3f}, tau={sg[i]['tau']:.2f};  measured optimum: alpha={rows[j]['alpha']:.3f}, "
              f"tau={rows[j]['tau']:.2f};  rule tau={tau_rule:.2f}")
    json.dump(res, open(os.path.join(OUT, "design_rule.json"), "w"), indent=1, default=float)
    print(f"\nwrote results/design_rule.json  ({time.time() - t0:.0f} s)")
