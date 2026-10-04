"""
run_design_points.py -- design point of every receiver (Table II).

For each receiver, the actuator gain alpha minimizes the Gaussian approximation of the BEP on the static channel,
subject to the bandwidth requirement omega_c >= 2 pi / T_c at the design coherence time T_c = 80 (Section III-C).
The approximation uses the full linear-noise variance of the loop, including the self-noise of the random symbol
stream.  The co-regulated receiver is fixed at alpha = 1 by its equilibrium cooperativity, and the matched
co-regulated design reduces the actuation rate to restore the loop gain of the other designs (Section III-E).  No
stability constraint is imposed, and the exact boundary zeta_crit is reported as a margin (Section III-D).

Output: results/design_points.json
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import model as M

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

HDR = (f"{'scheme':12s}{'alpha':>7}{'y*':>7}{'n_H':>5}{'zeta*':>7}{'rho':>6}{'w_c':>7}{'tau':>6}{'tau_f':>6}"
       f"{'V_red':>7}{'V_full':>7}{'V+bits':>7}{'D_bits':>8}{'z_crit':>7}{'margin':>7}"
       f"{'dth':>6}{'sig0':>7}{'sig1':>7}{'BEP_surr':>10}{'invent':>7}{'active':>9}")


def row(s, r):
    return (f"{s:12s}{r['alpha']:>7.3f}{r['ys']:>7.3f}{r['n_h']:>5.2f}{r['zeta']:>7.3f}{r['rho']:>6.3f}"
            f"{r['wc']:>7.4f}{r['tau']:>6.2f}{r['tau_full']:>6.2f}{r['var_dy_reduced_chem']:>7.2f}"
            f"{r['var_dy_chem']:>7.2f}{r['var_dy']:>7.2f}{r['d_bits']:>8.4f}{r['zeta_crit']:>7.2f}"
            f"{r['margin']:>7.1f}{r['dmu']:>6.3f}{r['sig0']:>7.4f}{r['sig1']:>7.4f}{r['bep']:>10.2e}"
            f"{r['inventory']:>7.0f}")


if __name__ == "__main__":
    print(f"Coherence-time constraint: omega_c >= 2 pi / {M.T_C:.0f} = {M.WC_MIN:.4f};  T_sym = {M.T_SYM}\n")
    print(HDR)
    dp = {}
    for s in M.SCHEMES:
        r = M.design_point(s)
        dp[s] = r
        act = "w_c" if r["wc_bound_active"] else ("nH-cap" if r["alpha_lower_active"] else "interior")
        print(row(s, r) + f"{act:>9}")
    # unconstrained optimum of the co-regulated receiver (self-noise included), stored for reference
    def obj_c(a):
        rr = M.bep_surrogate("coregulated", a)
        return 1.0 if rr["zeta"] <= 0 or rr["wc"] < M.WC_MIN else rr["bep"]
    grid_c = np.linspace(M.alpha_lower_bound("coregulated"), M.ALPHA_RANGE[1], 600)
    a_c = float(grid_c[int(np.argmin([obj_c(x) for x in grid_c]))])
    rc = M.bep_surrogate("coregulated", a_c); rc.update(alpha=a_c, margin=rc["zeta_crit"] / rc["zeta"])
    print(row("coreg (uncon.)", rc) + f"{'record':>9}")
    # bandwidth-matched co-regulated design: alpha = 1 (n_H(y*) = 2.5) with the actuation rate reduced
    # so that the loop gain b = k rho theta zeta* equals that of the non-cooperative design (tau = 10.1)
    M.set_rates(K=M.K_COREG_SLOW)
    rs = M.bep_surrogate("coregulated", 1.0)
    rs.update(alpha=1.0, margin=rs["zeta_crit"] / rs["zeta"], sf_reference=M.sf_reference(rs["zeta"], rs["rho"]),
              zeta_half_occupancy=M.zeta_half_occupancy("coregulated", 1.0), wc_bound_active=True, alpha_lower_active=True,
              k=M.K_COREG_SLOW)
    M.set_rates()
    dp["coregulated_slow"] = rs
    print(row("coreg (slow k)", rs) + f"{'k-matched':>9}")
    # cooperative receptor at the actuator gain of the co-regulated receiver (control comparison, Section IV-C1)
    r = M.bep_surrogate("cooperative", 1.0)
    r.update(alpha=1.0, margin=r["zeta_crit"] / r["zeta"], sf_reference=M.sf_reference(r["zeta"], r["rho"]),
             zeta_half_occupancy=M.zeta_half_occupancy("cooperative", 1.0), wc_bound_active=False, alpha_lower_active=False)
    dp["cooperative_a1"] = r
    print(row("cooperative_a1", r) + f"{'control':>9}")
    for key in dp:
        dp[key].setdefault("k", M.K)

    # design points without the self-noise term (they coincide with the design points above)
    print("\nDesign points without the self-noise term (with_bits = False), for reference:")
    print(HDR)
    chem = {}
    for s in M.SCHEMES:
        M_design = M.design_point
        # temporarily evaluate the Gaussian approximation without the self-noise term
        def obj(a, s=s):
            rr = M.bep_surrogate(s, a, with_bits=False)
            return 1.0 if rr["zeta"] <= 0 or rr["wc"] < M.WC_MIN else rr["bep"]
        grid = np.linspace(M.alpha_lower_bound(s), M.ALPHA_RANGE[1], 600)
        a = float(grid[int(np.argmin([obj(x) for x in grid]))])
        rr = M.bep_surrogate(s, a, with_bits=False); rr.update(alpha=a, margin=rr["zeta_crit"] / rr["zeta"])
        chem[s] = rr
        print(row(s, rr))

    print("\nMinimum stability margin zeta_crit/zeta* over alpha in [0.1, 12]:")
    minmargin = {}
    for s in M.SCHEMES:
        best = (np.inf, None)
        for a in np.geomspace(0.1, 12.0, 400):
            if s == "coregulated" and a < M.alpha_lower_bound(s):
                continue
            ss = M.steady_state(s, a)
            m = M.zeta_crit_closed_form(ss) / M.zeta_star(s, a)
            if m < best[0]:
                best = (m, a)
        minmargin[s] = dict(margin=float(best[0]), alpha=float(best[1]))
        print(f"  {s:12s} min margin = {best[0]:6.2f} at alpha = {best[1]:.3f}")

    json.dump(dict(T_c=M.T_C, wc_min=M.WC_MIN, t_sym=M.T_SYM, design=dp, design_chem_only=chem, coregulated_unconstrained=rc,
                   min_margin=minmargin),
              open(os.path.join(OUT, "design_points.json"), "w"), indent=1, default=float)
    print("\nwrote results/design_points.json")
