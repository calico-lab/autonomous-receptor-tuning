"""
run_bep_lc_dense.py -- additional tau-leaping runs for Fig. 6(b).

Adds tau-leaping runs with 1000 trajectories at Omega = 65, 90, and 130 for every receiver to results/bep_lc.json.
Together with the tau-leaping runs at Omega = 45, 200, 400, and 800, this gives a uniform series for Omega >= 45,
whereas the exact simulation is used for Omega <= 30 and as the cross-check at 45 and 200.  Run after run_bep_lc.py.
"""
from __future__ import annotations

import json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from run_bep_lc import measure, CONFIGS, OUT, ci

if __name__ == "__main__":
    path = os.path.join(OUT, "bep_lc.json")
    data = json.load(open(path))
    res = data["results"]
    t0 = time.time()
    import model as M
    for label, scheme, alpha, kk in CONFIGS:
        M.set_rates(K=kk)
        for om in (65, 90, 130):
            key = f"{label}|AIF|{om}|tauleap"
            if key in res:
                continue
            ts = time.time()
            r = measure(scheme, alpha, "AIF", float(om), "tauleap", 1000, t_win=300.0)
            r["sec"] = time.time() - ts
            res[key] = r
            print(f"{label:15s} Omega={om:4d} tauleap  <g>={r['mean_g']:.4f}  Var(dy)Om={r['var_dy_omega']:.2f}  "
                  f"BEP={r['bep']:.3e} {ci(r)}  ({r['sec']:.0f} s)", flush=True)
            json.dump(data, open(path, "w"), indent=1, default=float)
        M.set_rates()
    print(f"\nupdated results/bep_lc.json  ({time.time() - t0:.0f} s)")
