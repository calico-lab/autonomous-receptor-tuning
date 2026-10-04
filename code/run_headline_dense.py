"""
run_headline_dense.py -- more trajectories for the cooperative receiver in Fig. 3(a).

Re-runs the T_c = 240 rows of the cooperative receiver with 192 trajectories instead of 48, since its adaptive BEP is
small enough that 48 trajectories leave visible sampling scatter.  The rows are replaced in results/headline.json and
results/headline.csv.  Run after run_headline.py.
"""
from __future__ import annotations

import json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from run_headline import one, OUT, DP, ci

if __name__ == "__main__":
    path = os.path.join(OUT, "headline.json")
    data = json.load(open(path))
    rows = data["rows"]
    label, scheme, alpha, n_h = "cooperative", "cooperative", DP["cooperative"]["alpha"], M.N_COOP
    T = 240.0
    t0 = time.time()
    for S in data["swings"]:
        r = one(scheme, alpha, n_h, S, T, n_traj=192)
        r.update(label=label, T_over_tau=T / DP[label]["tau"])
        idx = next(i for i, x in enumerate(rows) if x["label"] == label and x["T"] == T and x["S"] == S)
        rows[idx] = r
        print(f"S={S:6.1f}: adaptive {r['adaptive']:.3e} {ci(r)}  static {r['static_best']:.2e}  gain {r['gain']:.3g}", flush=True)
        json.dump(data, open(path, "w"), indent=1, default=float)
    keys = ["label", "scheme", "alpha", "n_h", "T", "T_over_tau", "S", "adaptive", "lo", "hi",
            "static_best", "static_kd", "static_thr", "oracle", "gain", "track_err_ln",
            "track_err_mean_ln", "clip"]
    with open(os.path.join(OUT, "headline.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) if isinstance(r[k], str) else f"{r[k]:.6g}" for k in keys) + "\n")
    print(f"\nupdated results/headline.json, headline.csv  ({time.time() - t0:.0f} s)")
