"""
run_headline.py -- adaptive and non-adaptive receivers on the same time-varying channel (Section IV-B, Fig. 3).

Both received levels are attenuated by a common factor,
    c0(t) = c0 g(t),  c1(t) = c1 g(t),  g(t) = exp(-1/2 ln S (1 - cos 2 pi t / T_c)),
which varies from one to 1/S and back with period T_c, such that the optimal affinity sqrt(c0 c1) g(t) moves.  The
bits are independent and equiprobable (T_sym = 0.5), with a separate symbol stream for every trajectory, and the
controller senses the instantaneous occupancy.  All receivers are scored by the exact binomial detector on the same
signal:
    adaptive:           AIF loop at the design point (non-cooperative, cooperative, co-regulated), fixed threshold,
                        no channel knowledge;
    best non-adaptive:  fixed K_D and threshold optimized over the whole channel ensemble, with the same Hill
                        coefficient as the adaptive receiver ("static_best" in the results);
    genie-aided:        K_D = sqrt(c0 c1) g(t) at every instant, without noise ("oracle" in the results).
The script sweeps the variation depth S and the coherence time T_c.

Outputs: results/headline.json and results/headline.csv
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from engine import tauleap
from detection import (bep_from_record, bootstrap_ci, best_static_receiver,
                       oracle_bep)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def ci(r):
    return "[%.2e,%.2e]" % (r["lo"], r["hi"])

DP = json.load(open(os.path.join(OUT, "design_points.json")))["design"]

RECEIVERS = [("allosteric", "allosteric", DP["allosteric"]["alpha"], 1.0, M.K),
             ("cooperative", "cooperative", DP["cooperative"]["alpha"], M.N_COOP, M.K),
             ("coregulated", "coregulated", DP["coregulated"]["alpha"], M.N_COOP, M.K),
             ("coregulated_slow", "coregulated", 1.0, M.N_COOP, M.K_COREG_SLOW)]
SWINGS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
PERIODS = [60.0, 120.0, 240.0, 480.0, 960.0]


def gain_fn(S, T):
    lnS = np.log(S)
    return lambda t: float(np.exp(-0.5 * lnS * (1.0 - np.cos(2 * np.pi * t / T))))


def one(scheme, alpha, n_h, S, T, n_traj=48, n_periods=2, t_burn=200.0, dt=0.01,
        dt_rec=1.0, seed=3, drive="stream"):
    g = gain_fn(S, T)
    c0 = lambda t: M.CL0_NOM * g(t)
    c1 = lambda t: M.CL1_NOM * g(t)
    t_end = t_burn + n_periods * T
    t, y, c0r, c1r, clip = tauleap(scheme, alpha, c0, c1, omega=M.OMEGA_HC, n_traj=n_traj,
                                   t_end=t_end, dt=dt, t_burn=t_burn, dt_rec=dt_rec, drive=drive,
                                   seed=M.stable_seed("hl", scheme, S, T, drive, base=seed))
    bep, per = bep_from_record(scheme, alpha, y, c0r, c1r)
    lo, hi = bootstrap_ci(per, seed=seed)
    st = best_static_receiver(c0r, c1r, n_hill=n_h)
    orc = float(oracle_bep(c0r, c1r, n_h).mean())
    kd = M.kd_of_y(y, alpha)
    kd_opt = np.sqrt(c0r * c1r)[:, None]
    track_err = float(np.mean(np.abs(np.log(kd / kd_opt))))
    track_err_mean = float(np.mean(np.abs(np.log(kd.mean(axis=1) / kd_opt[:, 0]))))
    return dict(scheme=scheme, alpha=alpha, n_h=n_h, k=M.K, S=S, T=T, adaptive=bep, lo=lo, hi=hi,
                static_best=st["bep"], static_kd=st["kd"], static_thr=st["thr"],
                oracle=orc, gain=st["bep"] / bep, track_err_ln=track_err,
                track_err_mean_ln=track_err_mean, clip=clip, n_traj=n_traj, n_periods=n_periods,
                drive=drive)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    swings = [1.0, 2.0, 5.0, 20.0] if quick else SWINGS
    periods = [60.0, 240.0, 960.0] if quick else PERIODS
    kw = dict(n_traj=16, n_periods=1) if quick else {}
    rows = []
    t0 = time.time()
    for label, scheme, alpha, n_h, kk in RECEIVERS:
        M.set_rates(K=kk)
        tau_s = DP[label]["tau"]
        print(f"\n{label}: alpha={alpha:.3f}, n_H={n_h}, k={kk:.3f}, tau={tau_s:.1f}")
        print(f"{'T':>6}{'T/tau':>7}{'S':>7}{'adaptive':>11}{'95% CI':>24}{'static':>10}{'oracle':>10}{'gain':>9}{'trk':>6}")
        for T in periods:
            for S in swings:
                r = one(scheme, alpha, n_h, S, T, **kw)
                r.update(label=label, T_over_tau=T / tau_s)
                rows.append(r)
                print(f"{T:>6.0f}{T / tau_s:>7.1f}{S:>7.1f}{r['adaptive']:>11.3e}"
                      f"{ci(r):>24}{r['static_best']:>10.2e}"
                      f"{r['oracle']:>10.2e}{r['gain']:>9.3g}{r['track_err_mean_ln']:>6.2f}"
                      + ("  [clip]" if r['clip'] > 1e-3 else ""), flush=True)
        M.set_rates()
    keys = ["label", "scheme", "alpha", "n_h", "T", "T_over_tau", "S", "adaptive", "lo", "hi",
            "static_best", "static_kd", "static_thr", "oracle", "gain", "track_err_ln",
            "track_err_mean_ln", "clip"]
    with open(os.path.join(OUT, "headline.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) if isinstance(r[k], str) else f"{r[k]:.6g}" for k in keys) + "\n")
    json.dump(dict(rows=rows, swings=swings, periods=periods,
                   tau_settle={k: DP[k]["tau"] for k in DP}, tau={k: DP[k]["tau"] for k in DP},
                   t_sym=M.T_SYM),
              open(os.path.join(OUT, "headline.json"), "w"), indent=1, default=float)
    print(f"\nwrote results/headline.csv, headline.json  ({time.time() - t0:.0f} s)")
