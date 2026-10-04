"""
run_tracking.py -- symbol averaging, channel tracking, and interference (Section IV-A, Fig. 2).

All runs use the non-cooperative receiver at its design point, Omega = 200, and random symbol streams that are
independent for every trajectory, with the controller sensing the instantaneous occupancy.
A  Symbol averaging: random bits with T_sym = 0.5, much shorter than the loop time constant, starting from an untuned
   state (y = 0), and alternating bits with T_sym = 40, longer than the loop time constant (Fig. 2(a), (b)).
B  Step tracking: attenuation steps every T_c = 80 (Fig. 2(c)), with the re-convergence time to within 10 % of the new
   optimum of the smoothed ensemble mean.
C  Sinusoidal attenuation with S = 5 and period 240: tracking error and lag (Fig. 2(d)).
D  Interference: the free-diffusion tail at T_sym = 3 t_peak, convolved with random symbol streams and truncated to
   80 symbols.  Run with the K_base of Table I, for which the shifted optimum lies outside the actuator range and the
   actuator saturates, and with K_base = 20, for which it lies inside.  The equilibrium affinity is compared with the
   prediction based on the mean interference, sqrt((c0 + cbar)(c1 + cbar)), and with the exact root of the
   symbol-averaged occupancy.

Outputs: results/tracking.npz (traces of Fig. 2) and results/tracking.json (numbers)
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from engine import tauleap, make_levels, n_symbols

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
DP = json.load(open(os.path.join(OUT, "design_points.json")))["design"]
ALPHA = DP["allosteric"]["alpha"]
TAU = DP["allosteric"]["tau"]
SCHEME = "allosteric"
C0, C1 = M.CL0_NOM, M.CL1_NOM
T_SYM = M.T_SYM


def kd_stats(y):
    kd = M.kd_of_y(y, ALPHA)
    return kd.mean(axis=1), kd.std(axis=1)


def exp_A(n_traj, dt, quick):
    out = {}
    for tag, t_sym, t_end, kind in (("fast", T_SYM, 130.0, "iid"), ("slow", 40.0, 330.0, "alt")):
        rng = np.random.default_rng(M.stable_seed("A-levels", tag, base=5))
        levels, bits = make_levels(rng, n_traj, n_symbols(t_end, t_sym), kind=kind)
        t, y, _, _, _ = tauleap(SCHEME, ALPHA, C0, C1, omega=M.OMEGA_HC, n_traj=n_traj,
                                t_end=t_end, dt=dt, t_burn=0.0, dt_rec=0.05,
                                seed=M.stable_seed("A", tag, base=5), drive="stream",
                                levels=levels, t_sym=t_sym, y_init=0.0)
        kd_m, kd_s = kd_stats(y)
        j = np.minimum((t / t_sym).astype(int), levels.shape[1] - 1)
        c_rec = levels[:, j].T * C0                      # (n_rec, n_traj) actual levels
        th = M.theta_B(SCHEME, c_rec, y, ALPHA)           # instantaneous occupancy
        m = t > t_end - 30
        b_rec = bits[:, j].T
        th0f = float(th[m][b_rec[m] == 0].mean()); th1f = float(th[m][b_rec[m] == 1].mean())
        # occupancy at the end of a symbol: mean over the final eighth of each complete symbol among the last five
        # symbols, averaged separately over the symbols of each bit (where the loop has responded to the current bit)
        w = t_sym / 8
        e0, e1 = [], []
        for jj in range(int(np.ceil(t_end / t_sym))):
            t_start, t_stop = jj * t_sym, (jj + 1) * t_sym
            if t_stop > t[-1] + 1e-9 or t_start < t_end - 5 * t_sym:
                continue
            sel = (t >= t_stop - w - 1e-9) & (t < t_stop - 1e-9)
            bb, tt = b_rec[sel], th[sel]
            if (bb == 0).any():
                e0.append(float(tt[bb == 0].mean()))
            if (bb == 1).any():
                e1.append(float(tt[bb == 1].mean()))
        th0e, th1e = (float(np.mean(e0)) if e0 else float("nan")), (float(np.mean(e1)) if e1 else float("nan"))
        print(f"  A/{tag:4s}: T_sym={t_sym:5.1f} ({kind})  final <K_D>={kd_m[m].mean():.3f} (target 2.000)  "
              f"<theta_B|0>={th0f:.3f}  <theta_B|1>={th1f:.3f}  end of symbol {th0e:.3f}/{th1e:.3f}  "
              f"sd(K_D)/K_D={kd_s[m].mean() / kd_m[m].mean():.3f}")
        out[tag] = dict(t=t, kd_mean=kd_m, kd_std=kd_s, th_mean=th.mean(axis=1), th_std=th.std(axis=1),
                        th_traj0=th[:, 0], bits_traj0=b_rec[:, 0], kd_final=float(kd_m[m].mean()),
                        th0_final=th0f, th1_final=th1f, th0_end=th0e, th1_end=th1e,
                        kd_rel_sd=float(kd_s[m].mean() / kd_m[m].mean()))
    return out


def exp_B(n_traj, dt, quick, gains=(1.0, 0.5, 0.2, 0.6, 1.0), t_c=80.0):
    t_end = t_c * len(gains) + 40.0
    g_of_t = lambda tt: float(np.asarray(gains)[min(int(tt // t_c), len(gains) - 1)])
    t, y, c0r, _, _ = tauleap(SCHEME, ALPHA, lambda tt: C0 * g_of_t(tt), lambda tt: C1 * g_of_t(tt),
                              omega=M.OMEGA_HC, n_traj=n_traj, t_end=t_end, dt=dt, t_burn=0.0, dt_rec=0.1,
                              seed=M.stable_seed("B", base=5), drive="stream", init_levels=(C0, C1))
    kd_m, kd_s = kd_stats(y)
    g_t = np.asarray(gains)[np.minimum((t // t_c).astype(int), len(gains) - 1)]
    kd_opt = M.KD_OPT * g_t
    win = int(5 / 0.1)
    kd_sm = np.convolve(kd_m, np.ones(win) / win, mode="same")   # 5-unit moving average
    reconvergence = []
    for i in range(1, len(gains)):
        t0 = i * t_c
        target = M.KD_OPT * gains[i]
        m = (t >= t0)
        within = np.abs(kd_sm[m] / target - 1) < 0.10
        tt = t[m]
        hit = None
        for j in range(len(tt)):
            if within[j]:
                hit = tt[j] - t0
                break
        reconvergence.append(hit)
        print(f"  B: step {gains[i - 1]:.2f}->{gains[i]:.2f} at {t0:.0f}  re-converged in "
              f"{hit if hit is None else round(hit, 1)}")
    return dict(t=t, kd_mean=kd_m, kd_std=kd_s, kd_opt=kd_opt, gains=list(gains), t_c=t_c,
                reconvergence=[None if r is None else float(r) for r in reconvergence])


def exp_C(n_traj, dt, quick, S=5.0, T=240.0, n_periods=2):
    t_burn = 200.0
    t_end = t_burn + n_periods * T
    lnS = np.log(S)
    g = lambda tt: np.exp(-0.5 * lnS * (1 - np.cos(2 * np.pi * tt / T)))
    t, y, _, _, _ = tauleap(SCHEME, ALPHA, lambda tt: C0 * float(g(tt)), lambda tt: C1 * float(g(tt)),
                            omega=M.OMEGA_HC, n_traj=n_traj, t_end=t_end, dt=dt, t_burn=t_burn, dt_rec=0.5,
                            seed=M.stable_seed("C", base=5), drive="stream", init_levels=(C0, C1))
    kd_m, kd_s = kd_stats(y)
    kd_opt = M.KD_OPT * g(t)
    err_mean = float(np.mean(np.abs(np.log(kd_m / kd_opt))))            # error of the ensemble mean
    err_traj = float(np.mean(np.abs(np.log(M.kd_of_y(y, ALPHA) / kd_opt[:, None]))))  # incl. noise
    a, b = np.log(kd_m) - np.log(kd_m).mean(), np.log(kd_opt) - np.log(kd_opt).mean()
    lags = np.arange(-200, 201)
    xc = [np.sum(a[max(0, l):len(a) + min(0, l)] * b[max(0, -l):len(b) + min(0, -l)]) for l in lags]
    lag = lags[int(np.argmax(xc))] * 0.5
    print(f"  C: S={S}, T={T}  |ln(<K_D>/K_opt)| = {err_mean:.3f} ({100 * (np.exp(err_mean) - 1):.1f} %),  "
          f"per-trajectory {err_traj:.3f} ({100 * (np.exp(err_traj) - 1):.1f} %),  lag = {lag:.1f}")
    return dict(t=t, kd_mean=kd_m, kd_std=kd_s, kd_opt=kd_opt, S=S, T=T,
                mean_abs_log_err=err_mean, mean_abs_log_err_traj=err_traj, lag_s=float(lag))


def exp_D(n_traj, dt, quick, D=1.0, d=1.0, n_taps=80):
    """Equilibrium affinity under a memoryless and an ISI symbol stream, for two actuator ranges."""
    h = lambda tt: (4 * np.pi * D * tt) ** -1.5 * np.exp(-d ** 2 / (4 * D * tt))
    t_peak = d ** 2 / (6 * D)
    w = np.array([h(t_peak + m * T_SYM) / h(t_peak) for m in range(n_taps)])
    tail = float(w[1:].sum())
    cbar = tail * 0.5 * (C0 + C1)
    pred = float(np.sqrt((C0 + cbar) * (C1 + cbar)))
    t_burn, t_win = 300.0, 300.0
    t_end = t_burn + t_win
    res = dict(tail_sum=tail, cbar=cbar, pred=pred, t_sym=T_SYM, t_peak=t_peak, cases={})
    kbase0 = M.ACT["kbase"]
    for kbase in (kbase0, 20.0):
        M.set_actuator("exp", kbase=kbase)
        for tag in ("memoryless", "isi"):
            rng = np.random.default_rng(M.stable_seed("D-levels", tag, base=5))
            levels, _ = make_levels(rng, n_traj, n_symbols(t_end, T_SYM) + n_taps,
                                    isi_taps=(w if tag == "isi" else None))
            t, y, _, _, clip = tauleap(SCHEME, ALPHA, C0, C1, omega=M.OMEGA_HC, n_traj=n_traj,
                                       t_end=t_end, dt=dt, t_burn=t_burn, dt_rec=0.5,
                                       seed=M.stable_seed("D", tag, kbase, base=5), drive="stream",
                                       levels=levels, init_levels=(C0, C1))
            kd = M.kd_of_y(y, ALPHA)
            per = kd.mean(axis=0)
            mean, se = float(per.mean()), float(per.std(ddof=1) / np.sqrt(n_traj))
            lv = levels[:, int(t_burn / T_SYM):int(t_end / T_SYM)].ravel() * C0
            root = float(brentq(lambda Kx: np.mean(lv / (lv + Kx)) - 0.5, 0.05, 200.0))
            target = M.KD_OPT if tag == "memoryless" else pred
            res["cases"][f"{tag}|{kbase:.4g}"] = dict(
                kbase=kbase, tag=tag, kd_mean=mean, kd_se=se, ci95=1.96 * se, target=float(target),
                root=root, dev_pct=100 * (mean / target - 1), dev_root_pct=100 * (mean / root - 1),
                mean_y=float(y.mean()), frac_y_small=float(np.mean(y < 0.05)), n_traj=n_traj, clip=clip)
            print(f"  D/{tag:10s} K_base={kbase:6.3f}: <K_D> = {mean:.3f} +/- {1.96 * se:.3f}  "
                  f"mean-bias {target:.3f} (dev {100 * (mean / target - 1):+.1f} %)  exact root {root:.3f} "
                  f"(dev {100 * (mean / root - 1):+.1f} %)  <y>={y.mean():.3f}  frac(y<0.05)={np.mean(y < 0.05):.2f}")
    M.set_actuator("exp", kbase=kbase0)
    return res


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n_traj, dt = (12, 0.02) if quick else (50, 0.01)
    n_isi = 12 if quick else 40
    t0 = time.time()
    print("A: symbol-rate regimes"); A = exp_A(n_traj, dt, quick)
    print("B: step tracking"); B = exp_B(n_traj, dt, quick)
    print("C: sinusoidal tracking"); Cc = exp_C(n_traj, dt, quick)
    print("D: equilibrium affinity under ISI"); Dd = exp_D(n_isi, dt, quick)
    np.savez_compressed(os.path.join(OUT, "tracking.npz"),
                        A_fast_t=A["fast"]["t"], A_fast_kd=A["fast"]["kd_mean"], A_fast_kds=A["fast"]["kd_std"],
                        A_fast_th=A["fast"]["th_mean"], A_fast_ths=A["fast"]["th_std"],
                        A_fast_th0=A["fast"]["th_traj0"], A_fast_bits0=A["fast"]["bits_traj0"],
                        A_slow_t=A["slow"]["t"], A_slow_kd=A["slow"]["kd_mean"], A_slow_kds=A["slow"]["kd_std"],
                        A_slow_th=A["slow"]["th_mean"], A_slow_ths=A["slow"]["th_std"],
                        B_t=B["t"], B_kd=B["kd_mean"], B_kds=B["kd_std"], B_kdopt=B["kd_opt"],
                        C_t=Cc["t"], C_kd=Cc["kd_mean"], C_kds=Cc["kd_std"], C_kdopt=Cc["kd_opt"])
    numbers = dict(alpha=ALPHA, tau=TAU, tau_settle=TAU,
                   A={k: {kk: v[kk] for kk in ("kd_final", "th0_final", "th1_final", "th0_end", "th1_end", "kd_rel_sd")}
                      for k, v in A.items()},
                   B=dict(gains=B["gains"], t_c=B["t_c"], reconvergence=B["reconvergence"]),
                   C=dict(S=Cc["S"], T=Cc["T"], mean_abs_log_err=Cc["mean_abs_log_err"],
                          mean_abs_log_err_traj=Cc["mean_abs_log_err_traj"], lag_s=Cc["lag_s"]),
                   D=Dd)
    json.dump(numbers, open(os.path.join(OUT, "tracking.json"), "w"), indent=1, default=float)
    print(f"\nwrote results/tracking.npz, tracking.json  ({time.time() - t0:.0f} s)")
