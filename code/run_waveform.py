"""
run_waveform.py -- the CIR-based model under strong ISI (Section IV-A).

The controller senses the receptor occupancy continuously, whereas the detector samples it at the peak of every pulse.
Under the CIR-based model, the time average of the bit-averaged occupancy at the affinity that is optimal for the peak
samples lies below one half, such that a controller with a setpoint of one half settles below this affinity.  The
remedy is to set the reference to the waveform-averaged occupancy at the optimal affinity,
    s* = < [theta_B(a0 q(t); K_peak) + theta_B(a1 q(t); K_peak)] / 2 >_t,
which depends on the normalized waveform and the symbol statistics but is invariant under a common attenuation g,
since theta_B is homogeneous of degree zero in (c, K_D).  The reference affinity K_peak minimizes the error probability
of a fixed receptor over the same symbol streams.  A time-varying case (S = 5, T_c = 240) compares the adaptive
receiver with the best non-adaptive and the genie-aided receivers on the same channel.

Non-cooperative receiver at its design point, Omega = 200, random bits independent for every trajectory, strong ISI
(T_sym = 3 t_peak), and K_base = 20, such that every target affinity lies inside the actuator range.
Output: results/waveform.json
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import brentq
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from engine import tauleap

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
DP = json.load(open(os.path.join(OUT, "design_points.json")))["design"]
ALPHA = DP["allosteric"]["alpha"]
C0, C1, T_SYM, NR = M.CL0_NOM, M.CL1_NOM, M.T_SYM, M.N_RECEPTORS
D_, d_ = 1.0, 1.0
KBASE_WF = 20.0


def h(tt):
    tt = np.maximum(np.asarray(tt, float), 1e-12)
    return np.where(tt > 1e-12, (4 * np.pi * D_ * tt) ** -1.5 * np.exp(-d_ ** 2 / (4 * D_ * tt)), 0.0)


def build(n_traj, t_end, dt, seed):
    """Received signal c(t) under the CIR-based model on the dt grid (peak of symbol j at t = j T_sym), the
    piecewise-constant levels sampled at the peaks, the bits, and the peak samples."""
    t_peak = d_ ** 2 / (6 * D_)
    n_steps = int(round(t_end / dt)) + 1
    rng = np.random.default_rng(seed)
    n_sym = int(np.ceil(t_end / T_SYM)) + 2
    bits = rng.integers(0, 2, (n_traj, n_sym))
    levels = np.where(bits == 1, C1, C0).astype(float)
    kern = h(np.arange(0, 80 * T_SYM, dt)) / h(t_peak)
    imp = np.zeros((n_traj, n_steps + int(round(t_peak / dt)) + 1))
    for j in range(n_sym):
        i0 = int(round((j * T_SYM - t_peak) / dt))
        if 0 <= i0 < imp.shape[1]:
            imp[:, i0] = levels[:, j]
    wave = np.maximum(fftconvolve(imp, kern[None, :], axes=1)[:, :n_steps], 0.0)
    peak_idx = np.array([int(round(j * T_SYM / dt)) for j in range(n_sym) if j * T_SYM <= t_end + 1e-9])
    peak_vals = wave[:, peak_idx]
    idx_sym = (np.arange(n_steps) * dt / T_SYM).astype(int).clip(0, peak_vals.shape[1] - 1)
    held = peak_vals[:, idx_sym]
    return dict(wave=wave, held=held, bits=bits[:, :peak_vals.shape[1]], peak_vals=peak_vals,
                t_peak=t_peak, n_steps=n_steps)


def bep_peaks(peak_vals, bits, rec_sel, K=None, y=None, alpha=ALPHA):
    """Exact-binomial BEP (randomized midpoint rule) over the recorded peak samples."""
    c = peak_vals[:, rec_sel].T
    b = bits[:, rec_sel].T
    Kd = K if y is None else M.kd_of_y(y, alpha)
    th = np.clip(c / (c + Kd), 1e-12, 1 - 1e-12)
    m = NR // 2
    e = np.where(b == 0, binom.sf(m, NR, th) + 0.5 * binom.pmf(m, NR, th),
                 binom.cdf(m - 1, NR, th) + 0.5 * binom.pmf(m, NR, th))
    return float(e.mean())


def run_loop(W, n_traj, t_end, dt, t_burn, seed, setpoint=None, gain=1.0):
    n_steps = W.shape[1]
    fn = lambda t: gain * W[:, min(int(round(t / dt)), n_steps - 1)]
    if setpoint is not None:
        M.set_rates(MU=setpoint * M.THETA)
    try:
        t, y, _, _, clip = tauleap("allosteric", ALPHA, fn, C1, omega=M.OMEGA_HC, n_traj=n_traj, t_end=t_end,
                                   dt=dt, t_burn=t_burn, dt_rec=T_SYM, seed=seed, drive="instantaneous",
                                   init_levels=(gain * C0, gain * C1))
    finally:
        M.set_rates()
    return y, clip


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n_traj, dt = (12, 0.02) if quick else (40, 0.01)
    t_burn, t_win = 300.0, 300.0
    t_end = t_burn + t_win
    t0 = time.time()
    M.set_actuator("exp", kbase=KBASE_WF)
    B = build(n_traj, t_end, dt, seed=M.stable_seed("wf-levels", base=9))
    rec_sel = np.array([j for j in range(B["peak_vals"].shape[1]) if j * T_SYM >= t_burn - 1e-9])
    pv, bits = B["peak_vals"], B["bits"]
    # references
    cbar = float(np.array([h(B["t_peak"] + m * T_SYM) / h(B["t_peak"]) for m in range(1, 80)]).sum()) * 0.5 * (C0 + C1)
    K_mb = float(np.sqrt((C0 + cbar) * (C1 + cbar)))
    cs = pv[:, rec_sel]
    K_held = float(brentq(lambda Kx: np.mean(cs / (cs + Kx)) - 0.5, 0.05, 200))
    sub = B["wave"][:, int(t_burn / dt):]
    K_cont = float(brentq(lambda Kx: np.mean(sub / (sub + Kx)) - 0.5, 0.05, 200))
    Ks = np.geomspace(0.5, 30, 300)
    beps = [bep_peaks(pv, bits, rec_sel, K=Kx) for Kx in Ks]
    K_best = float(Ks[int(np.argmin(beps))]); bep_best = float(min(beps))
    # setpoint calibrated at the exact fixed-affinity optimum of the peak-sample error over this stream
    s_star = float(np.mean(sub / (sub + K_best)))
    s_star_root = float(np.mean(sub / (sub + K_held)))
    inv = {g: float(np.mean(g * sub / (g * sub + g * K_best))) for g in (0.3, 3.0)}
    res = dict(alpha=ALPHA, kbase=KBASE_WF, t_peak=B["t_peak"], t_sym=T_SYM, cbar=cbar, K_meanbias=K_mb,
               K_held_root=K_held, K_cont_root=K_cont, K_best_fixed=K_best, bep_best_fixed=bep_best,
               bep_at_K_meanbias=bep_peaks(pv, bits, rec_sel, K=K_mb),
               bep_at_K_cont=bep_peaks(pv, bits, rec_sel, K=K_cont), bep_at_K_held=bep_peaks(pv, bits, rec_sel, K=K_held),
               s_star=s_star, s_star_at_root=s_star_root, s_star_invariance=inv, n_traj=n_traj, cases={})
    print(f"mean-bias K = {K_mb:.3f}; held-level root {K_held:.3f}; continuous-sensing root {K_cont:.3f}; "
          f"fixed-affinity optimum {K_best:.3f} (BEP {bep_best:.3e}); s* at the optimum = {s_star:.4f} (at the root {s_star_root:.4f}); invariance {inv}")
    for tag, W, sp, gain in (("held", B["held"], None, 1.0), ("waveform", B["wave"], None, 1.0),
                             ("waveform_renorm", B["wave"], s_star, 1.0),
                             ("waveform_renorm_g0p3", B["wave"], s_star, 0.3)):
        target = gain * K_best
        y, clip = run_loop(W, n_traj, t_end, dt, t_burn, seed=M.stable_seed("wf", tag, base=9), setpoint=sp, gain=gain)
        kd = M.kd_of_y(y, ALPHA)
        per = kd.mean(axis=0)
        # BEP at the peaks: scale the peak samples by the gain (the detector sees the attenuated waveform)
        b = bep_peaks(gain * pv, bits, rec_sel, y=y)
        res["cases"][tag] = dict(kd_mean=float(per.mean()), ci95=float(1.96 * per.std(ddof=1) / np.sqrt(n_traj)),
                                 bep=b, clip=clip, setpoint=(0.5 if sp is None else sp), gain=gain,
                                 target=target, bep_fixed_at_target=bep_peaks(gain * pv, bits, rec_sel, K=target))
        print(f"  {tag:22s}: <K_D> = {per.mean():.3f} +/- {1.96 * per.std(ddof=1) / np.sqrt(n_traj):.3f}  "
              f"(fixed-affinity optimum {target:.3f}, BEP {res['cases'][tag]['bep_fixed_at_target']:.3e})  "
              f"closed-loop BEP at the peaks = {b:.3e}  clip={clip:.1e}", flush=True)

    # ---- time-varying channel under the CIR-based model: the whole waveform attenuated by g(t), depth S, period T ----
    S_sw, T_sw = 5.0, 240.0
    lnS = np.log(S_sw)
    g_t = lambda tt: float(np.exp(-0.5 * lnS * (1.0 - np.cos(2 * np.pi * tt / T_sw))))
    t_end_sw = 200.0 + 2 * T_sw
    Bsw = build(n_traj, t_end_sw, dt, seed=M.stable_seed("wf-levels-sw", base=9))
    gvec = np.array([g_t(i * dt) for i in range(Bsw["n_steps"])])
    Wsw = Bsw["wave"] * gvec[None, :]
    rec_sw = np.array([j for j in range(Bsw["peak_vals"].shape[1]) if j * T_SYM >= 200.0 - 1e-9])
    g_peaks = np.array([g_t(j * T_SYM) for j in range(Bsw["peak_vals"].shape[1])])
    pv_sw = Bsw["peak_vals"] * g_peaks[None, :]          # attenuated peak samples
    fn_sw = lambda t: Wsw[:, min(int(round(t / dt)), Bsw["n_steps"] - 1)]
    M.set_rates(MU=s_star * M.THETA)
    try:
        t, y, _, _, clip = tauleap("allosteric", ALPHA, fn_sw, C1, omega=M.OMEGA_HC, n_traj=n_traj, t_end=t_end_sw,
                                   dt=dt, t_burn=200.0, dt_rec=T_SYM, seed=M.stable_seed("wf-sw", base=9),
                                   drive="instantaneous", init_levels=(C0, C1))
    finally:
        M.set_rates()
    b_ad = bep_peaks(pv_sw, Bsw["bits"], rec_sw, y=y)
    # genie-aided receiver: fixed-affinity optimum rescaled with the attenuation, without noise
    K_or = K_best * g_peaks[rec_sw]
    c = pv_sw[:, rec_sw].T; bb = Bsw["bits"][:, rec_sw].T
    th = np.clip(c / (c + K_or[:, None]), 1e-12, 1 - 1e-12); m = NR // 2
    e = np.where(bb == 0, binom.sf(m, NR, th) + 0.5 * binom.pmf(m, NR, th), binom.cdf(m - 1, NR, th) + 0.5 * binom.pmf(m, NR, th))
    b_or = float(e.mean())
    # best non-adaptive receiver on the time-varying channel: fixed K and threshold optimized on the recorded peak samples
    sub_tr = slice(0, min(n_traj, 8))
    cs = pv_sw[sub_tr][:, rec_sw].ravel(); bs = Bsw["bits"][sub_tr][:, rec_sw].ravel()
    best = (np.inf, None, None)
    for Kx in np.geomspace(0.05, 30, 80):
        thx = np.clip(cs / (cs + Kx), 1e-12, 1 - 1e-12)
        for thr in np.arange(60, 141, 1.0):
            mm = int(thr)
            e0 = binom.sf(mm, NR, thx) + 0.5 * binom.pmf(mm, NR, thx)
            e1 = binom.cdf(mm - 1, NR, thx) + 0.5 * binom.pmf(mm, NR, thx)
            v = float(np.where(bs == 0, e0, e1).mean())
            if v < best[0]:
                best = (v, float(Kx), float(thr))
    res["pulsed_swing"] = dict(S=S_sw, T=T_sw, adaptive=b_ad, oracle=b_or, static_best=best[0], static_kd=best[1],
                               static_thr=best[2], gain=best[0] / b_ad, setpoint=s_star, clip=clip)
    print(f"  time-varying channel S={S_sw:g}, T={T_sw:g}: adaptive (setpoint s*) {b_ad:.3e}, best static {best[0]:.3e} "
          f"(K={best[1]:.3f}, thr={best[2]:.0f}), oracle {b_or:.3e}, gain {best[0] / b_ad:.3g}", flush=True)
    M.set_actuator("exp", kbase=M.KBASE)
    json.dump(res, open(os.path.join(OUT, "waveform.json"), "w"), indent=1, default=float)
    print(f"\nwrote results/waveform.json  ({time.time() - t0:.0f} s)")
