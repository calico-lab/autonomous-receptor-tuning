"""
run_waveform_isi.py -- the CIR-based model at several normalized symbol durations (Section IV-B, Fig. 4).

Repeats the time-varying comparison of run_waveform.py (non-cooperative receiver at its design point, Omega = 200,
K_base = 20, setpoint s* calibrated to each waveform, S = 5, T_c = 240 and 960) for the normalized symbol durations
T_sym / t_peak = 3, 4, 6, 8, 12, and 16.  The symbol duration T_sym = 0.5 and all biochemical rates except the
reference rate mu = s* theta are retained, and the peak time is varied through the diffusion coefficient,
t_peak = d^2 / (6 D) = T_sym / ratio with d = 1.  The CIR is normalized to its peak.  For each value, the script
reports, on the static channel, the fixed-affinity optimum K_best and its BEP, the setpoint s*, the closed-loop BEP,
and the relative affinity variance, and, on the time-varying channel, the adaptive, best non-adaptive, and
genie-aided receivers, the BEP reduction factor, and the tracking error.  The value 3 reproduces run_waveform.py.

Outputs: results/waveform_isi.json and the LaTeX macros manuscript/numbers/numbers_isi.tex.
Options: --quick (fewer trajectories), --retuned (also run the actuator gain re-selected for each waveform),
--macros-only (rewrite the macros from the stored results without simulating).
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from engine import tauleap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
NUMBERS = os.path.join(HERE, "..", "manuscript", "numbers")
DP = json.load(open(os.path.join(OUT, "design_points.json")))["design"]
ALPHA = DP["allosteric"]["alpha"]
C0, C1, T_SYM, NR = M.CL0_NOM, M.CL1_NOM, M.T_SYM, M.N_RECEPTORS
d_ = 1.0
KBASE_WF = 20.0
RATIOS = (3, 4, 6, 8, 12, 16)
S_SW, T_SW_LIST = 5.0, (240.0, 960.0)


def h(tt, D):
    tt = np.maximum(np.asarray(tt, float), 1e-12)
    return np.where(tt > 1e-12, (4 * np.pi * D * tt) ** -1.5 * np.exp(-d_ ** 2 / (4 * D * tt)), 0.0)


def build(n_traj, t_end, dt, seed, D):
    """Waveform on the dt grid (peak of symbol j at t = j T_sym), the bits, and the peak samples."""
    t_peak = d_ ** 2 / (6 * D)
    n_steps = int(round(t_end / dt)) + 1
    rng = np.random.default_rng(seed)
    n_sym = int(np.ceil(t_end / T_SYM)) + 2
    bits = rng.integers(0, 2, (n_traj, n_sym))
    levels = np.where(bits == 1, C1, C0).astype(float)
    kern = h(np.arange(0, 80 * T_SYM, dt), D) / h(t_peak, D)
    imp = np.zeros((n_traj, n_steps + int(round(t_peak / dt)) + 1))
    for j in range(n_sym):
        i0 = int(round((j * T_SYM - t_peak) / dt))
        if 0 <= i0 < imp.shape[1]:
            imp[:, i0] = levels[:, j]
    wave = np.maximum(fftconvolve(imp, kern[None, :], axes=1)[:, :n_steps], 0.0)
    peak_idx = np.array([int(round(j * T_SYM / dt)) for j in range(n_sym) if j * T_SYM <= t_end + 1e-9])
    peak_vals = wave[:, peak_idx]
    return dict(wave=wave, bits=bits[:, :peak_vals.shape[1]], peak_vals=peak_vals, t_peak=t_peak, n_steps=n_steps)


def bep_peaks(peak_vals, bits, rec_sel, K=None, y=None, alpha=ALPHA):
    """Exact-binomial BEP (fixed threshold N_R/2, random tie-breaking) over the recorded peak samples."""
    c = peak_vals[:, rec_sel].T
    b = bits[:, rec_sel].T
    Kd = K if y is None else M.kd_of_y(y, alpha)
    th = np.clip(c / (c + Kd), 1e-12, 1 - 1e-12)
    m = NR // 2
    e = np.where(b == 0, binom.sf(m, NR, th) + 0.5 * binom.pmf(m, NR, th),
                 binom.cdf(m - 1, NR, th) + 0.5 * binom.pmf(m, NR, th))
    return float(e.mean())


def run_loop(W, n_traj, t_end, dt, t_burn, seed, setpoint, gain=1.0, alpha=ALPHA):
    n_steps = W.shape[1]
    fn = lambda t: gain * W[:, min(int(round(t / dt)), n_steps - 1)]
    M.set_rates(MU=setpoint * M.THETA)
    try:
        t, y, _, _, clip = tauleap("allosteric", alpha, fn, C1, omega=M.OMEGA_HC, n_traj=n_traj, t_end=t_end,
                                   dt=dt, t_burn=t_burn, dt_rec=T_SYM, seed=seed, drive="instantaneous",
                                   init_levels=(gain * C0, gain * C1))
    finally:
        M.set_rates()
    return y, clip


def best_static(pv, bits, rec, n_use):
    """Fixed K_D and threshold optimized on the recorded (attenuated) peak samples, as in run_waveform.py."""
    cs = pv[:n_use][:, rec].ravel(); bs = bits[:n_use][:, rec].ravel()
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
    return best


def loop_gain(alpha, K_best, s_star, avg_tt):
    """Loop gain b = k rho theta zeta_eff under the CIR-based model: y* from K_base and K_best, rho = z1*/(z1*+z2*) with the
    setpoint s* (mu = s* theta), and zeta_eff = alpha <theta_B (1 - theta_B)>_t over the waveform at K_best."""
    y_star = np.log(KBASE_WF / K_best) / alpha
    z1 = M.GP * y_star / M.K
    z2 = (s_star * M.THETA) / (M.ETA * z1)
    rho = z1 / (z1 + z2)
    return M.K * rho * M.THETA * alpha * avg_tt, rho


def fix(x, n):
    return f"{x:.{n}f}"


def sci(x, n=2):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.{n - 1}f}\\times10^{{{e}}}"


def write_macros(res, path=os.path.join(NUMBERS, "numbers_isi.tex")):
    """LaTeX macros of Section IV-B and Fig. 4 from the stored results."""
    # T_c = 240 without suffix, T_c = 960 with the suffix "Long"; design actuator gain unless the suffix is "Re"
    lines = ["% numbers_isi.tex -- generated by code/run_waveform_isi.py; do not edit."]
    names = {3: "Three", 4: "Four", 6: "Six", 8: "Eight", 12: "Twelve", 16: "Sixteen"}
    lines.append(f"\\newcommand{{\\physPulsedTcLong}}{{{fix(960.0 * M.TIME_UNIT_S, 0)}}}")
    for ratio, r in res["ratios"].items():
        nm = names[int(ratio)]
        d_um = float(np.sqrt(6.0 * 100.0 * (T_SYM * M.TIME_UNIT_S) / int(ratio)))   # d for t_peak = T_sym/ratio at D = 100 um^2/s (Section V)
        lines += [f"\\newcommand{{\\wfIsi{nm}FirstTapPct}}{{{int(np.floor(100 * r['first_tap'] + 0.5))}}}",
                  f"\\newcommand{{\\physIsi{nm}Dist}}{{{fix(d_um, 0)}}}",
                  f"\\newcommand{{\\wfIsi{nm}TailSum}}{{{fix(r['tail_sum'], 2)}}}",
                  f"\\newcommand{{\\wfIsi{nm}Kbest}}{{{fix(r['K_best_fixed'], 2)}}}",
                  f"\\newcommand{{\\wfIsi{nm}BepFixed}}{{{sci(r['bep_best_fixed'])}}}",
                  f"\\newcommand{{\\wfIsi{nm}Sstar}}{{{fix(r['s_star'], 3)}}}",
                  f"\\newcommand{{\\wfIsi{nm}AvgSlope}}{{{fix(r['avg_theta_one_minus_theta'], 3)}}}",
                  f"\\newcommand{{\\wfIsi{nm}LoopGainPct}}{{{fix(100 * r['b_eff_design_alpha'] / res['b_design'], 0)}}}",
                  f"\\newcommand{{\\wfIsi{nm}AlphaRe}}{{{fix(r['alpha_retuned'], 2)}}}"]
        for tag, sfx in (("design", ""), ("retuned", "Re")):
            if tag not in r["static"]:
                continue
            st = r["static"][tag]
            lines += [f"\\newcommand{{\\wfIsi{nm}BepLoop{sfx}}}{{{sci(st['bep'])}}}",
                      f"\\newcommand{{\\wfIsi{nm}CostFeedback{sfx}}}{{{fix(st['cost_feedback'], 1) if st['cost_feedback'] < 10 else fix(st['cost_feedback'], 0)}}}",
                      f"\\newcommand{{\\wfIsi{nm}RelKdVar{sfx}}}{{{fix(st['rel_kd_var'], 4)}}}",
                      f"\\newcommand{{\\wfIsi{nm}OmegaVarY{sfx}}}{{{fix(st['omega_var_y'], 1)}}}"]
        for T_key, tsfx in (("240", ""), ("960", "Long")):
            sw = r["swing"][T_key]
            lines += [f"\\newcommand{{\\wfIsi{nm}Sw{tsfx}Static}}{{{sci(sw['static_best'])}}}",
                      f"\\newcommand{{\\wfIsi{nm}Sw{tsfx}Oracle}}{{{sci(sw['oracle'])}}}"]
            for tag, sfx in (("design", ""), ("retuned", "Re")):
                if tag not in sw:
                    continue
                e = sw[tag]; g = e["gain"]
                lines += [f"\\newcommand{{\\wfIsi{nm}Sw{tsfx}Adaptive{sfx}}}{{{sci(e['adaptive'])}}}",
                          f"\\newcommand{{\\wfIsi{nm}Sw{tsfx}Gain{sfx}}}{{{fix(g, 1) if g < 10 else fix(g, 0)}}}",
                          f"\\newcommand{{\\wfIsi{nm}Sw{tsfx}Trk{sfx}}}{{{fix(100 * (np.exp(e['track_err_ln']) - 1), 0)}}}"]
    rk = [r["static"]["design"]["rel_kd_var"] for r in res["ratios"].values()]
    lines += [f"\\newcommand{{\\wfIsiRelKdVarMin}}{{{fix(min(rk), 4)}}}", f"\\newcommand{{\\wfIsiRelKdVarMax}}{{{fix(max(rk), 4)}}}",
              f"\\newcommand{{\\wfIsiPcRelKdVar}}{{{fix(res['pc_rel_kd_var'], 4)}}}"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    if "--macros-only" in sys.argv:
        write_macros(json.load(open(os.path.join(OUT, "waveform_isi.json"))))
        sys.exit(0)
    quick = "--quick" in sys.argv
    RETUNED = "--retuned" in sys.argv                     # also run the actuator gain re-selected for each waveform (diagnostic)
    n_traj, dt = (12, 0.02) if quick else (40, 0.01)
    t_burn, t_win = 300.0, 300.0
    t_end = t_burn + t_win
    t0 = time.time()
    from scipy.optimize import brentq
    M.set_actuator("exp", kbase=KBASE_WF)
    b_design = M.K * DP["allosteric"]["rho"] * M.THETA * DP["allosteric"]["zeta"]
    res = dict(alpha=ALPHA, kbase=KBASE_WF, t_sym=T_SYM, n_traj=n_traj, S=S_SW, T_list=list(T_SW_LIST), b_design=b_design, ratios={})
    HC = json.load(open(os.path.join(OUT, "bep_hc.json")))["configs"]["allosteric|AIF"]
    res["pc_rel_kd_var"] = ALPHA ** 2 * HC["var_dy_omega"] / M.OMEGA_HC        # alpha^2 Var(dy) of the piecewise-constant static run (Table II)
    try:
        for ratio in RATIOS:
            D = ratio / (6.0 * T_SYM)                         # t_peak = d^2/(6 D) = T_sym / ratio
            B = build(n_traj, t_end, dt, seed=M.stable_seed("wf-levels", base=9), D=D)
            tp = B["t_peak"]
            rec = np.array([j for j in range(B["peak_vals"].shape[1]) if j * T_SYM >= t_burn - 1e-9])
            pv, bits = B["peak_vals"], B["bits"]
            taps = np.array([h(tp + m * T_SYM, D) / h(tp, D) for m in range(1, 80)])
            cbar = float(taps.sum()) * 0.5 * (C0 + C1)
            # fixed-affinity optimum of the peak-sample error on the static channel, and the setpoint s*
            Ks = np.geomspace(0.5, 30, 300)
            beps = [bep_peaks(pv, bits, rec, K=Kx) for Kx in Ks]
            K_best = float(Ks[int(np.argmin(beps))]); bep_best = float(min(beps))
            sub = B["wave"][:, int(t_burn / dt):]
            s_star = float(np.mean(sub / (sub + K_best)))
            th_t = sub / (sub + K_best)
            avg_tt = float(np.mean(th_t * (1 - th_t)))                    # <theta_B (1 - theta_B)>_t at K_best
            b_eff, rho_eff = loop_gain(ALPHA, K_best, s_star, avg_tt)
            alpha_re = float(brentq(lambda a: loop_gain(a, K_best, s_star, avg_tt)[0] - b_design, 0.05, 5.0))
            entry = dict(ratio=ratio, D=D, t_peak=tp, first_tap=float(taps[0]), tail_sum=float(taps.sum()), cbar=cbar,
                         K_best_fixed=K_best, bep_best_fixed=bep_best, s_star=s_star, avg_theta_one_minus_theta=avg_tt,
                         b_eff_design_alpha=b_eff, rho_eff=rho_eff, alpha_retuned=alpha_re, static={}, swing={})
            # closed loop on the static channel (setpoint s*), at the design actuator gain and, optionally, the re-selected one
            variants = (("design", ALPHA),) + ((("retuned", alpha_re),) if RETUNED else ())
            for tag, alpha in variants:
                y, clip_st = run_loop(B["wave"], n_traj, t_end, dt, t_burn, seed=M.stable_seed("wf", "waveform_renorm", base=9),
                                      setpoint=s_star, alpha=alpha)
                per = M.kd_of_y(y, alpha).mean(axis=0)                            # time average per trajectory
                vy = float(np.var(y - y.mean(axis=0, keepdims=True)))            # modulator variance about the per-trajectory mean
                bep_cl = bep_peaks(pv, bits, rec, y=y, alpha=alpha)
                entry["static"][tag] = dict(alpha=alpha, kd_mean=float(per.mean()), ci95=float(1.96 * per.std(ddof=1) / np.sqrt(n_traj)),
                                            bep=bep_cl, cost_feedback=bep_cl / bep_best, clip=clip_st,
                                            omega_var_y=vy * M.OMEGA_HC, rel_kd_var=alpha ** 2 * vy)
            print(f"ratio {ratio:2d}: t_peak={tp:.4f} first tap={taps[0]:.3f} tail sum={taps.sum():.3f} | fixed optimum K_best={K_best:.3f} "
                  f"BEP={bep_best:.3e} s*={s_star:.4f} <th(1-th)>={avg_tt:.4f} b_eff/b_design={b_eff / b_design:.2f} alpha_re={alpha_re:.3f} | "
                  f"static loop BEP={entry['static']['design']['bep']:.3e} (cost {entry['static']['design']['cost_feedback']:.2f})  [{time.time() - t0:.0f} s]", flush=True)
            # time-varying channel: the whole waveform attenuated by g(t), for each coherence time
            lnS = np.log(S_SW)
            for T_sw in T_SW_LIST:
                g_t = lambda tt, T=T_sw: float(np.exp(-0.5 * lnS * (1.0 - np.cos(2 * np.pi * tt / T))))
                t_end_sw = 200.0 + 2 * T_sw
                Bsw = build(n_traj, t_end_sw, dt, seed=M.stable_seed("wf-levels-sw", base=9), D=D)
                gvec = np.array([g_t(i * dt) for i in range(Bsw["n_steps"])])
                Wsw = Bsw["wave"] * gvec[None, :]
                rec_sw = np.array([j for j in range(Bsw["peak_vals"].shape[1]) if j * T_SYM >= 200.0 - 1e-9])
                g_peaks = np.array([g_t(j * T_SYM) for j in range(Bsw["peak_vals"].shape[1])])
                pv_sw = Bsw["peak_vals"] * g_peaks[None, :]
                K_or = K_best * g_peaks[rec_sw]
                c = pv_sw[:, rec_sw].T; bb = Bsw["bits"][:, rec_sw].T
                th = np.clip(c / (c + K_or[:, None]), 1e-12, 1 - 1e-12); m = NR // 2
                e = np.where(bb == 0, binom.sf(m, NR, th) + 0.5 * binom.pmf(m, NR, th), binom.cdf(m - 1, NR, th) + 0.5 * binom.pmf(m, NR, th))
                b_or = float(e.mean())
                bst = best_static(pv_sw, Bsw["bits"], rec_sw, min(n_traj, 8))
                sw = dict(T=T_sw, oracle=b_or, static_best=bst[0], static_kd=bst[1], static_thr=bst[2])
                for tag, alpha in variants:
                    y_sw, clip_sw = run_loop(Wsw, n_traj, t_end_sw, dt, 200.0, seed=M.stable_seed("wf-sw", base=9), setpoint=s_star, alpha=alpha)
                    b_ad = bep_peaks(pv_sw, Bsw["bits"], rec_sw, y=y_sw, alpha=alpha)
                    kd_sw = M.kd_of_y(y_sw, alpha)                                    # (n_rec, n_traj)
                    tgt = K_best * g_peaks[rec_sw][:kd_sw.shape[0]]
                    trk = float(np.mean(np.abs(np.log(kd_sw.mean(axis=1) / tgt))))          # tracking error of the ensemble-mean affinity (as in run_headline.py)
                    trk_traj = float(np.mean(np.abs(np.log(kd_sw / tgt[:, None]))))          # per-trajectory (includes the noise)
                    sw[tag] = dict(alpha=alpha, adaptive=b_ad, gain=bst[0] / b_ad, track_err_ln=trk, track_err_ln_traj=trk_traj, clip=clip_sw)
                    print(f"    T_c={T_sw:4.0f} {tag:8s}: adaptive={b_ad:.3e} static={bst[0]:.3e} (K={bst[1]:.3f}, thr={bst[2]:.0f}) "
                          f"oracle={b_or:.3e} gain={bst[0] / b_ad:.3g} trk={100 * trk:.0f}% (per-traj {100 * trk_traj:.0f}%)  [{time.time() - t0:.0f} s]", flush=True)
                entry["swing"][str(int(T_sw))] = sw
            res["ratios"][str(ratio)] = entry
    finally:
        M.set_actuator("exp", kbase=M.KBASE)
    json.dump(res, open(os.path.join(OUT, "waveform_isi.json"), "w"), indent=1, default=float)
    write_macros(res)
    print(f"\nwrote results/waveform_isi.json  ({time.time() - t0:.0f} s)")
