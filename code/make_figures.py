"""
make_figures.py -- data figures of the paper, generated from ../results only.

    Fig. 2  fig2_tracking.pdf         symbol averaging and channel tracking (run_tracking.py)
    Fig. 3  fig3_headline.pdf         adaptive vs. non-adaptive receivers on the time-varying channel (run_headline.py)
    Fig. 5  fig5_design_rule.pdf      loop-speed design rule (run_design_rule.py)
    Fig. 6  fig6_static_channel.pdf   static channel: operating points, copy-number dependence, and stability boundary
                                      (run_bep_hc.py, run_bep_lc.py, run_zeta_crit.py)
Fig. 1 is drawn by make_fig1_schematic.py and Fig. 4 by make_fig4_cir_sweep.py.  The figures are written as PDF (for
LaTeX) and PNG (for inspection) to ../manuscript/figures.  Error probabilities below the reporting threshold of 1e-8 are
drawn at the threshold with open markers.
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "manuscript", "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, HERE)
import model as M

FLOOR = 1e-8
COL = {"allosteric": "#0072B2", "cooperative": "#009E73", "coregulated": "#CC79A7",
       "weighted": "#E69F00", "cooperative_a1": "#009E73", "coregulated_slow": "#7B3294",
       "static": "#000000", "oracle": "#7f7f7f"}
LBL = {"allosteric": r"non-cooperative, $n_\mathrm{H}=1$", "cooperative": r"cooperative, $n_\mathrm{H}=2.5$",
       "coregulated": r"co-regulated, $\tau=4$", "weighted": r"mixed array ($w=0.6$)",
       "cooperative_a1": r"cooperative, $\alpha=1$", "coregulated_slow": r"co-regulated, $\tau=10$"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300,
    "pdf.fonttype": 42, "axes.unicode_minus": False,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def panel_label(ax, s, x=-0.22, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def floored(v):
    v = np.asarray(v, float)
    return np.maximum(v, FLOOR), v < FLOOR


def loginterp_x(xs, ys, y_target):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    for j in range(1, len(xs)):
        if ys[j] >= y_target > ys[j - 1]:
            f = (np.log(y_target) - np.log(ys[j - 1])) / (np.log(ys[j]) - np.log(ys[j - 1]))
            return float(np.exp(np.log(xs[j - 1]) + f * (np.log(xs[j]) - np.log(xs[j - 1]))))
    return None


# =========================================================================== #
#  Fig. 2 -- tracking
# =========================================================================== #
def fig2():
    d = np.load(os.path.join(RES, "tracking.npz"))
    tr = json.load(open(os.path.join(RES, "tracking.json")))
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.1))
    # (a) K_D under fast random symbols and slow alternating symbols
    ax = axs[0, 0]
    for tag, c, lab in (("fast", COL["allosteric"], r"$T_\mathrm{sym}=0.5$"),
                        ("slow", COL["weighted"], r"$T_\mathrm{sym}=40$")):
        t, kd, ks = d[f"A_{tag}_t"], d[f"A_{tag}_kd"], d[f"A_{tag}_kds"]
        m = t <= 130
        ax.plot(t[m], kd[m], color=c, lw=0.9, label=lab)
        ax.fill_between(t[m], (kd - ks)[m], (kd + ks)[m], color=c, alpha=0.18, lw=0)
    ax.axhline(M.KD_OPT, color="k", ls="--", lw=0.7)
    ax.text(30, M.KD_OPT * 1.12, r"$\sqrt{c_0c_1}$", ha="center", va="bottom", fontsize=7)
    ax.set_xlim(0, 130); ax.set_ylim(0, 6); ax.set_xlabel("time"); ax.set_ylabel(r"$K_\mathrm{D}(y)$")
    ax.legend(loc="upper right", handlelength=1.2, borderaxespad=0.2)
    panel_label(ax, "(a)")
    # (b) one trajectory's occupancy under the random stream (fast)
    ax = axs[0, 1]
    t, th = d["A_fast_t"], d["A_fast_th0"]
    m = (t >= 100) & (t <= 106)
    ax.step(t[m], th[m], color=COL["allosteric"], lw=0.9, where="post")
    for yv in (1 / 3, 2 / 3):
        ax.axhline(yv, color="k", ls=":", lw=0.6)
    ax.text(105.9, 0.29, r"$\theta_0$", ha="right", va="top", fontsize=7)
    ax.text(105.9, 0.70, r"$\theta_1$", ha="right", va="bottom", fontsize=7)
    ax.axhline(0.5, color="k", ls="--", lw=0.7)
    ax.text(100.1, 0.51, "threshold", ha="left", va="bottom", fontsize=6.5)
    ax.set_xlim(100, 106); ax.set_ylim(0.15, 0.85); ax.set_xlabel("time"); ax.set_ylabel(r"$\theta_\mathrm{B}(t)$")
    panel_label(ax, "(b)")
    # (c) step tracking
    ax = axs[1, 0]
    t, kd, ks, ko = d["B_t"], d["B_kd"], d["B_kds"], d["B_kdopt"]
    ax.plot(t, ko, color="k", ls="--", lw=0.8, label=r"$\sqrt{c_0(t)c_1(t)}$")
    ax.plot(t, kd, color=COL["allosteric"], lw=0.9, label=r"$K_\mathrm{D}(y)$")
    ax.fill_between(t, kd - ks, kd + ks, color=COL["allosteric"], alpha=0.18, lw=0)
    ax.set_xlim(0, 440); ax.set_ylim(0, 3.1); ax.set_xlabel("time"); ax.set_ylabel(r"$K_\mathrm{D}$")
    ax.legend(loc="lower right", handlelength=1.2, borderaxespad=0.2)
    rl = [r for r in tr["B"]["reconvergence"] if r is not None]
    ax.text(0.03, 0.95, f"re-convergence {min(rl):.0f} to {max(rl):.0f}", transform=ax.transAxes,
            fontsize=6.5, va="top")
    panel_label(ax, "(c)")
    # (d) sinusoid
    ax = axs[1, 1]
    t, kd, ks, ko = d["C_t"], d["C_kd"], d["C_kds"], d["C_kdopt"]
    ax.plot(t, ko, color="k", ls="--", lw=0.8)
    ax.plot(t, kd, color=COL["allosteric"], lw=0.9)
    ax.fill_between(t, kd - ks, kd + ks, color=COL["allosteric"], alpha=0.18, lw=0)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, 3.1); ax.set_xlabel("time"); ax.set_ylabel(r"$K_\mathrm{D}$")
    ax.text(0.03, 0.95, f"lag {tr['C']['lag_s']:.0f}", transform=ax.transAxes, fontsize=6.5, va="top")
    panel_label(ax, "(d)")
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.6)
    save(fig, "fig2_tracking")


# =========================================================================== #
#  Fig. 3 -- adaptive vs. non-adaptive receivers on the time-varying channel
# =========================================================================== #
def fig3():
    h = json.load(open(os.path.join(RES, "headline.json")))
    rows = h["rows"]; tau = h["tau"]
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.6))
    # (a) BEP vs. variation depth at T_c = 240
    ax = axs[0]
    T0 = 240.0
    for lab, n in (("allosteric", 1.0), ("cooperative", M.N_COOP)):
        rs = sorted([r for r in rows if r["label"] == lab and r["T"] == T0], key=lambda r: r["S"])
        S = np.array([r["S"] for r in rs])
        ad, adf = floored([r["adaptive"] for r in rs])
        lo, _ = floored([r["lo"] for r in rs]); hi, _ = floored([r["hi"] for r in rs])
        st, stf = floored([r["static_best"] for r in rs])
        orc, orf = floored([r["oracle"] for r in rs])
        c = COL[lab]
        ax.plot(S, ad, color=c, lw=1.1, marker="o", ms=3.2, label=f"adaptive, $n_\\mathrm{{H}}={n:g}$")
        ax.scatter(S[adf], ad[adf], s=14, facecolor="white", edgecolor=c, zorder=5, lw=0.8)
        ax.fill_between(S, lo, hi, color=c, alpha=0.2, lw=0)
        ax.plot(S, st, color="k", lw=0.9, ls="-" if lab == "allosteric" else "--", marker="s", ms=2.6,
                mfc="k" if lab == "allosteric" else "white", label=f"best non-adaptive, $n_\\mathrm{{H}}={n:g}$")
        ax.scatter(S[stf], st[stf], s=12, facecolor="white", edgecolor="k", zorder=5, lw=0.8, marker="s")
        if lab == "allosteric":
            ax.plot(S, orc, color=COL["oracle"], lw=0.8, ls=":", label=r"genie-aided, $n_\mathrm{H}=1$")
    ax.axhline(FLOOR, color="0.6", lw=0.5, ls="-")
    ax.text(1.05, FLOOR * 1.4, r"$10^{-8}$ floor", fontsize=6, color="0.4", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 100); ax.set_ylim(FLOOR / 2, 1)
    ax.set_xlabel(r"variation depth $S=\max\,g/\min\,g$"); ax.set_ylabel("BEP")
    ax.set_title(rf"$T_\mathrm{{c}}=240={T0 / tau['allosteric']:.0f}\,\tau$", fontsize=7.5, pad=2)
    ax.legend(loc="lower right", ncol=1, handlelength=1.6, borderaxespad=0.2, bbox_to_anchor=(1.0, 0.04),
              frameon=True, framealpha=0.92, edgecolor="none", fancybox=False)
    rs1 = sorted([r for r in rows if r["label"] == "allosteric" and r["T"] == T0], key=lambda r: r["S"])
    xs = np.array([r["S"] for r in rs1]); ya = np.array([r["adaptive"] for r in rs1]); ys = np.array([r["static_best"] for r in rs1])
    if np.any(ya < ys):
        i = int(np.argmax(ya < ys))
        if i > 0:
            f = (np.log(ys[i-1]) - np.log(ya[i-1])) / ((np.log(ys[i-1]) - np.log(ya[i-1])) - (np.log(ys[i]) - np.log(ya[i])))
            xc = float(np.exp(np.log(xs[i-1]) + f * (np.log(xs[i]) - np.log(xs[i-1]))))
            ax.annotate(f"break-even $S\\approx{xc:.1f}$", xy=(xc, 3e-4), xytext=(3.2, 2e-5), fontsize=6.3,
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.3"), color="0.3")
    panel_label(ax, "(a)", x=-0.18)
    # (b) BEP reduction factor vs. T_c/tau
    ax = axs[1]
    for S0, mk in ((2.0, "o"), (5.0, "s"), (20.0, "^")):
        rs = sorted([r for r in rows if r["label"] == "allosteric" and r["S"] == S0], key=lambda r: r["T"])
        x = np.array([r["T"] / tau["allosteric"] for r in rs]); g = np.array([r["gain"] for r in rs])
        ax.plot(x, g, color=COL["allosteric"], marker=mk, ms=3.2, lw=1.0, label=rf"$n_\mathrm{{H}}=1$, $S={S0:g}$")
    for lab, mk, ls, lab_txt in (("cooperative", "s", "-", r"$n_\mathrm{H}=2.5$, $S=5$"),
                                 ("coregulated_slow", "D", "--", r"co-regulated $\tau=10$, $S=5$"),
                                 ("coregulated", "s", ":", r"co-regulated $\tau=4$, $S=5$")):
        rs = sorted([r for r in rows if r["label"] == lab and r["S"] == 5.0], key=lambda r: r["T"])
        if not rs:
            continue
        x = np.array([r["T"] / tau[lab] for r in rs]); g = np.array([r["gain"] for r in rs])
        ax.plot(x, g, color=COL[lab], marker=mk, ms=3.0, lw=0.9, ls=ls, label=lab_txt)
    ax.axhline(1, color="k", lw=0.7, ls="--")
    ax.text(5.2, 1.25, "break-even", fontsize=6.5, va="bottom")
    rs5 = sorted([r for r in rows if r["label"] == "allosteric" and r["S"] == 5.0], key=lambda r: r["T"])
    x5 = [r["T"] / tau["allosteric"] for r in rs5]; g5 = [r["gain"] for r in rs5]
    on10, on100 = loginterp_x(x5, g5, 10.0), loginterp_x(x5, g5, 100.0)
    if on10 and on100:
        ax.axvspan(on10, on100, color="0.88", lw=0)
        ax.text(np.sqrt(on10 * on100), 4e4, "10× to 100×", ha="center", va="top", fontsize=6.3, color="0.35")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(5, 300); ax.set_ylim(0.3, 1e5)
    ax.set_xlabel(r"$T_\mathrm{c}/\tau$"); ax.set_ylabel(r"BEP reduction factor")
    ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.2)
    panel_label(ax, "(b)", x=-0.18)
    fig.tight_layout(pad=0.3, h_pad=0.9)
    save(fig, "fig3_headline")


# =========================================================================== #
#  Fig. 6 -- static channel: operating points, copy number, stability
# =========================================================================== #
def fig6():
    hc = json.load(open(os.path.join(RES, "bep_hc.json")))
    lc = json.load(open(os.path.join(RES, "bep_lc.json")))
    zc = json.load(open(os.path.join(RES, "zeta_crit.json")))
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.2))
    # (a) operating-point plane
    ax = axs[0]
    sg = np.linspace(0.02, 0.13, 220); dm = np.linspace(0.25, 0.8, 200)
    SG, DM = np.meshgrid(sg, dm)
    B = norm.sf(DM / (2 * SG))
    cs = ax.contour(SG, DM, np.log10(np.maximum(B, 1e-40)), levels=[-12, -8, -6, -4, -3, -2, -1],
                    colors="0.75", linewidths=0.5)
    ax.clabel(cs, fmt=lambda v: rf"$10^{{{int(v)}}}$", fontsize=5.5, inline=True, inline_spacing=2)
    ax.text(0.128, 0.79, "Gaussian approximation\ncontours", ha="right", va="top", fontsize=5.6, color="0.45")
    for key, r in hc["configs"].items():
        label, motif = key.split("|")
        if motif != "AIF":
            continue                                    # the cascaded variant (CAIF) is not shown
        base = "cooperative" if label.startswith("cooperative") else label
        c = COL[base]
        x = max(r["sigma0"], r["sigma1"]); y = r["dmu"]
        mk = "o" if label not in ("cooperative_a1", "coregulated_slow") else "D"
        ax.scatter(x, y, s=22 if motif == "AIF" else 20, marker=mk, facecolor=c if motif == "AIF" else "white",
                   edgecolor=c, lw=0.9, zorder=5)
    for key, r in hc["bitavg_reference"].items():          # averaged-input references without self-noise (small crosses)
        label = key.split("|")[0]
        ax.scatter(max(r["sigma0"], r["sigma1"]), r["dmu"], s=16, marker="x", color=COL[label], lw=0.8, zorder=4)
    ref = hc["references"]
    for n, key in ((1.0, "static_at_kdopt_n1"), (M.N_COOP, f"static_at_kdopt_n{M.N_COOP:g}")):
        ax.scatter(ref[key]["sigma"], ref[key]["dmu"], s=18, marker="*", facecolor="k", edgecolor="k", zorder=6)
    ax.set_xlim(0.02, 0.13); ax.set_ylim(0.25, 0.8)
    ax.set_xlabel(r"detector-input noise $\sigma$"); ax.set_ylabel(r"separation $\Delta\theta$")
    handles = [Line2D([], [], marker="o", ls="", mfc=COL[k], mec=COL[k], ms=4.5, label=LBL[k])
               for k in ("allosteric", "weighted", "cooperative", "coregulated")]
    handles += [Line2D([], [], marker="D", ls="", mfc=COL["coregulated_slow"], mec=COL["coregulated_slow"], ms=3.8,
                       label=LBL["coregulated_slow"]),
                Line2D([], [], marker="D", ls="", mfc=COL["cooperative"], mec=COL["cooperative"], ms=3.8,
                       label=r"cooperative, $\alpha=1$"),
                Line2D([], [], marker="x", ls="", color="k", ms=4, label="averaged input (no self-noise)"),
                Line2D([], [], marker="*", ls="", mfc="k", mec="k", ms=6, label=r"non-adaptive, static channel")]
    ax.legend(handles=handles, loc="lower right", fontsize=5.4, handlelength=1.0, borderaxespad=0.2, labelspacing=0.22)
    panel_label(ax, "(a)", x=-0.25)
    # (b) BEP vs Omega
    ax = axs[1]
    R = lc["results"]
    for s in ("allosteric", "weighted", "coregulated", "coregulated_slow", "cooperative", "cooperative_a1"):
        pts = []
        for k, r in R.items():
            sch, motif, om, meth = k.split("|")
            if sch != s or motif != "AIF":
                continue
            om = int(om)
            if (om <= 30 and meth == "exact") or (om >= 45 and meth == "tauleap"):
                pts.append((om, r["bep"], r["lo"], r["hi"]))
        pts.sort()
        if not pts:
            continue
        oms = np.array([p[0] for p in pts]); b = np.array([p[1] for p in pts])
        lo = np.array([p[2] for p in pts]); hi = np.array([p[3] for p in pts])
        bf, ff = floored(b); lof, _ = floored(lo); hif, _ = floored(hi)
        c = COL[s]; ls = ":" if s == "cooperative_a1" else ("--" if s == "coregulated_slow" else "-")
        ax.errorbar(oms, bf, yerr=[np.maximum(bf - lof, 0), np.maximum(hif - bf, 0)], color=c, lw=0.9, ls=ls,
                    marker="o" if s not in ("cooperative_a1", "coregulated_slow") else "D",
                    ms=3 if s not in ("cooperative_a1", "coregulated_slow") else 2.4,
                    capsize=1.5, elinewidth=0.6, label=LBL[s])
        ax.scatter(oms[ff], bf[ff], s=14, facecolor="white", edgecolor=c, zorder=6, lw=0.8)
    ax.axhline(hc["references"]["static_at_kdopt_n1"]["bep"], color="k", ls=":", lw=0.7)
    ax.text(850, hc["references"]["static_at_kdopt_n1"]["bep"] * 1.5, r"non-adaptive, static channel, $n_\mathrm{H}=1$", fontsize=5.6, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(10, 1000); ax.set_ylim(1e-7, 0.3)
    ax.set_xlabel(r"system size $\Omega$"); ax.set_ylabel("BEP (static channel)")
    ax.legend(loc="upper right", fontsize=5.2, handlelength=1.4, borderaxespad=0.2, labelspacing=0.2)
    panel_label(ax, "(b)", x=-0.25)
    # (c) stability validation
    ax = axs[2]
    lin = zc["linear"]
    x = np.array([r["zeta_over_crit"] for r in lin])
    ax.plot(x, [r["kurt200"] for r in lin], color="#D55E00", marker="o", ms=3, lw=0.9, label=r"kurtosis, $\Omega=200$")
    ax.plot(x, [r["kurt400"] for r in lin], color="#D55E00", marker="s", ms=2.6, lw=0.9, ls="--", mfc="white",
            label=r"kurtosis, $\Omega=400$")
    ax.axhline(0, color="0.7", lw=0.5); ax.axhline(-1.5, color="0.7", lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_xlim(0.04, 2.2); ax.set_ylim(-1.6, 0.6)
    ax.set_xlabel(r"$\zeta/\zeta_\mathrm{crit}$"); ax.set_ylabel(r"excess kurtosis of $\delta y$", color="#D55E00")
    ax.axvline(1.0, color="k", lw=0.8); ax.axvspan(1.0, 2.2, color="0.9", lw=0)
    ax.axvline(zc["ratio_sf1"], color="0.4", lw=0.7, ls="--")
    ax.text(zc["ratio_sf1"] * 1.08, -0.62, r"$S_\mathrm{f}=1$", fontsize=6, color="0.3")
    ax.text(1.05, 0.45, r"$\zeta_\mathrm{crit}$", fontsize=6.5)
    ax2 = ax.twinx()
    ax2.plot(x, [r["varOmega200"] for r in lin], color=COL["allosteric"], marker="o", ms=3, lw=0.9)
    ax2.plot(x, [r["varOmega400"] for r in lin], color=COL["allosteric"], marker="s", ms=2.6, lw=0.9, ls="--", mfc="white")
    ax2.set_yscale("log"); ax2.set_ylim(0.03, 2e3)
    ax2.set_ylabel(r"$\Omega\,\zeta^2\,\mathrm{Var}(\delta y)$", color=COL["allosteric"])
    ax2.tick_params(axis="y", labelsize=7)
    handles = [Line2D([], [], color="k", marker="o", ms=3, lw=0.9, label=r"$\Omega=200$"),
               Line2D([], [], color="k", marker="s", ms=2.6, lw=0.9, ls="--", mfc="white", label=r"$\Omega=400$")]
    ax.legend(handles=handles, loc="upper left", fontsize=6, handlelength=1.6, borderaxespad=0.2)
    panel_label(ax, "(c)", x=-0.25)
    fig.tight_layout(pad=0.3, w_pad=1.6)
    save(fig, "fig6_static_channel")


# =========================================================================== #
#  Fig. 5 -- loop-speed design rule
# =========================================================================== #
def tau_lf(alpha):
    """Tracking time constant gamma_p/b of the non-cooperative loop at the actuator gain alpha."""
    ss = M.steady_state("allosteric", alpha); rho = ss["z1"] / (ss["z1"] + ss["z2"])
    return M.GP / (M.K * rho * M.THETA * M.zeta_star("allosteric", alpha))


def fig5():
    dr = json.load(open(os.path.join(RES, "design_rule.json")))
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    styles = {(5.0, 240.0): ("#0072B2", "o", r"$S=5$, $T_\mathrm{c}=240$"), (5.0, 960.0): ("#009E73", "s", r"$S=5$, $T_\mathrm{c}=960$"),
              (20.0, 240.0): ("#E69F00", "^", r"$S=20$, $T_\mathrm{c}=240$")}
    for cse, sur, rule in zip(dr["cases"], dr["surrogate"], dr["rule"]):
        key = (cse["S"], cse["T"])
        if key not in styles:
            continue
        c, mk, lab = styles[key]
        rows = sorted(cse["rows"], key=lambda r: r["alpha"])
        x = np.array([tau_lf(r["alpha"]) for r in rows]); y = np.array([r["adaptive"] for r in rows])
        lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows])
        ax.errorbar(x, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], color=c, marker=mk, ms=3.2, lw=0, elinewidth=0.6,
                    capsize=1.5, label=lab)
        sa = np.array(sur["alpha"]); sx = np.array([tau_lf(a) for a in sa]); sb = np.array(sur["bep"])
        keep = sa <= 1.5                                # the tested range of actuator gains
        ax.plot(sx[keep], sb[keep], color=c, lw=0.9, ls="-", alpha=0.85)
        ax.axvline(rule["tau_opt"], color=c, lw=0.7, ls=":")
        tau_lf_min = min(tau_lf(a) for a in np.linspace(0.2, 5.0, 2401))    # smallest tracking time of the alpha-only family
        if rule["tau_opt"] < tau_lf_min:                                    # unconstrained candidate infeasible: show the projection
            ax.axvline(tau_lf_min, color=c, lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"tracking time constant $\tau_\mathrm{LF}=\gamma_\mathrm{p}/b$"); ax.set_ylabel("BEP")
    ax.set_xlim(5, 40)
    ax.set_xticks([5, 7, 10, 15, 20, 30, 40]); ax.set_xticklabels(["5", "7", "10", "15", "20", "30", "40"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="upper left", ncol=1, handlelength=1.6, borderaxespad=0.4, fontsize=6.2, framealpha=0.9)
    fig.tight_layout(pad=0.3)
    save(fig, "fig5_design_rule")


if __name__ == "__main__":
    for name, fn, needs in (("fig2", fig2, ["tracking.npz", "tracking.json"]),
                            ("fig3", fig3, ["headline.json"]),
                            ("fig5", fig5, ["design_rule.json"]),
                            ("fig6", fig6, ["bep_hc.json", "bep_lc.json", "zeta_crit.json"])):
        missing = [f for f in needs if not os.path.exists(os.path.join(RES, f))]
        if missing:
            print(f"[skip] {name}: missing {', '.join(missing)} (run the corresponding run_*.py first)")
            continue
        fn()
