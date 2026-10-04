"""
make_fig4_cir_sweep.py -- Fig. 4, the CIR-based model as a function of the normalized symbol duration T_sym / t_peak.

    (a) BEP of the adaptive, best non-adaptive, and genie-aided receivers for S = 5 at T_c = 240 (filled markers) and
        960 (open markers).
    (b) BEP reduction factor, with the values of the piecewise-constant signal model at S = 5 (results/headline.json)
        as dashed lines.
Reads results/waveform_isi.json and results/headline.json.  Output: ../manuscript/figures/fig4_cir_sweep.{pdf,png}, or
the directory given as the first argument.
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "manuscript", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300,
    "pdf.fonttype": 42, "axes.unicode_minus": False,
})
BLUE, BLACK, GRAY = "#0072B2", "#000000", "#7f7f7f"


def panel_label(ax, s, x=-0.2, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


d = json.load(open(os.path.join(RES, "waveform_isi.json")))
ratios = sorted(int(k) for k in d["ratios"])
R = [d["ratios"][str(r)] for r in ratios]
hl = json.load(open(os.path.join(RES, "headline.json")))["rows"]
ref = {int(r["T"]): r["gain"] for r in hl if r["label"] == "allosteric" and r["S"] == 5.0 and int(r["T"]) in (240, 960)}

fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.5, 3.4))
# (a) BEPs
for T, ls, mfc, mk in ((240, "-", None, "o"), (960, "--", "white", "s")):
    sw = [r["swing"][str(T)] for r in R]
    ax.plot(ratios, [s["design"]["adaptive"] for s in sw], color=BLUE, ls=ls, marker=mk, ms=3.4, mfc=mfc or BLUE,
            label=rf"adaptive, $T_\mathrm{{c}}={T}$")
    ax.plot(ratios, [s["static_best"] for s in sw], color=BLACK, ls=ls, marker=mk, ms=3.2, mfc=mfc or BLACK,
            label=rf"best non-adaptive, $T_\mathrm{{c}}={T}$")
ax.plot(ratios, [r["swing"]["240"]["oracle"] for r in R], color=GRAY, ls=":", marker="^", ms=3.2, label="genie-aided")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(ratios); ax.set_xticklabels([str(r) for r in ratios]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlim(2.7, 18); ax.set_ylim(3e-6, 1)
ax.set_ylabel("BEP")
ax.legend(loc="lower left", ncol=1, handlelength=1.8, borderaxespad=0.2, labelspacing=0.25)
panel_label(ax, "(a)")
# (b) reduction factor
for T, ls, mfc, mk in ((240, "-", None, "o"), (960, "--", "white", "s")):
    sw = [r["swing"][str(T)] for r in R]
    bx.plot(ratios, [s["design"]["gain"] for s in sw], color=BLUE, ls=ls, marker=mk, ms=3.4, mfc=mfc or BLUE,
            label=rf"CIR-based model, $T_\mathrm{{c}}={T}$")
for T, y in ref.items():
    bx.axhline(y, color=GRAY, ls="--", lw=0.7)
    bx.text(2.85, y * 1.2, rf"piecewise-constant, $T_\mathrm{{c}}={T}$", fontsize=6, color="0.35", ha="left", va="bottom")
bx.axhline(10, color="0.6", ls=":", lw=0.6)
bx.text(17.6, 10 * 1.15, "tenfold", fontsize=6, color="0.4", ha="right", va="bottom")
bx.set_xscale("log"); bx.set_yscale("log")
bx.set_xticks(ratios); bx.set_xticklabels([str(r) for r in ratios]); bx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
bx.set_xlim(2.7, 18); bx.set_ylim(1, 1000)
bx.set_xlabel(r"$T_\mathrm{sym}/t_\mathrm{peak}$"); bx.set_ylabel("BEP reduction factor")
bx.legend(loc="lower right", handlelength=1.8, borderaxespad=0.3, labelspacing=0.25)
panel_label(bx, "(b)")
fig.tight_layout(pad=0.3, h_pad=0.9)
fig.savefig(os.path.join(OUT, "fig4_cir_sweep.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(OUT, "fig4_cir_sweep.png"), bbox_inches="tight", pad_inches=0.02)
print("wrote", os.path.normpath(os.path.join(OUT, "fig4_cir_sweep.pdf")),
      "| reduction factors of the piecewise-constant signal model:", ref)
