#!/usr/bin/env python3
"""
make_fig1_schematic.py -- Fig. 1, (a) illustration of the considered scenario, i.e., the time-varying diffusive
channel with enzymatic degradation, an interfering source, and ISI, and the receiver cell with its receptors, the
intracellular AIF controller, the modulator, and the detector, and (b) the block diagram of the adaptive receiver.

Writes the standalone TikZ source ../manuscript/figures/fig1_schematic.tex and compiles it with pdflatex into
../manuscript/figures/fig1_schematic.pdf, which requires a TeX distribution with TikZ and the newtx fonts.  The inset
traces are illustrative, with c0 = 1 and c1 = 4 (Table I), a periodic attenuation with S = 5, and a first-order lag of
the affinity, and do not use simulation data.
"""
import math, os, random, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "manuscript", "figures")
os.makedirs(OUT, exist_ok=True)
rnd = random.Random(7)

def f(v):
    return f"{v:.3f}"

# ------------------------------------------------------------------ panel (a) geometry
CX, CY, A, B = 14.15, 2.65, 3.75, 2.50        # receiver cell (outer membrane ellipse)
AM, BM = A - 0.06, B - 0.06                   # membrane mid-line
TXC, TXA, TXB = (1.05, 1.85), 0.78, 1.0       # transmitter
BAND = (1.85, 0.52, 10.75, 3.22)              # channel band (x0, y0, x1, y1)

def on_membrane(phi_deg):
    p = math.radians(phi_deg)
    x, y = CX + AM * math.cos(p), CY + BM * math.sin(p)
    nrm = math.degrees(math.atan2(A * math.sin(p), B * math.cos(p)))
    return x, y, nrm

def inside_cell(x, y, margin=0.0):
    return ((x - CX) / (A + margin)) ** 2 + ((y - CY) / (B + margin)) ** 2 < 1.0

# channel objects: enzymes, interfering source, ISI inset, label boxes (kept free of molecules)
ENZ = [(5.0, 2.55), (8.55, 2.62)]
SRC = (4.25, 0.84)                              # interfering source (background interference)
ISIBOX = (7.72, 0.56, 10.22, 1.62)              # ISI inset frame
LIGDOT = (2.98, 1.5)                            # ligand pointed at by the "ligands" label
KEEP_OUT = [(1.95, 2.8, 4.35, 3.22),            # "free diffusion" label
            (5.18, 2.42, 7.3, 2.7),             # "enzymatic degradation" label
            (4.55, 0.55, 7.45, 1.12),           # interfering-source label
            (1.95, 0.55, 2.75, 1.05),           # "ligands" label
            (ISIBOX[0] - 0.06, ISIBOX[1], ISIBOX[2] + 0.06, ISIBOX[3] + 0.06)]
def free(x, y, pts, dmin):
    if inside_cell(x, y, 0.28) or any(a <= x <= c and b <= y <= d for a, b, c, d in KEEP_OUT):
        return False
    if any((x - e[0]) ** 2 + (y - e[1]) ** 2 < 0.3 ** 2 for e in ENZ) or (x - SRC[0]) ** 2 + (y - SRC[1]) ** 2 < 0.42 ** 2:
        return False
    return all((x - a) ** 2 + (y - b) ** 2 >= dmin ** 2 for a, b in pts)

# signal ligands from the Tx: density decays away from the transmitter (diffusion)
lig = [LIGDOT]
while len(lig) < 104:
    x = TXC[0] + TXA + 0.05 + rnd.expovariate(1 / 2.6)
    if x > 10.55:
        continue
    y = rnd.gauss(2.08, 0.22 + 0.14 * (x - 1.8))
    if not (1.36 < y < BAND[3] - 0.08) or not free(x, y, lig, 0.13):
        continue
    lig.append((x, y))
# background-interference molecules emitted by the interfering source (lighter), some reaching the receiver
bg = []
while len(bg) < 17:
    r, ang = 0.42 + rnd.expovariate(1 / 0.75), math.radians(rnd.uniform(8, 172))
    x, y = SRC[0] + 1.5 * r * math.cos(ang), SRC[1] + r * math.sin(ang)
    if BAND[1] + 0.08 < y < BAND[3] - 0.08 and BAND[0] + 0.1 < x < 10.5 and free(x, y, lig + bg, 0.15):
        bg.append((x, y))
while len(bg) < 27:
    x, y = rnd.uniform(6.0, 10.4), rnd.uniform(1.3, 3.05)
    if free(x, y, lig + bg, 0.17):
        bg.append((x, y))

# ISI inset: received pulses of successive symbols (free diffusion, T_sym = 3 t_peak) and their sum
ISI_BITS, ISI_N0 = [1, 1, 0, 1, 1], 0.25
def hpulse(t, tp=1 / 3):
    return 0.0 if t <= 0 else (tp / t) ** 1.5 * math.exp(1.5 * (1 - tp / t))
IPX0, IPX1, IPY0, IPY1 = ISIBOX[0] + 0.14, ISIBOX[2] - 0.32, ISIBOX[1] + 0.12, ISIBOX[3] - 0.3
TMAX = 5.3
ts = [i * 0.02 for i in range(int(TMAX / 0.02) + 1)]
pulses = [[(1.0 if b else ISI_N0) * hpulse(t - j) for t in ts] for j, b in enumerate(ISI_BITS)]
total = [sum(p[i] for p in pulses) for i in range(len(ts))]
HMAX = max(total) * 1.05
def IX(t): return IPX0 + (IPX1 - IPX0) * t / TMAX
def IY(v): return IPY0 + (IPY1 - IPY0) * v / HMAX

# receptors on the left arc of the membrane
REC_PHI = [152, 163, 174, 185, 196, 207, 218]
BOUND = {163, 185, 196, 218}

# ------------------------------------------------------------------ inset: received signal and affinity
IX0, IY0, IX1, IY1 = 2.35, 3.40, 9.80, 5.30      # inset frame
PX0, PX1, PY0, PY1 = 3.00, 8.45, 3.80, 5.14      # plot area
NSYM, S, C0, C1 = 44, 5.0, 1.0, 4.0
CMAX = 4.3
def gatt(t):   # t in symbols, one full period over the window
    return math.exp(-0.5 * math.log(S) * (1 - math.cos(2 * math.pi * t / NSYM)))
def X(t): return PX0 + (PX1 - PX0) * t / NSYM
def Y(c): return PY0 + (PY1 - PY0) * c / CMAX
bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1,
        1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
sig = []
for i, b in enumerate(bits):
    c = (C1 if b else C0) * gatt(i + 0.5)
    sig += [(X(i), Y(c)), (X(i + 1), Y(c))]
env1 = [(X(t / 4), Y(C1 * gatt(t / 4))) for t in range(0, 4 * NSYM + 1)]
env0 = [(X(t / 4), Y(C0 * gatt(t / 4))) for t in range(0, 4 * NSYM + 1)]
# affinity: first-order tracking of ln(sqrt(c0 c1) g) with a lag of 3 symbols (illustrative)
lk, kd, dt, tau = math.log(2.0), [], 0.05, 3.0
t = 0.0
while t <= NSYM + 1e-9:
    target = math.log(2.0 * gatt(t))
    lk += dt / tau * (target - lk)
    if abs(round(t / 0.25) * 0.25 - t) < 1e-9:
        kd.append((X(t), Y(math.exp(lk))))
    t += dt

def coords(pts):
    return " ".join(f"({f(a)},{f(b)})" for a, b in pts)

# ------------------------------------------------------------------ TikZ source
L = []
P = L.append
P(r"""\documentclass[border=1.5pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc,decorations.pathmorphing,positioning,fit,backgrounds}
\definecolor{lig}{HTML}{0072B2}\definecolor{ligL}{HTML}{A9CBE8}\definecolor{chan}{HTML}{E6F0F9}
\definecolor{rec}{HTML}{009E73}\definecolor{recL}{HTML}{D3EEE5}
\definecolor{ctl}{HTML}{D55E00}\definecolor{ctlL}{HTML}{FBE3D4}
\definecolor{mod}{HTML}{7B3294}\definecolor{modL}{HTML}{EDE0F3}
\definecolor{memb}{HTML}{E2B77E}\definecolor{membD}{HTML}{A8783F}\definecolor{cellF}{HTML}{FFFAF2}
\definecolor{txF}{HTML}{EEEEEE}\definecolor{gr}{HTML}{6B6B6B}\definecolor{enz}{HTML}{8F8F8F}
\definecolor{srcF}{HTML}{E6D8C4}\definecolor{srcD}{HTML}{9A7F5E}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,font=\fontsize{7}{8.2}\selectfont,>={Stealth[length=3.4pt,width=2.6pt]},
  line cap=round,line join=round,
  spc/.style={circle,draw=ctl,fill=ctlL,line width=0.5pt,inner sep=0.6pt,minimum size=0.44cm},
  mspc/.style={circle,draw=mod,fill=modL,line width=0.5pt,inner sep=0.6pt,minimum size=0.44cm},
  blk/.style={draw,rounded corners=2pt,line width=0.5pt,align=center,inner sep=2.2pt,minimum height=0.82cm},
  lab/.style={font=\fontsize{6.5}{7.6}\selectfont,text=gr},
  clab/.style={lab,fill=white,fill opacity=0.85,text opacity=1,rounded corners=1.5pt,inner sep=1.2pt}]
""")
# ---------------- panel (a)
P(r"\node[anchor=north west,font=\fontsize{8}{9}\selectfont\bfseries] at (-0.15,5.42) {(a)};")
x0, y0, x1, y1 = BAND
P(rf"\shade[left color=chan,right color=chan!35,rounded corners=5pt] ({x0},{y0}) rectangle ({x1},{y1});")
# transmitter
P(rf"\draw[gr,fill=txF,line width=0.6pt] ({TXC[0]},{TXC[1]}) ellipse ({TXA} and {TXB});")
P(rf"\node[font=\fontsize{{8}}{{9}}\selectfont\bfseries] at ({TXC[0]},{TXC[1] + 0.18}) {{Tx}};")
P(rf"\node[lab,align=center] at ({TXC[0]},{TXC[1] - 0.28}) {{BCSK\\$N_{{\mathrm{{L}}|s}}$}};")
P(rf"\node[lab] at ({TXC[0]},{TXC[1] + TXB + 0.2}) {{bits: $1\,0\,1\,1\,0\cdots$}};")
# molecules and channel objects
for x, y in bg:
    P(rf"\fill[ligL] ({f(x)},{f(y)}) circle (0.043);")
for x, y in lig:
    P(rf"\fill[lig] ({f(x)},{f(y)}) circle (0.048);")
for i, (x, y) in enumerate(ENZ):
    P(rf"\fill[enz] ({x},{y}) -- ++(35:0.15) arc (35:325:0.15) -- cycle;")
    P(rf"\fill[lig] ({f(x + 0.17)},{y}) circle (0.048);")
P(rf"\node[clab,anchor=west] at ({ENZ[0][0] + 0.22},{ENZ[0][1]}) {{enzymatic degradation}};")
P(r"\node[clab,anchor=south west] at (2.05,2.9) {free diffusion, $h(t,d)$};")
# ligands: plain label with an arrow (the signal carrier, not an impairment)
P(rf"\node[anchor=west,text=lig,inner sep=1pt,font=\fontsize{{6.5}}{{7.6}}\selectfont] (liglab) at (1.98,0.8) {{ligands}};")
P(rf"\draw[->,lig,line width=0.45pt,shorten >=1.6pt] (liglab.north east) -- ({LIGDOT[0]},{LIGDOT[1]});")
# interfering source: a different kind of emitter
sx, sy = SRC
blob = [(sx - 0.34, sy - 0.02), (sx - 0.22, sy + 0.2), (sx + 0.02, sy + 0.26), (sx + 0.27, sy + 0.17),
        (sx + 0.35, sy - 0.04), (sx + 0.2, sy - 0.22), (sx - 0.08, sy - 0.24), (sx - 0.28, sy - 0.17)]
P(r"\draw[srcD,fill=srcF,line width=0.55pt] plot[smooth cycle,tension=0.75] coordinates {" + coords(blob) + "};")
for dx, dy in ((-0.12, 0.05), (0.1, 0.09), (0.02, -0.1)):
    P(rf"\fill[ligL] ({f(sx + dx)},{f(sy + dy)}) circle (0.043);")
P(rf"\node[lab,anchor=west,align=left] at ({sx + 0.42},{sy}) {{interfering source\\(background interference)}};")
# ISI inset: accumulation of the tails of previous pulses
a, b, c, d = ISIBOX
P(rf"\draw[gr!70,fill=white,line width=0.4pt,rounded corners=2pt] ({a},{b}) rectangle ({c},{d});")
P(rf"\node[lab,anchor=north west,inner sep=1.5pt] at ({a},{d}) {{ISI: pulse tails accumulate}};")
P(rf"\draw[->,line width=0.35pt] ({f(IPX0)},{f(IPY0)}) -- ({f(IPX1 + 0.12)},{f(IPY0)}) node[right,inner sep=0.5pt,font=\fontsize{{6}}{{7}}\selectfont] {{$t$}};")
for pl in pulses:
    P(r"\draw[lig!45,line width=0.4pt] plot coordinates {" + coords([(IX(t), IY(v)) for t, v in zip(ts[::2], pl[::2])]) + "};")
P(r"\draw[lig,line width=0.75pt] plot coordinates {" + coords([(IX(t), IY(v)) for t, v in zip(ts[::2], total[::2])]) + "};")
for j in range(1, len(ISI_BITS)):
    P(rf"\draw[gr!60,line width=0.25pt,densely dotted] ({f(IX(j))},{f(IPY0)}) -- ({f(IX(j))},{f(IPY1)});")
# distance arrow
P(r"\draw[<->,gr,line width=0.45pt] (1.9,0.2) -- (10.45,0.2) node[pos=0.45,fill=white,inner sep=1pt,text=gr,font=\fontsize{6.5}{7.6}\selectfont] {time-varying distance $d(t)$};")
# inset
P(rf"\draw[gr!70,fill=white,line width=0.4pt,rounded corners=2pt] ({IX0},{IY0}) rectangle ({IX1},{IY1});")
P(rf"\draw[->,line width=0.4pt] ({PX0},{PY0}) -- ({PX1 + 0.18},{PY0}) node[right,inner sep=1pt] {{$t$}};")
P(rf"\draw[->,line width=0.4pt] ({PX0},{PY0}) -- ({PX0},{PY1 + 0.14});")
P(rf"\node[rotate=90,anchor=south,align=center,inner sep=1pt] at ({PX0 - 0.04},{(PY0 + PY1) / 2}) {{received\\concentration}};")
P(rf"\draw[gr,densely dashed,line width=0.35pt] plot coordinates {{{coords(env1)}}};")
P(rf"\draw[gr,densely dashed,line width=0.35pt] plot coordinates {{{coords(env0)}}};")
P(rf"\draw[lig,line width=0.55pt] plot coordinates {{{coords(sig)}}};")
P(rf"\draw[mod,line width=0.9pt] plot coordinates {{{coords(kd)}}};")
P(rf"\node[anchor=west,inner sep=1pt] at ({f(PX1 + 0.12)},{f(Y(C1) - 0.02)}) {{$c_1\,g(t)$}};")
P(rf"\node[anchor=west,inner sep=1pt] at ({f(PX1 + 0.12)},{f(Y(C0) - 0.03)}) {{$c_0\,g(t)$}};")
mid = (PX0 + PX1) / 2
P(rf"\draw[<->,gr,line width=0.35pt] ({PX0},{PY0 - 0.2}) -- ({PX1},{PY0 - 0.2}) node[midway,fill=white,inner sep=0.8pt,text=gr] {{period $T_\mathrm{{c}}$}};")
P(rf"\node[lab,anchor=north] at ({f(mid)},{PY1 + 0.02}) {{slow common attenuation $g(t)$}};")
P(rf"\node[anchor=north,text=mod,inner sep=1pt] at ({f(mid)},{PY1 - 0.26}) {{$K_\mathrm{{D}}(t)\approx\sqrt{{c_0c_1}}\,g(t)$}};")
# receiver cell
P(rf"\fill[cellF] ({CX},{CY}) ellipse ({A} and {B});")
P(rf"\draw[memb,line width=3.1pt] ({CX},{CY}) ellipse ({AM} and {BM});")
P(rf"\draw[membD,line width=0.35pt] ({CX},{CY}) ellipse ({A} and {B});")
P(rf"\draw[membD,line width=0.35pt] ({CX},{CY}) ellipse ({A - 0.12} and {B - 0.12});")
for phi in REC_PHI:
    x, y, ang = on_membrane(phi)
    P(rf"\begin{{scope}}[shift={{({f(x)},{f(y)})}},rotate={f(ang)}]")
    P(r"\draw[rec,line width=1.5pt] (-0.36,0) -- (-0.1,0);")
    P(r"\fill[rec] (-0.13,-0.05) rectangle (0.1,0.05);")
    P(r"\draw[rec,line width=1.3pt] (0.08,0.02) -- (0.3,0.13); \draw[rec,line width=1.3pt] (0.08,-0.02) -- (0.3,-0.13);")
    P(r"\fill[mod] (-0.29,0.075) circle (0.034); \fill[mod] (-0.21,-0.075) circle (0.034);")
    if phi in BOUND:
        P(r"\fill[lig] (0.25,0) circle (0.052);")
    P(r"\end{scope}")
# intracellular network
P(r"""\node[spc] (Z1) at (13.85,3.25) {$Z_1$};
\node[spc] (Z2) at (12.2,2.2) {$Z_2$};
\node[mspc] (Y) at (13.62,4.28) {$Y$};
\node (e1) at (15.25,3.25) {$\varnothing$};
\coordinate (J) at (13.05,2.72);
\node (e2) at (13.62,2.3) {$\varnothing$};
\node (e3) at (14.95,4.28) {$\varnothing$};
\draw[->,ctl,line width=0.55pt] (e1) -- (Z1) node[midway,above,inner sep=1pt] {$\mu$};
\draw[ctl,line width=0.55pt] (Z1) -- (J) -- (Z2);
\fill[ctl] (J) circle (0.045);
\draw[->,ctl,line width=0.55pt] (J) -- (e2) node[midway,below left,inner sep=0.5pt] {$\eta$};
\draw[->,ctl,line width=0.55pt,decorate,decoration={snake,amplitude=0.35mm,segment length=1.5mm,post length=1.2mm}] (11.12,2.38) -- (Z2) node[midway,above,inner sep=1.5pt] {$\theta\,\theta_\mathrm{B}$};
\draw[->,mod,line width=0.55pt,densely dashed] (Z1) -- (Y) node[midway,right,inner sep=1pt] {$k$};
\draw[->,mod,line width=0.55pt] (Y) -- (e3) node[midway,above,inner sep=1pt] {$\gamma_\mathrm{p}$};
\draw[->,mod,line width=0.65pt] (Y.west) .. controls (12.6,4.45) and (11.75,4.15) .. (11.12,3.62) node[pos=0.42,above,inner sep=1.5pt,text=mod] {tunes $K_\mathrm{D}(y)$};
\node[draw=ctl,dashed,line width=0.45pt,rounded corners=3pt,fit=(Z1)(Z2)(e1)(e2),inner sep=3pt] (ctlbox) {};
\node[anchor=north west,text=ctl,inner sep=1.5pt,font=\fontsize{6.5}{7.6}\selectfont] at (ctlbox.north west) {AIF controller};
\node[draw=gr,fill=txF,rounded corners=2pt,align=center,inner sep=2pt,line width=0.45pt] (det) at (13.35,0.98) {detector: $n_\mathrm{B}\gtrless N_\mathrm{R}/2$};
\draw[->,line width=0.55pt] (11.45,1.42) -- (det.west) node[pos=0.45,below left,inner sep=0.5pt,font=\fontsize{6.5}{7.6}\selectfont] {$n_\mathrm{B}$};
\draw[->,line width=0.6pt] (det.east) -- (17.75,0.98) node[right,inner sep=1pt] {$\hat b$};
\node[lab,anchor=south] at (15.75,1.0) {once per symbol};
\node[anchor=east,inner sep=1pt,font=\fontsize{7.5}{8.5}\selectfont\bfseries] at (17.45,3.2) {Rx};
\node[lab,anchor=east,align=right] at (17.45,2.72) {receiver\\cell};
""")
P(r"\node[anchor=north west,text=rec,inner sep=1pt,align=left,font=\fontsize{6.5}{7.6}\selectfont] (reclab) at (9.97,5.3) {ligand receptors\\$N_\mathrm{R}$, $K_\mathrm{D}(y)$, $n_\mathrm{H}$};")
P(r"\draw[rec,line width=0.4pt] (10.3,4.6) -- (10.58,4.1);")
# ---------------- panel (b)
P(r"\node[anchor=north west,font=\fontsize{8}{9}\selectfont\bfseries] at (-0.15,-0.28) {(b)};")
P(r"""\begin{scope}[yshift=0cm]
\node[blk,draw=gr,fill=txF,minimum width=1.9cm] (bTx) at (1.35,-1.1) {transmitter\\(BCSK)};
\node[blk,draw=lig,fill=chan,minimum width=2.7cm] (bCh) at (4.75,-1.1) {time-varying channel\\$h(t,d)$, $g(t)$};
\node[blk,draw=rec,fill=recL,minimum width=3.1cm] (bRx) at (8.95,-1.1) {receptor array, $N_\mathrm{R}$\\$\mathrm{U}\rightleftharpoons\mathrm{B}$, \ $K_\mathrm{D}(y)$, \ $n_\mathrm{H}$};
\node[circle,draw,line width=0.5pt,inner sep=0.8pt] (bTh) at (11.45,-1.1) {$\theta_\mathrm{B}$};
\node[blk,draw=gr,fill=txF,minimum width=3.0cm] (bDet) at (14.25,-1.1) {fixed-threshold detector\\$n_\mathrm{B}\gtrless N_\mathrm{R}/2$};
\draw[->,line width=0.55pt] (bTx) -- (bCh) node[midway,above,inner sep=1.5pt] {$N_{\mathrm{L}|s}$};
\draw[->,line width=0.55pt] (bCh) -- (bRx) node[midway,above,inner sep=1.5pt] {$c_\mathrm{L}(t)$};
\draw[->,line width=0.55pt] (bRx) -- (bTh);
\draw[->,line width=0.55pt] (bTh) -- (bDet);
\draw[->,line width=0.55pt] (bDet.east) -- ++(0.75,0) node[right,inner sep=1pt] {$\hat b$};
\node[lab,anchor=south] at (bDet.north) {per symbol (fast)};
\node[blk,draw=ctl,fill=ctlL,minimum width=5.0cm] (bCtl) at (13.45,-2.66) {antithetic integral feedback controller\\[1pt]
  $\varnothing\xrightarrow{\,\mu\,}Z_1$, \ $\varnothing\xrightarrow{\,\theta\theta_\mathrm{B}(t)\,}Z_2$, \ $Z_1+Z_2\xrightarrow{\,\eta\,}\varnothing$\\
  $Z_1\xrightarrow{\,k\,}Z_1+Y$, \ $Y\xrightarrow{\,\gamma_\mathrm{p}\,}\varnothing$};
\node[blk,draw=mod,fill=modL,minimum width=3.1cm] (bAct) at (8.95,-2.66) {actuation law\\$K_\mathrm{D}(y)=K_\mathrm{base}\,e^{-\alpha y}$};
\draw[->,ctl,line width=0.6pt] (bTh.south) -- (bTh.south |- bCtl.north) node[midway,right,inner sep=2pt,align=left,font=\fontsize{6.5}{7.6}\selectfont,yshift=-1pt] {$\theta_\mathrm{B}(t)$, instantaneous occupancy\\(averaged by the slow loop)};
\draw[->,mod,line width=0.6pt] (bCtl.west) -- (bAct.east) node[midway,above,inner sep=1.5pt] {$y$};
\draw[->,mod,line width=0.6pt] (bAct.north) -- (bRx.south) node[midway,left,inner sep=2pt,font=\fontsize{6.5}{7.6}\selectfont] {affinity actuation};
\node[lab,align=center] at (4.75,-2.66) {integral action: $\bar{\theta}_\mathrm{B}\to\mu/\theta=\tfrac12$\\$\Rightarrow\;K_\mathrm{D}\to\sqrt{c_0c_1}\,g(t)$};
\node[lab,align=center] at (1.35,-2.66) {timescales:\\$T_\mathrm{sym}\ll\tau<T_\mathrm{c}$};
\end{scope}
""")
P(r"""\end{tikzpicture}
\end{document}
""")

tex = os.path.join(OUT, "fig1_schematic.tex")
open(tex, "w").write("\n".join(L))
r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig1_schematic.tex"],
                   cwd=OUT, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-3000:]); raise SystemExit("pdflatex failed")
for ext in (".aux", ".log"):
    aux = os.path.join(OUT, "fig1_schematic" + ext)
    if os.path.exists(aux):
        os.remove(aux)
print("wrote", os.path.normpath(tex), "and", os.path.normpath(os.path.join(OUT, "fig1_schematic.pdf")))
