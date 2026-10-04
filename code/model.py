"""
model.py -- analytical model of the adaptive receiver.

This module collects the analytical ingredients of the paper:
* the model parameters of Table I;
* the receptor maps of the compared receivers (Sections II-B, II-D, and III-E);
* the actuation laws of Section II-D, i.e., the exponential law K_D(y) = K_base exp(-alpha y) and the bounded law
  K_D(y) = K_base (1 + y/K_Y)/(1 + chi y/K_Y) with the local actuator gain beta(y) = -d ln K_D/dy;
* the steady state of the antithetic integral feedback (AIF) loop, including leaky integration (Sections III-A, III-F);
* the receptor gain zeta*, the reduced-loop quantities (fraction rho, crossover frequency, loop time constant), and
  the modulator variance including the self-noise of the random symbol stream (Section III-B);
* the exact local stability boundary zeta_crit of the full loop (Section III-D);
* the Gaussian approximation of the BEP, the loop-speed design rule (Section III-C), and the design-point optimizer
  (Table II).

Conventions
-----------
* Concentrations are given for the system size Omega = 1, and molecule counts are concentration times Omega.
* Time is nondimensional, and one model time unit corresponds to 25 s in Section V.
* y denotes the modulator concentration.  The actuation law is selected with set_actuator("exp" | "atcm").
* The controller senses the instantaneous occupancy theta_B(c(t); y).  For symbols much shorter than the loop time
  constant, its slow response is governed by the symbol-averaged occupancy g(y) = [theta_B(c0; y) + theta_B(c1; y)]/2,
  whose slope at the equilibrium is the receptor gain zeta* = n_H alpha theta0 theta1, with theta0 = 1/(1 + r^(n_H/2))
  and the level ratio r = c1/c0.
* The random symbol stream adds the self-noise intensity D_bits = theta^2 (Delta theta)^2 T_sym / 4, which does not
  decrease with Omega.

Receivers and their labels in the code and in the result files
---------------------------------------------------------------
  allosteric    non-cooperative receiver, n_H = 1
  cooperative   cooperative receiver, fixed n_H = N_COOP
  coregulated   co-regulated receiver, n_H(y) = N_BASE + KAPPA y (the matched design uses a reduced actuation rate)
  weighted      mixed array, with a fraction w of tunable receptors and the rest fixed at K_FIX
Controller motifs: "AIF", used in the paper, and "CAIF", a cascaded variant with a second sequestration channel that
is included for completeness but not reported in the paper.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.optimize import brentq, fsolve, minimize_scalar
from scipy.linalg import solve_continuous_lyapunov
from scipy.stats import norm

# --------------------------------------------------------------------------- #
#  Table I -- controller and receptor parameters (nondimensional calibration)
# --------------------------------------------------------------------------- #
MU, THETA, K, ETA, GP = 0.5, 1.0, 0.5, 1.0, 0.5      # AIF rates
GC_DEFAULT = 0.0                                     # ideal integrator
EPS, ETA1 = 0.5, 1.0                                 # cascaded variant (CAIF, not reported): reference split and second sequestration
B_CAL, M0 = 1.0, 0.0                                 # receptor calibration
CL0_NOM, CL1_NOM = 1.0, 4.0                          # received levels c0, c1 (Table I)
KD_OPT = float(np.sqrt(CL0_NOM * CL1_NOM))           # optimal affinity sqrt(c0 c1) = 2
KBASE = KD_OPT * float(np.exp(B_CAL))                # K_base = 2e, such that y* = B_CAL / alpha
N_COOP = 2.5                                         # cooperative receiver n_H
N_BASE, KAPPA = 1.5, 1.0                             # co-regulated receiver: n_H(y) = N_BASE + KAPPA y
W_WEIGHT = 0.6                                       # mixed array: tunable fraction w
K_FIX = KD_OPT                                       # mixed array: K_D of the fixed receptors
OMEGA_HC = 200.0                                     # system size Omega (high copy numbers)
N_RECEPTORS = 200                                    # receptor array size
T_SYM = 0.5                                          # symbol duration (model time)
T_C = 80.0                                           # design coherence time
WC_MIN = 2.0 * np.pi / T_C
ALPHA_RANGE = (0.10, 6.4)                            # search range of the actuator gain
NH_CAP = 2.5                                         # largest Hill coefficient at the equilibrium (co-regulated receiver)
TIME_UNIT_S = 25.0                                   # Section V mapping: 1 model time unit = 25 s (T_sym = 3 t_peak at d = 50 um, D = 100 um^2/s)

SCHEMES = ("allosteric", "cooperative", "coregulated", "weighted")
MOTIFS = ("AIF", "CAIF")
K_COREG_SLOW = 0.128634428                           # actuation rate of the bandwidth-matched co-regulated design
RATES_DEFAULT = dict(MU=0.5, THETA=1.0, K=0.5, ETA=1.0, GP=0.5)


def set_rates(**kw):
    """Override controller rates module-wide (e.g. set_rates(K=K_COREG_SLOW)); set_rates() restores Table I."""
    vals = dict(RATES_DEFAULT); vals.update(kw)
    globals().update(vals)

Q = norm.sf


def stable_seed(*parts, base: int = 0) -> int:
    key = "|".join(str(p) for p in parts).encode()
    return (base + int(hashlib.md5(key).hexdigest()[:8], 16)) % (2 ** 31)


# --------------------------------------------------------------------------- #
#  Actuator law
# --------------------------------------------------------------------------- #
ACT = dict(law="exp", kbase=KBASE, ky=None, chi=None)


def set_actuator(law="exp", kbase=KBASE, ky=None, chi=None):
    """Select the actuation law used by kd_of_y (module-wide): "exp" for the exponential law and "atcm" for the
    bounded law of an allosteric two-conformation model, K_D(y) = kbase (1 + y/ky) / (1 + chi y/ky)."""
    ACT.update(law=law, kbase=float(kbase), ky=None if ky is None else float(ky),
               chi=None if chi is None else float(chi))


def kd_of_y(y, alpha):
    """Dissociation constant K_D(y) under the selected actuation law."""
    y = np.asarray(y, float)
    if ACT["law"] == "exp":
        return ACT["kbase"] * np.exp(-alpha * (y - M0))
    kb, ky, chi = ACT["kbase"], ACT["ky"], ACT["chi"]
    return kb * (1.0 + y / ky) / (1.0 + chi * y / ky)


def beta_local(y, alpha):
    """Local actuator gain beta(y) = -d ln K_D/dy (equals alpha for the exponential law)."""
    y = np.asarray(y, float)
    if ACT["law"] == "exp":
        return alpha * np.ones_like(y)
    ky, chi = ACT["ky"], ACT["chi"]
    return (chi - 1.0) * ky / ((ky + y) * (ky + chi * y))


def atcm_calibrate(y_star, beta_star, chi, kd_star=KD_OPT):
    """Parameters (kbase, ky) of the bounded law with K_D(y*) = kd_star and local
    actuator gain beta_star at y*.  Requires beta_star y* < (sqrt(chi)-1)/(sqrt(chi)+1).
    Returns the root with the larger ky (closer to the exponential regime)."""
    bound = (np.sqrt(chi) - 1.0) / (np.sqrt(chi) + 1.0)
    if beta_star * y_star >= bound:
        raise ValueError(f"beta* y* = {beta_star * y_star:.3f} exceeds the bound {bound:.3f} of the bounded law for chi = {chi}")
    f = lambda ky: (chi - 1.0) * ky / ((ky + y_star) * (ky + chi * y_star)) - beta_star
    ky_peak = y_star * np.sqrt(chi)
    ky = brentq(f, ky_peak, 1e6 * y_star)
    kbase = kd_star * (1.0 + chi * y_star / ky) / (1.0 + y_star / ky)
    return dict(kbase=float(kbase), ky=float(ky), chi=float(chi),
                kd_min=float(kbase / chi), kd_max=float(kbase))


# --------------------------------------------------------------------------- #
#  Receptor maps
# --------------------------------------------------------------------------- #
def hill_n(scheme, y):
    if scheme == "cooperative":
        return N_COOP
    if scheme == "coregulated":
        return N_BASE + KAPPA * np.asarray(y, float)
    return 1.0


def theta_B(scheme, c, y, alpha):
    """Occupancy (bound fraction of the array) at ligand level c and modulator y."""
    y = np.asarray(y, float)
    kd = kd_of_y(y, alpha)
    if scheme == "weighted":
        return W_WEIGHT * c / (c + kd) + (1.0 - W_WEIGHT) * c / (c + K_FIX)
    n = hill_n(scheme, y)
    cn = np.power(c, n)
    return cn / (cn + np.power(kd, n))


def sensed(scheme, c0, c1, y, alpha):
    """Symbol-averaged occupancy g(y) = [theta_B(c0; y) + theta_B(c1; y)] / 2, which governs the slow loop response."""
    return 0.5 * theta_B(scheme, c0, y, alpha) + 0.5 * theta_B(scheme, c1, y, alpha)


# --------------------------------------------------------------------------- #
#  Equilibrium and steady state
# --------------------------------------------------------------------------- #
def y_equilibrium(scheme, alpha, c0=CL0_NOM, c1=CL1_NOM, setpoint=MU / THETA):
    """Equilibrium modulator concentration y*, at which g(y) equals the setpoint.  Closed form for the exponential
    law (K_D(y*) = sqrt(c0 c1) for every Hill-type receiver when the setpoint is 1/2), numerical root otherwise.
    Raises if the target lies outside the actuator range."""
    if ACT["law"] == "exp" and setpoint == 0.5 and scheme != "weighted":
        return float(np.log(ACT["kbase"] / np.sqrt(c0 * c1)) / alpha + M0)
    f = lambda y: sensed(scheme, c0, c1, y, alpha) - setpoint
    if ACT["law"] == "exp":
        lo, hi = -50.0 / alpha, 50.0 / alpha
    else:
        lo, hi = 0.0, 1e4 * ACT["ky"]
    if f(lo) * f(hi) > 0:
        raise ValueError("setpoint not reachable by the actuator (reachability condition violated)")
    return float(brentq(f, lo, hi))


def steady_state(scheme, alpha, motif="AIF", gc=0.0, c0=CL0_NOM, c1=CL1_NOM):
    """Deterministic steady state (concentrations). Exact for gc = 0; numerical
    root for leaky integration gc > 0.  Returns dict with y, z1, z2, z3, g."""
    if gc == 0.0:
        ys = y_equilibrium(scheme, alpha, c0, c1)
        z1 = GP * ys / K
        if motif == "AIF":
            z2, z3 = MU / (ETA * z1), 0.0
        else:
            z2 = EPS * MU / (ETA * z1)
            z3 = (1.0 - EPS) * MU / (ETA1 * z2)
        return dict(y=ys, z1=z1, z2=z2, z3=z3, g=sensed(scheme, c0, c1, ys, alpha))
    ss0 = steady_state(scheme, alpha, motif, 0.0, c0, c1)

    def F(v):
        z1, z2, y, z3 = v
        g = sensed(scheme, c0, c1, y, alpha)
        if motif == "AIF":
            return [MU - ETA * z1 * z2 - gc * z1,
                    THETA * g - ETA * z1 * z2 - gc * z2,
                    K * z1 - GP * y,
                    z3]
        return [EPS * MU - ETA * z1 * z2 - gc * z1,
                THETA * g - ETA * z1 * z2 - ETA1 * z2 * z3 - gc * z2,
                K * z1 - GP * y,
                (1.0 - EPS) * MU - ETA1 * z2 * z3 - gc * z3]

    sol, info, ier, msg = fsolve(F, [ss0["z1"], ss0["z2"], ss0["y"], ss0["z3"]],
                                 full_output=True, xtol=1e-12)
    if ier != 1:
        raise RuntimeError(f"steady state did not converge: {msg}")
    z1, z2, y, z3 = sol
    return dict(y=y, z1=z1, z2=z2, z3=z3, g=sensed(scheme, c0, c1, y, alpha))


def inventory(ss, omega=OMEGA_HC):
    """Expected total controller and modulator molecules Omega (y* + z1* + z2* + z3*)."""
    return float(omega * (ss["y"] + ss["z1"] + ss["z2"] + ss["z3"]))


# --------------------------------------------------------------------------- #
#  Receptor gain (symbol-averaged occupancy) and per-level occupancy slopes
# --------------------------------------------------------------------------- #
def level_slopes(scheme, alpha, ys, c0=CL0_NOM, c1=CL1_NOM, h=1e-6):
    """d theta_B(c_s; y)/dy at y = ys for s = 0, 1 (central difference)."""
    s0 = (theta_B(scheme, c0, ys + h, alpha) - theta_B(scheme, c0, ys - h, alpha)) / (2 * h)
    s1 = (theta_B(scheme, c1, ys + h, alpha) - theta_B(scheme, c1, ys - h, alpha)) / (2 * h)
    return float(s0), float(s1)


def zeta_star(scheme, alpha, c0=CL0_NOM, c1=CL1_NOM):
    """Receptor gain zeta* = dg/dy of the symbol-averaged occupancy at the equilibrium."""
    ys = y_equilibrium(scheme, alpha, c0, c1)
    s0, s1 = level_slopes(scheme, alpha, ys, c0, c1)
    return 0.5 * (s0 + s1)


def zeta_star_closed_form(scheme, alpha, c0=CL0_NOM, c1=CL1_NOM):
    """Closed form zeta* = n_H beta* theta0 theta1 (times w for the mixed array), valid at the equilibrium."""
    ys = y_equilibrium(scheme, alpha, c0, c1)
    n = float(hill_n(scheme, ys)) if scheme != "weighted" else 1.0
    r = c1 / c0
    th0 = 1.0 / (1.0 + r ** (n / 2.0))
    z = n * float(beta_local(ys, alpha)) * th0 * (1.0 - th0)
    return W_WEIGHT * z if scheme == "weighted" else z


def zeta_half_occupancy(scheme, alpha):
    """Slope n_H alpha / 4 at half occupancy, which overestimates the receptor gain."""
    ys = y_equilibrium(scheme, alpha)
    n = float(hill_n(scheme, ys)) if scheme != "weighted" else 1.0
    z = n * alpha / 4.0
    return W_WEIGHT * z if scheme == "weighted" else z


def occ_levels(scheme, alpha, ys, c0=CL0_NOM, c1=CL1_NOM):
    return float(theta_B(scheme, c0, ys, alpha)), float(theta_B(scheme, c1, ys, alpha))


def d_bits(scheme, alpha, t_sym=T_SYM, c0=CL0_NOM, c1=CL1_NOM):
    """Self-noise intensity D_bits = theta^2 (Delta theta)^2 T_sym / 4, i.e., the white-noise equivalent of the
    symbol-by-symbol switching of the sensing propensity (concentration units, independent of Omega)."""
    ys = y_equilibrium(scheme, alpha, c0, c1)
    th0, th1 = occ_levels(scheme, alpha, ys, c0, c1)
    return float(THETA ** 2 * (th1 - th0) ** 2 * t_sym / 4.0)


# --------------------------------------------------------------------------- #
#  Reduced (slow-manifold) loop quantities
# --------------------------------------------------------------------------- #
def rho_of(ss):
    return ss["z1"] / (ss["z1"] + ss["z2"])


def omega_c(zeta, rho, gc=0.0):
    if zeta <= 0:
        return 0.0
    a = (GP + gc) ** 2
    rhs = K * rho * THETA * zeta
    return float(np.sqrt((-a + np.sqrt(a ** 2 + 4 * rhs ** 2)) / 2))


def time_constant(zeta, rho, gc=0.0):
    """Loop time constant tau = 1/|Re lambda_slow| of the reduced 2x2 loop (e, y).
    (A first-order response reaches 2 % of a step after about 3.9 tau.)"""
    D = (GP - gc) ** 2 - 4 * K * THETA * zeta * rho
    return float(2.0 / (GP + gc - np.sqrt(max(D, 0.0))))



def var_dy(zeta, rho, ys, gc=0.0, omega=OMEGA_HC, dbits=0.0):
    """Stationary Var(dy) of the reduced loop as an Omega-scaled coefficient
    (concentration variance = value/Omega).  Closed form at gc = 0:
        y* + k rho (2 mu + Omega D_bits)/(2 gp theta zeta).
    dbits = 0 gives the chemical-noise part of eq. (14)."""
    Gy, Ge = 2 * GP * ys, 2 * MU + omega * dbits
    if gc == 0.0:
        return Gy / (2 * GP) + K * rho * Ge / (2 * GP * THETA * zeta)
    A = np.array([[-gc, -THETA * zeta], [K * rho, -GP]])
    D = np.diag([Ge, Gy])
    S = solve_continuous_lyapunov(A, -D)
    return float(S[1, 1])


def var_dy_lyapunov(zeta, rho, ys, gc=0.0, omega=OMEGA_HC, dbits=0.0):
    """Reduced-loop quantity by the numerical Lyapunov solve (cross-check)."""
    Gy, Ge = 2 * GP * ys, 2 * MU + omega * dbits
    A = np.array([[-gc, -THETA * zeta], [K * rho, -GP]])
    D = np.diag([Ge, Gy])
    S = solve_continuous_lyapunov(A, -D)
    return float(S[1, 1])


# --------------------------------------------------------------------------- #
#  Full three-state linear-noise covariance (AIF), including the self-noise
# --------------------------------------------------------------------------- #
def jacobian(zeta, ss, motif="AIF", gc=0.0):
    """Jacobian of (z1, z2, y[, z3]) with the occupancy algebraic in y."""
    z1, z2, z3 = ss["z1"], ss["z2"], ss["z3"]
    if motif == "AIF":
        return np.array([[-ETA * z2 - gc, -ETA * z1, 0.0],
                         [-ETA * z2, -ETA * z1 - gc, THETA * zeta],
                         [K, 0.0, -GP]])
    return np.array([[-ETA * z2 - gc, -ETA * z1, 0.0, 0.0],
                     [-ETA * z2, -ETA * z1 - ETA1 * z3 - gc, THETA * zeta, -ETA1 * z2],
                     [K, 0.0, -GP, 0.0],
                     [0.0, -ETA1 * z3, 0.0, -ETA1 * z2 - gc]])


def var_dy_full(zeta, ss, gc=0.0, omega=OMEGA_HC, dbits=0.0):
    """Var(dy) as an Omega-scaled coefficient from the full 3-state Lyapunov equation
    J C + C J^T + D = 0 with
        D = [[2 mu + gc z1, mu, 0], [mu, 2 mu + gc z2 + Omega D_bits, 0], [0, 0, 2 gp y*]]
    (the off-diagonal mu arises because one sequestration event removes both
    controller species, and the self-noise enters the production of Z2 only)."""
    J = jacobian(zeta, ss, "AIF", gc)
    z1, z2, y = ss["z1"], ss["z2"], ss["y"]
    D = np.array([[2 * MU + gc * z1, MU, 0.0],
                  [MU, 2 * MU + gc * z2 + omega * dbits, 0.0],
                  [0.0, 0.0, 2 * GP * y]])
    C = solve_continuous_lyapunov(J, -D)
    return float(C[2, 2])


def time_constant_full(zeta, ss, gc=0.0):
    ev = np.linalg.eigvals(jacobian(zeta, ss, "AIF", gc))
    return float(-1.0 / np.max(ev.real))


def is_stable(zeta, ss, motif="AIF", gc=0.0):
    return np.max(np.real(np.linalg.eigvals(jacobian(zeta, ss, motif, gc)))) < 0


def zeta_crit_numeric(ss, motif="AIF", gc=0.0, lo=1e-4, hi=1e4):
    """Smallest zeta at which the full loop loses stability (Hopf), by bisection."""
    if is_stable(hi, ss, motif, gc):
        return np.inf
    for _ in range(300):
        mid = np.sqrt(lo * hi)
        if is_stable(mid, ss, motif, gc):
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def zeta_crit_closed_form(ss, gc=0.0):
    """
    Routh-Hurwitz boundary for the 3-species AIF loop with algebraic occupancy.
    Characteristic polynomial s^3 + a2 s^2 + a1 s + a0 with
        zS = z1* + z2*,   q = gc (eta zS + gc),
        a2 = eta zS + 2 gc + gp,
        a1 = q + gp (eta zS + 2 gc),
        a0 = gp q + k theta zeta eta z1*.
    Stability iff a2 a1 > a0, i.e.  zeta < zeta_crit = [a2 a1 - gp q] / (k theta eta z1*).
    At gc = 0:  zeta_crit = gp (gp + eta zS) / (k theta rho),  rho = z1*/zS.
    """
    z1, S = ss["z1"], ss["z1"] + ss["z2"]
    c2 = gc * (ETA * S + gc)
    a2 = ETA * S + 2 * gc + GP
    a1 = c2 + GP * (ETA * S + 2 * gc)
    return float((a2 * a1 - GP * c2) / (K * THETA * ETA * z1))


def sf_reference(zeta, rho, gc=0.0):
    """Normalized two-stage reference of Section III-D, based on the production-degradation criterion of
    Olsman et al. (2019), theta1 theta2 k / 2 < gp (gp + gc)^2, with the loop gain k rho theta zeta of the present
    receiver in place of theta1 theta2 k.  Returns 1/S_f = [2 gp (gp + gc)^2 / (k theta zeta rho)]^(1/3), which is
    larger than one where the criterion predicts stability and equal to one at the reference S_f = 1.  This is a
    numerical reference only, whereas the stability boundary of the receiver is zeta_crit."""
    if zeta <= 0:
        return np.inf
    return float((2.0 * GP * (GP + gc) ** 2 / (K * THETA * zeta * rho)) ** (1.0 / 3.0))


# --------------------------------------------------------------------------- #
#  Gaussian approximation of the BEP (high copy numbers), used only to locate the design points
# --------------------------------------------------------------------------- #
def var_bind(theta, n_r=N_RECEPTORS):
    return theta * (1.0 - theta) / n_r


def bep_surrogate(scheme, alpha, omega=OMEGA_HC, gc=0.0, n_r=N_RECEPTORS, with_bits=True,
                  t_sym=T_SYM):
    """Gaussian BEP with per-level output slopes and the full-LNA modulator variance
    including the self-noise: sigma_s^2 = slope_s^2 Var(dy) + theta_s(1-theta_s)/N_R."""
    ss = steady_state(scheme, alpha, "AIF", gc)
    ys, rho = ss["y"], rho_of(ss)
    s0, s1 = level_slopes(scheme, alpha, ys)
    z = 0.5 * (s0 + s1)
    if z <= 0:
        return dict(bep=0.5, zeta=z)
    th0, th1 = occ_levels(scheme, alpha, ys)
    db = THETA ** 2 * (th1 - th0) ** 2 * t_sym / 4.0 if with_bits else 0.0
    v_full = var_dy_full(z, ss, gc, omega, db)          # Omega-scaled coefficient
    v_chem = var_dy_full(z, ss, gc, omega, 0.0)
    v_red = var_dy(z, rho, ys, gc, omega, db)
    v_red_chem = var_dy(z, rho, ys, gc, omega, 0.0)
    vdy = v_full / omega
    sig0 = np.sqrt(s0 ** 2 * vdy + var_bind(th0, n_r))
    sig1 = np.sqrt(s1 ** 2 * vdy + var_bind(th1, n_r))
    bep = 0.5 * Q((0.5 - th0) / sig0) + 0.5 * Q((th1 - 0.5) / sig1)
    return dict(bep=float(bep), zeta=z, rho=rho, ys=ys, th0=th0, th1=th1,
                dmu=th1 - th0, sig0=float(sig0), sig1=float(sig1),
                var_dy=v_full, var_dy_chem=v_chem, var_dy_reduced=v_red,
                var_dy_reduced_chem=v_red_chem, d_bits=db,
                var_dy_bits_term=v_full - v_chem,
                wc=omega_c(z, rho, gc), settling=time_constant(z, rho, gc),
                tau=time_constant(z, rho, gc), tau_full=time_constant_full(z, ss, gc),
                zeta_crit=zeta_crit_closed_form(ss, gc), slope0=s0, slope1=s1,
                n_h=float(hill_n(scheme, ys)) if scheme != "weighted" else 1.0,
                inventory=inventory(ss, omega), z1=ss["z1"], z2=ss["z2"],
                beta=float(beta_local(ys, alpha)))


def alpha_lower_bound(scheme):
    """Physical cap n_H(y*) <= NH_CAP for the coregulated scheme."""
    if scheme == "coregulated":
        return KAPPA * B_CAL / (NH_CAP - N_BASE)
    return ALPHA_RANGE[0]


def design_point(scheme, omega=OMEGA_HC, wc_min=WC_MIN, gc=0.0):
    """BEP-optimal actuator gain alpha subject to the bandwidth requirement
    omega_c >= wc_min.  No stability constraint is imposed: the exact boundary
    zeta_crit is reported as a margin instead (it is never active here)."""
    lo, hi = alpha_lower_bound(scheme), ALPHA_RANGE[1]
    if scheme == "coregulated":
        # co-regulated design: n_H(y*) = NH_CAP (the same cooperativity at the equilibrium as the
        # cooperative receptor), which fixes alpha = KAPPA B_CAL/(NH_CAP - N_BASE) = 1.
        out = bep_surrogate(scheme, lo, omega, gc)
        out.update(alpha=lo, margin=out["zeta_crit"] / out["zeta"],
                   sf_reference=sf_reference(out["zeta"], out["rho"], gc),
                   zeta_half_occupancy=zeta_half_occupancy(scheme, lo),
                   wc_bound_active=False, alpha_lower_active=True)
        return out

    def obj(a):
        r = bep_surrogate(scheme, a, omega, gc)
        if r["zeta"] <= 0 or r["wc"] < wc_min:
            return 1.0
        return r["bep"]

    grid = np.linspace(lo, hi, 600)
    vals = np.array([obj(a) for a in grid])
    a0 = grid[int(np.argmin(vals))]
    res = minimize_scalar(obj, bounds=(max(lo, a0 - 0.05), min(hi, a0 + 0.05)),
                          method="bounded", options=dict(xatol=1e-6))
    a = float(res.x) if res.fun <= vals.min() else float(a0)
    out = bep_surrogate(scheme, a, omega, gc)
    out.update(alpha=a, margin=out["zeta_crit"] / out["zeta"],
               sf_reference=sf_reference(out["zeta"], out["rho"], gc),
               zeta_half_occupancy=zeta_half_occupancy(scheme, a),
               wc_bound_active=bool(abs(out["wc"] - wc_min) < 1e-3 * wc_min),
               alpha_lower_active=bool(abs(a - lo) < 1e-6))
    return out


# --------------------------------------------------------------------------- #
#  Loop-speed design rule: self-noise and reaction noise (decreasing with tau) against
#  the tracking error under a periodic attenuation (increasing with tau)
# --------------------------------------------------------------------------- #
def tracking_mismatch_amplitude(b, omega_ch):
    """|1 - T(j omega)| of the reduced loop T(s) = b/(s^2 + gp s + b)."""
    s = 1j * omega_ch
    T = b / (s * s + GP * s + b)
    return float(abs(1.0 - T))


def surrogate_bep_varying(scheme, alpha, S, T_c, omega=OMEGA_HC, n_r=N_RECEPTORS, n_phase=64):
    """Gaussian approximation of the BEP on the periodic channel g(t) = exp(-1/2 ln S (1 - cos 2 pi t/T_c)):
    the affinity lags the moving optimum by delta ln K_D = -A |1-T| cos(...) with A = 1/2 ln S,
    which shifts both occupancies; the noise sigma is the static-channel value of (15)."""
    r = bep_surrogate(scheme, alpha, omega)
    b = K * r["rho"] * THETA * r["zeta"]
    mis = tracking_mismatch_amplitude(b, 2 * np.pi / T_c) * 0.5 * np.log(S)
    n = r["n_h"]
    sig = r["sig0"]
    th0, th1 = r["th0"], r["th1"]
    # occupancies under a mistuning e = ln(K_D/K_opt): theta_s = c^n/(c^n + K^n e^{n e})
    ph = np.linspace(0, 2 * np.pi, n_phase, endpoint=False)
    e = mis * np.cos(ph)
    l0 = np.log(th0 / (1 - th0)) - n * e          # logit of the occupancy for bit 0
    l1 = np.log(th1 / (1 - th1)) - n * e
    t0 = 1 / (1 + np.exp(-l0)); t1 = 1 / (1 + np.exp(-l1))
    bep = np.mean(0.5 * Q((0.5 - t0) / sig) + 0.5 * Q((t1 - 0.5) / sig))
    return dict(bep=float(bep), mismatch=mis, tau=r["tau"], b=b, sigma=sig, zeta=r["zeta"], alpha=alpha)


def design_rule_tau(scheme, S, T_c, omega=OMEGA_HC, n_r=N_RECEPTORS):
    """Leading-order rule tau_opt = (A/2B)^(1/3): noise variance A/tau against tracking variance B tau^2,
    with A = Delta-theta^2 T_sym/8 + [n theta0 theta1 gp/(k rho theta) + mu/theta^2]/Omega (the reaction-noise
    part is evaluated at rho of the design) and B = (n theta0 theta1 pi ln S)^2/(2 T_c^2)."""
    r = bep_surrogate(scheme, design_point(scheme)["alpha"], omega) if scheme != "coregulated" else bep_surrogate(scheme, 1.0, omega)
    n, th0, th1 = r["n_h"], r["th0"], r["th1"]
    A = (th1 - th0) ** 2 * T_SYM / 8 + (n * th0 * th1 * GP / (K * r["rho"] * THETA) + MU / THETA ** 2) / omega
    B = (n * th0 * th1 * np.pi * np.log(S)) ** 2 / (2 * T_c ** 2)
    return float((A / (2 * B)) ** (1 / 3)), float(A), float(B)


if __name__ == "__main__":
    print("zeta*: numeric vs closed form (alpha = 1.6)")
    for s in SCHEMES:
        print(f"  {s:12s} num={zeta_star(s, 1.6):.4f}  closed={zeta_star_closed_form(s, 1.6):.4f}"
              f"  half-occupancy slope={zeta_half_occupancy(s, 1.6):.4f}")
    print("\nzeta_crit: closed form vs numeric Hopf (AIF, gc = 0)")
    for a in (1.6, 1.0, 0.336):
        ss = steady_state("allosteric", a)
        print(f"  alpha={a:<5} y*={ss['y']:.3f}  closed={zeta_crit_closed_form(ss):.4f}"
              f"  numeric={zeta_crit_numeric(ss):.4f}")
    print("\nOmega Var(dy) at the non-cooperative design (alpha = 0.384): reduced and full loop, without and with self-noise")
    a = 0.384; ss = steady_state("allosteric", a); z = zeta_star("allosteric", a); rho = rho_of(ss)
    db = d_bits("allosteric", a)
    print(f"  D_bits={db:.5f}  reduced={var_dy(z, rho, ss['y']):.3f}  full={var_dy_full(z, ss):.3f}"
          f"  reduced+self-noise={var_dy(z, rho, ss['y'], dbits=db):.3f}  full+self-noise={var_dy_full(z, ss, dbits=db):.3f}")
    print("\nBounded-law calibration example (chi = 100, y* = 1.5, beta* = 0.437):")
    print(" ", atcm_calibrate(1.5, 0.437, 100.0))
