"""Exact benchmarks and analytic deflection formulas from the main articles (equatorial plane).

All functions take the impact parameter `b = L/E` in the normalisation used by the
simulation (photon with E = -p_t = 1 in the time coordinate of `symbolic.metric`).
That time coordinate is t_std / Xi, with Xi = 1 + Lambda a^2 / 3, so the impact
parameter of the standard Kerr-de Sitter metric used in the articles is
b_std = Xi * b (a relative difference of ~1e-6 for the parameters used here).
Angles are returned in degrees.

Exact (non-perturbative) results follow from the first integrals of the equatorial
null geodesic. With B = Xi b and P = r^2 + a^2 - a B,

    R(r)    = P^2 - Delta_r (B - a)^2            (a quartic in r)
    dphi/dr = Xi [a P / Delta_r + (B - a)] / sqrt(R).

References
----------
Belhaj, Belmahi, Benali, El Moumni, Chin. J. Phys. 80, 229 (2022), Eq. (5.1).
Huang, Fu, Lu, Qin, arXiv:2505.06999 (2025), Eqs. (12) and (24).
Ishihara, Suzuki, Ono, Kitamura, Asada, Phys. Rev. D 94, 084015 (2016).
Congdon, Iyer, Keeton, arXiv:2207.08987 (2022), Eq. (17).
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

_QUAD = dict(epsabs=1e-13, epsrel=1e-13, limit=500)


def _xi(a, lmd):
    return 1 + lmd * a**2 / 3


def _delta_r(r, M, a, lmd):
    return (r**2 + a**2) * (1 - lmd * r**2 / 3) - 2 * M * r


def _radial_quartic(M, a, lmd, b):
    """Coefficients of R(r) (highest power first)."""
    B = _xi(a, lmd) * b
    k = (B - a)**2
    c0 = a**2 - a * B
    return np.array([1 + k * lmd / 3, 0.0, 2 * c0 - k * (1 - lmd * a**2 / 3), 2 * M * k, c0**2 - k * a**2])


def turning_point(M, a, lmd, b, r_max=np.inf):
    """Radius of closest approach r_min (largest root of R below r_max).

    Raises ValueError if there is no turning point (the photon is captured).
    """
    coeffs = _radial_quartic(M, a, lmd, b)
    roots = [z.real for z in np.roots(coeffs) if abs(z.imag) <= 1e-9 * max(1.0, abs(z))]
    roots = [z for z in roots if 0 < z < r_max]
    if not roots:
        raise ValueError(f"No turning point for M={M}, a={a}, Lambda={lmd}, b={b}: photon is captured")
    r = max(roots)
    dcoeffs = np.polyder(coeffs)
    for _ in range(5):  # Newton polish of the numerical root
        r -= np.polyval(coeffs, r) / np.polyval(dcoeffs, r)
    return r


def _phi_leg(M, a, lmd, b, r_min, r_end):
    """Azimuth swept between the turning point and r_end (rad), by quadrature.

    The root r_min is factored out of R(r) exactly and r = r_min + y^2 removes the
    integrable 1/sqrt singularity at the turning point.
    """
    q, _ = np.polydiv(_radial_quartic(M, a, lmd, b), np.array([1.0, -r_min]))
    B = _xi(a, lmd) * b
    xi = _xi(a, lmd)

    def integrand(y):
        r = r_min + y * y
        P = r**2 + a**2 - a * B
        return 2 * xi * (a * P / _delta_r(r, M, a, lmd) + (B - a)) / np.sqrt(np.polyval(q, r))

    return quad(integrand, 0.0, np.sqrt(r_end - r_min), **_QUAD)[0]


def azimuth_sweep_exact(M, a, lmd, b, r_source, r_observer):
    """Exact coordinate azimuth phi_OS (rad) swept from r_source to r_observer."""
    r_min = turning_point(M, a, lmd, b, r_max=min(r_source, r_observer))
    return _phi_leg(M, a, lmd, b, r_min, r_source) + _phi_leg(M, a, lmd, b, r_min, r_observer)


def static_observer_angle(M, a, lmd, b, r):
    """Acute angle psi (rad) between the ray and the radial direction for a static observer at r.

    sin(psi) = sqrt(-g_tt) (b + g_tphi / g_tt) / sqrt(g_phiphi - g_tphi^2 / g_tt), equatorial plane.
    For Schwarzschild-de Sitter this is sin(psi) = b sqrt(1 - 2M/r - Lambda r^2 / 3) / r.
    """
    D, xi = _delta_r(r, M, a, lmd), _xi(a, lmd)
    g_tt = (a**2 - D) / r**2
    g_tp = a / (r**2 * xi) * (D - (r**2 + a**2))
    g_pp = ((r**2 + a**2)**2 - D * a**2) / (r**2 * xi**2)
    if not g_tt < 0:
        return np.nan
    return np.arcsin(np.sqrt(-g_tt) * (b + g_tp / g_tt) / np.sqrt(g_pp - g_tp**2 / g_tt))


def deflection_exact_deg(M, a, lmd, b, r_source, r_observer):
    """Exact finite-distance deflection angle delta = Psi_O - Psi_S + phi_OS (degrees).

    Same definition as `TrajectoryResult.deflection_deg`, with Psi_S = pi - psi_S.
    """
    phi_os = azimuth_sweep_exact(M, a, lmd, b, r_source, r_observer)
    psi_s = static_observer_angle(M, a, lmd, b, r_source)
    psi_o = static_observer_angle(M, a, lmd, b, r_observer)
    return np.degrees(phi_os - np.pi + psi_s + psi_o)


def coordinate_deflection_exact_deg(M, a, lmd, b, r_source, r_observer):
    """Exact phi_OS - pi (degrees); same quantity as `TrajectoryResult.deflection_coord_deg`."""
    return np.degrees(azimuth_sweep_exact(M, a, lmd, b, r_source, r_observer) - np.pi)


def asymptotic_kerr_deflection_deg(M, a, b):
    """Exact deflection angle in Kerr (Lambda = 0) with source and observer at infinity (degrees)."""
    r_min = turning_point(M, a, 0.0, b)
    q, _ = np.polydiv(_radial_quartic(M, a, 0.0, b), np.array([1.0, -r_min]))

    def integrand(x):
        # r = r_min / (1 - x^2) maps x in [0, 1) to r in [r_min, inf)
        r = r_min / (1 - x * x)
        dr_dx = 2 * x * r_min / (1 - x * x)**2
        P = r**2 + a**2 - a * b
        dphi_dr = (a * P / _delta_r(r, M, a, 0.0) + (b - a)) / np.sqrt((r - r_min) * np.polyval(q, r))
        return dphi_dr * dr_dx

    return np.degrees(2 * quad(integrand, 0.0, 1.0, **_QUAD)[0] - np.pi)


def huang2025_deflection_deg(M, a, lmd, b, r_source, r_observer):
    """Huang et al. (2025), Eq. (24): finite-distance deflection angle in Kerr-de Sitter.

    Weak-field result to second order in M/b, a/b and Lambda b^2 (degrees).
    """
    b = _xi(a, lmd) * b
    uS, uO = 1 / r_source, 1 / r_observer
    sS, sO = np.sqrt(1 - b**2 * uS**2), np.sqrt(1 - b**2 * uO**2)
    delta = (M * 2 / b * (sO + sS)
             - lmd * b / 6 * (sO / uO + sS / uS)
             + M**2 / (4 * b**2) * (15 * (np.arccos(b * uO) + np.arccos(b * uS))
                                    + (15 * b * uO - 7 * b**3 * uO**3) / sO
                                    + (15 * b * uS - 7 * b**3 * uS**3) / sS)
             - M * a * 2 / b**2 * (sO + sS)
             + M * lmd * b / 6 * (1 / sO + 1 / sS)
             + a * lmd * 2 / 3 * (sO / uO + sS / uS)
             - lmd**2 * b / 72 * ((1 - b**2 * uO**2 - 2 * b**4 * uO**4) / (uO**3 * sO)
                                  + (1 - b**2 * uS**2 - 2 * b**4 * uS**4) / (uS**3 * sS)))
    return np.degrees(delta)


def belhaj2022_deflection_deg(M, a, lmd, b, r_source, r_observer):
    """Belhaj et al. (2022), Eq. (5.1): deflection angle in Kerr with a cosmological constant (degrees)."""
    b = _xi(a, lmd) * b
    inv_u = r_source + r_observer  # 1/u_R + 1/u_S
    return np.degrees(4 * M / b - 4 * a * M / b**2 + b * lmd * M / 3
                      - inv_u * b * lmd / 6 + inv_u * 2 * a * lmd / 3)


def critical_impact_parameter(M, a, lmd):
    """Critical impact parameter b_c (simulation normalisation) and photon-orbit radius, for L > 0.

    Photons with b > b_c are scattered, those with b < b_c are captured. For Lambda = 0
    this is |b_c| = -a + 6M cos[arccos(-a/M) / 3] (Congdon et al. 2022, Eq. 17), e.g.
    b_c = 3 sqrt(3) M for a = 0 and b_c = 7M for a = -M (retrograde, extremal).
    """
    def B_of_r(r):
        sD = np.sqrt(_delta_r(r, M, a, lmd))
        return (r**2 + a**2 + a * sD) / (a + sD)

    # Bracket the minimum on a grid outside the event horizon, then refine
    r = np.linspace(M * 0.5, M * 10, 20001)
    with np.errstate(invalid="ignore", divide="ignore"):
        D = _delta_r(r, M, a, lmd)
        valid = (D > 0) & (r > _outer_horizon(M, a, lmd)) & (a + np.sqrt(np.where(D > 0, D, 0)) > 0)
        B = np.where(valid, B_of_r(np.where(valid, r, 1.0)), np.inf)
    i = int(np.argmin(B))
    res = minimize_scalar(B_of_r, bounds=(r[max(i - 1, 0)], r[min(i + 1, len(r) - 1)]),
                          method="bounded", options=dict(xatol=1e-12))
    return res.fun / _xi(a, lmd), res.x


def _outer_horizon(M, a, lmd):
    """Event horizon: largest root of Delta_r below 10 M (excludes the cosmological horizon)."""
    coeffs = [-lmd / 3, 0.0, 1 - lmd * a**2 / 3, -2 * M, a**2] if lmd != 0 else [1.0, -2 * M, a**2]
    roots = [z.real for z in np.roots(coeffs) if abs(z.imag) < 1e-9 and 0 < z.real < 10 * M]
    return max(roots) if roots else 0.0


def cosmological_horizon(M, a, lmd):
    """Cosmological horizon r_c: largest positive root of Delta_r = 0 above 10 M (inf for Lambda <= 0)."""
    if lmd <= 0:
        return np.inf
    coeffs = np.array([-lmd / 3, 0.0, 1 - lmd * a**2 / 3, -2 * M, a**2], dtype=float)
    roots = [z.real for z in np.roots(coeffs) if abs(z.imag) < 1e-9 and z.real > 10 * M]
    if not roots:
        return np.inf
    r = float(max(roots))
    dcoeffs = np.polyder(coeffs)
    for _ in range(5):
        r -= float(np.polyval(coeffs, r) / np.polyval(dcoeffs, r))
    return r


def static_limit_radius(M, a, lmd):
    """Outer cosmological static limit on the equatorial plane where g_tt = 0 (inf for Lambda <= 0)."""
    if lmd <= 0:
        return np.inf
    coeffs = np.array([-lmd / 3, 0.0, 1 - lmd * a**2 / 3, -2 * M], dtype=float)
    roots = [z.real for z in np.roots(coeffs) if abs(z.imag) < 1e-9 and z.real > 10 * M]
    if not roots:
        return np.inf
    r = float(max(roots))
    dcoeffs = np.polyder(coeffs)
    for _ in range(5):
        r -= float(np.polyval(coeffs, r) / np.polyval(dcoeffs, r))
    return r


def _t_leg(M, a, lmd, b, r_min, r_end):
    """Coordinate time (in simulation time t_code) elapsed between r_min and r_end."""
    q, _ = np.polydiv(_radial_quartic(M, a, lmd, b), np.array([1.0, -r_min]))
    B = _xi(a, lmd) * b

    def integrand(y):
        r = r_min + y * y
        P = r**2 + a**2 - a * B
        num = (r**2 + a**2) * P / _delta_r(r, M, a, lmd) + a * (B - a)
        return 2 * num / np.sqrt(np.polyval(q, r))

    return quad(integrand, 0.0, np.sqrt(r_end - r_min), **_QUAD)[0]


def travel_time_exact(M, a, lmd, b, r_source, r_observer):
    """Exact coordinate travel time Delta t_OS (units of M, simulation t_code normalisation)."""
    r_min = turning_point(M, a, lmd, b, r_max=min(r_source, r_observer))
    return _t_leg(M, a, lmd, b, r_min, r_source) + _t_leg(M, a, lmd, b, r_min, r_observer)


def shapiro_delay_exact(M, a, lmd, b, r_source, r_observer):
    """Coordinate time delay relative to the Euclidean straight-line path sqrt(r_S^2 - b^2) + sqrt(r_O^2 - b^2)."""
    t_flat = np.sqrt(r_source**2 - b**2) + np.sqrt(r_observer**2 - b**2)
    return travel_time_exact(M, a, lmd, b, r_source, r_observer) - t_flat


def strong_deflection_coefficients(M, a, lmd, r_source, r_observer):
    """Strong-deflection limit coefficients (a_bar_deg, b_bar_deg) of Bozza (2002) at finite distance.

    Near b -> b_c^+, the deflection angle (in degrees) obeys
        delta(b) = -a_bar_deg * ln(b / b_c - 1) + b_bar_deg + O((b - b_c) ln(b - b_c)).
    a_bar is computed from the exact second derivative of R(r) at the photon sphere r_ph;
    b_bar is extracted from the exact finite-distance deflection very close to b_c.
    """
    b_c, r_ph = critical_impact_parameter(M, a, lmd)
    coeffs = _radial_quartic(M, a, lmd, b_c)
    r_pp = np.polyval(np.polyder(coeffs, 2), r_ph)
    xi = _xi(a, lmd)
    B_c = xi * b_c
    P_ph = r_ph**2 + a**2 - a * B_c
    f_phi = xi * (a * P_ph / _delta_r(r_ph, M, a, lmd) + (B_c - a))
    a_bar_rad = f_phi / np.sqrt(0.5 * r_pp)
    a_bar_deg = float(np.degrees(a_bar_rad))

    eps1, eps2 = 1e-6, 2e-6
    d1 = deflection_exact_deg(M, a, lmd, b_c * (1.0 + eps1), r_source, r_observer) + a_bar_deg * np.log(eps1)
    d2 = deflection_exact_deg(M, a, lmd, b_c * (1.0 + eps2), r_source, r_observer) + a_bar_deg * np.log(eps2)
    b_bar_deg = float(2.0 * d1 - d2)
    return a_bar_deg, b_bar_deg


def lambda_crossover_impact_parameter(M, a, lmd, r_source, r_observer, b_max=50.0):
    """Impact parameter b_cross > b_c where delta(M, a, lmd, b) - delta(M, a, 0, b) == 0.

    Returns np.nan if no sign reversal occurs on (b_c, b_max] (e.g. prograde spins).
    """
    if lmd == 0:
        return np.nan
    bc_lmd, _ = critical_impact_parameter(M, a, lmd)
    bc_0, _ = critical_impact_parameter(M, a, 0.0)
    b_low = max(bc_lmd, bc_0) * (1.0 + 1e-5)
    if b_low >= b_max:
        return np.nan

    def diff(b):
        return deflection_exact_deg(M, a, lmd, b, r_source, r_observer) - deflection_exact_deg(
            M, a, 0.0, b, r_source, r_observer
        )

    f_low, f_high = diff(b_low), diff(b_max)
    if np.isnan(f_low) or np.isnan(f_high) or f_low * f_high > 0:
        return np.nan
    return float(brentq(diff, b_low, b_max, xtol=1e-9))


def shadow_angular_radius_deg(M, a, lmd, r_observer):
    """Angular radius psi_sh (deg) of the black-hole shadow at the equatorial plane for a static observer at r_observer."""
    b_c, _ = critical_impact_parameter(M, a, lmd)
    return float(np.degrees(static_observer_angle(M, a, lmd, b_c, r_observer)))


def einstein_ring_angle_deg(M, a, lmd, r_source, r_observer):
    """Einstein ring angular radius theta_E (deg) and impact parameter b_E (units of M) for an aligned source.

    Solves the finite-distance on-axis alignment condition psi_O(b) + psi_S(b) = delta_rad(b).
    """
    b_c, _ = critical_impact_parameter(M, a, lmd)
    b_low = b_c * 1.01
    b_est = np.sqrt(4.0 * M * r_source * r_observer / (r_source + r_observer))
    r_end = min(r_source, r_observer)
    D, xi = _delta_r(r_end, M, a, lmd), _xi(a, lmd)
    g_tt = (a**2 - D) / r_end**2
    g_tp = a / (r_end**2 * xi) * (D - (r_end**2 + a**2))
    g_pp = ((r_end**2 + a**2)**2 - D * a**2) / (r_end**2 * xi**2)
    b_turn_max = (np.sqrt(g_pp - g_tp**2 / g_tt) / np.sqrt(-g_tt) - g_tp / g_tt) * 0.9
    b_high = min(max(b_est * 2.5, b_low * 2.0), b_turn_max)

    def lens_eq(b):
        psi_s = static_observer_angle(M, a, lmd, b, r_source)
        psi_o = static_observer_angle(M, a, lmd, b, r_observer)
        delta_rad = np.radians(deflection_exact_deg(M, a, lmd, b, r_source, r_observer))
        return (psi_o + psi_s) - delta_rad

    b_e = float(brentq(lens_eq, b_low, b_high, xtol=1e-10))
    theta_e_deg = float(np.degrees(static_observer_angle(M, a, lmd, b_e, r_observer)))
    return theta_e_deg, b_e


def astrophysical_lensing_summary(M_solar, lmd_si, r_source_m, r_observer_m, b_over_M=7.0, a=0.0):
    """Convert SI astrophysical black-hole parameters to geometrized units (M = 1) and estimate Lambda shift.

    Parameters
    ----------
    M_solar : float
        Black-hole mass in solar masses.
    lmd_si : float
        Cosmological constant in m^-2 (e.g. 1.11e-52).
    r_source_m, r_observer_m : float
        Source and observer distances from the black hole in meters.
    b_over_M : float
        Impact parameter in units of M = GM/c^2.
    a : float
        Dimensionless spin parameter a/M.
    """
    G_SI = 6.67430e-11
    C_SI = 299792458.0
    M_SUN_KG = 1.98847e30
    r_g = G_SI * (M_solar * M_SUN_KG) / C_SI**2  # gravitational radius in meters
    lmd_geom = lmd_si * r_g**2
    r_s = r_source_m / r_g
    r_o = r_observer_m / r_g
    r_c = np.sqrt(3.0 / lmd_geom) if lmd_geom > 0 else np.inf
    # Leading cosmological deflection shift (rad): -Lambda * b * (r_S + r_O) / 6 + 2 * a * Lambda * (r_S + r_O) / 3
    delta_lambda_rad = (-lmd_geom * b_over_M / 6.0 + 2.0 * a * lmd_geom / 3.0) * (r_s + r_o)
    delta_lambda_uas = float(np.degrees(delta_lambda_rad) * 3600.0 * 1e6)
    return {
        "r_g_m": float(r_g),
        "lmd_geom": float(lmd_geom),
        "r_source_M": float(r_s),
        "r_observer_M": float(r_o),
        "r_c_M": float(r_c),
        "pi_lambda": float(lmd_geom * b_over_M * (r_s + r_o)),
        "delta_lambda_uas": delta_lambda_uas,
    }
