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
from scipy.optimize import minimize_scalar

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
