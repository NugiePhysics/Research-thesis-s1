"""Tests of the deflection-angle definitions against exact and analytic results."""
import numpy as np
import pytest

from kerr_ds import KerrDeSitter, analytic, initial_state, kerr_horizon_radius, trace_photon


def _simulate(M, a, lmd, b, r0):
    sptm = KerrDeSitter(M, a, lmd)
    x, p = initial_state(sptm, r0, np.pi / 2, 0.0, 1.0, b)
    return trace_photon(sptm, x, p, r0, kerr_horizon_radius(M, a))


def test_flat_space_has_zero_deflection():
    # M = 0: the article definition gives exactly zero; the coordinate value is the geometric offset
    res = _simulate(0.0, 0.0, 0.0, 7.0, 100.0)
    assert res.escaped
    assert abs(res.deflection_deg) < 1e-7
    offset = np.degrees(np.arcsin(7 / 100) + np.arcsin(7 / res.r[-1]))
    assert res.deflection_coord_deg == pytest.approx(-offset, abs=1e-7)


@pytest.mark.parametrize("a, lmd", [(0.0, 5e-5), (0.99, -5e-5), (-0.6, 5e-5)])
def test_simulation_matches_exact_quadrature(a, lmd):
    res = _simulate(1.0, a, lmd, 7.0, 100.0)
    r_obs = res.r[-1]
    assert res.deflection_deg == pytest.approx(analytic.deflection_exact_deg(1.0, a, lmd, 7.0, 100.0, r_obs),
                                               abs=1e-5)
    assert res.deflection_coord_deg == pytest.approx(
        analytic.coordinate_deflection_exact_deg(1.0, a, lmd, 7.0, 100.0, r_obs), abs=1e-5)


@pytest.mark.parametrize("a", [0.0, 0.6, -0.6])
def test_positive_lambda_reduces_deflection_as_in_articles(a):
    def delta(lmd):
        return analytic.deflection_exact_deg(1.0, a, lmd, 7.0, 500.0, 505.0)

    assert delta(1e-6) < delta(0.0) < delta(-1e-6)
    # The coordinate sweep alone has the opposite trend for a = 0 (Huang et al. 2025, Sec. 4.1.2)
    if a == 0.0:
        coord = [analytic.coordinate_deflection_exact_deg(1.0, a, lmd, 7.0, 500.0, 505.0) for lmd in (0.0, 1e-6)]
        assert coord[1] > coord[0]


@pytest.mark.parametrize("a", [0.0, 0.5, -0.5])
def test_lambda_shift_agrees_with_articles_in_weak_field(a):
    # b = 50: weak field, where the expansions of Huang et al. and Belhaj et al. apply
    args = (1.0, a, 1e-7, 50.0, 500.0, 500.0)
    args0 = (1.0, a, 0.0, 50.0, 500.0, 500.0)
    exact = analytic.deflection_exact_deg(*args) - analytic.deflection_exact_deg(*args0)
    huang = analytic.huang2025_deflection_deg(*args) - analytic.huang2025_deflection_deg(*args0)
    belhaj = analytic.belhaj2022_deflection_deg(*args) - analytic.belhaj2022_deflection_deg(*args0)
    assert exact < 0
    assert huang == pytest.approx(exact, rel=0.03)
    assert belhaj == pytest.approx(exact, rel=0.05)


def test_huang_formula_far_distance_limit():
    M, a, b = 1.0, 0.3, 40.0
    expected = 4 * M / b + 15 * np.pi * M**2 / (4 * b**2) - 4 * a * M / b**2
    assert analytic.huang2025_deflection_deg(M, a, 0.0, b, 1e12, 1e12) == pytest.approx(np.degrees(expected),
                                                                                       rel=1e-9)


@pytest.mark.parametrize("a", [0.0, 0.5, 0.99, -0.5, -0.99])
def test_critical_impact_parameter_matches_congdon(a):
    # Congdon et al. (2022), Eq. (17), for L > 0 (a < 0: retrograde)
    expected = -a + 6 * np.cos(np.arccos(-a) / 3)
    b_c, _ = analytic.critical_impact_parameter(1.0, a, 0.0)
    assert b_c == pytest.approx(expected, rel=1e-9)


def test_asymptotic_schwarzschild_deflection():
    assert analytic.asymptotic_kerr_deflection_deg(1.0, 0.0, 7.0) == pytest.approx(64.608962, abs=1e-6)


def test_cosmological_horizon_and_static_limit():
    rc = analytic.cosmological_horizon(1.0, 0.99, 5e-6)
    assert rc == pytest.approx(773.5947, abs=1e-3)
    assert abs(analytic._delta_r(rc, 1.0, 0.99, 5e-6)) < 1e-9
    r_stat = analytic.static_limit_radius(1.0, 0.99, 5e-6)
    assert r_stat < rc and r_stat == pytest.approx(rc, abs=1e-3)
    assert np.isinf(analytic.cosmological_horizon(1.0, 0.5, 0.0))
    assert np.isinf(analytic.cosmological_horizon(1.0, 0.5, -1e-6))


@pytest.mark.parametrize("a, lmd", [(0.0, 0.0), (0.99, 5e-6), (-0.6, -5e-6)])
def test_travel_time_and_periapsis_match_simulation(a, lmd):
    res = _simulate(1.0, a, lmd, 7.0, 100.0)
    r_obs = float(res.r[-1])
    t_exact = analytic.travel_time_exact(1.0, a, lmd, 7.0, 100.0, r_obs)
    assert res.delta_t == pytest.approx(t_exact, abs=1e-4)
    assert res.r_min == pytest.approx(analytic.turning_point(1.0, a, lmd, 7.0), abs=5e-3)


def test_flat_space_shapiro_delay_is_zero():
    assert analytic.shapiro_delay_exact(0.0, 0.0, 0.0, 7.0, 100.0, 105.0) == pytest.approx(0.0, abs=1e-11)


def test_strong_deflection_coefficients_schwarzschild():
    a_bar, b_bar = analytic.strong_deflection_coefficients(1.0, 0.0, 0.0, 500.0, 505.0)
    # For Schwarzschild (a = 0, Lambda = 0), a_bar = 1 rad = 180 / pi deg
    assert a_bar == pytest.approx(np.degrees(1.0), rel=1e-6)
    bc, _ = analytic.critical_impact_parameter(1.0, 0.0, 0.0)
    b_test = bc * 1.0001
    pred = -a_bar * np.log(b_test / bc - 1.0) + b_bar
    exact = analytic.deflection_exact_deg(1.0, 0.0, 0.0, b_test, 500.0, 505.0)
    assert pred == pytest.approx(exact, abs=0.05)


@pytest.mark.parametrize("a", [-0.99, 0.0, 0.99])
def test_lambda_crossover_impact_parameter(a):
    # Every spin has a strong-field crossover b_cross > b_c where delta(Lambda) - delta(0) = 0;
    # only for extreme retrograde spin (a = -0.99) is b_cross > 7M.
    bc, _ = analytic.critical_impact_parameter(1.0, a, 5e-6)
    b_cross = analytic.lambda_crossover_impact_parameter(1.0, a, 5e-6, 500.0, 505.0)
    assert bc < b_cross < 15.0
    if a == -0.99:
        assert b_cross > 7.0
    else:
        assert b_cross < 7.0
    diff = analytic.deflection_exact_deg(1.0, a, 5e-6, b_cross, 500.0, 505.0) - analytic.deflection_exact_deg(
        1.0, a, 0.0, b_cross, 500.0, 505.0
    )
    assert abs(diff) < 1e-7


def test_shadow_and_einstein_ring_angles():
    # Positive Lambda shrinks the shadow angular radius for a static observer
    sh_0 = analytic.shadow_angular_radius_deg(1.0, 0.0, 0.0, 500.0)
    sh_pos = analytic.shadow_angular_radius_deg(1.0, 0.0, 5e-6, 500.0)
    assert sh_pos < sh_0
    theta_e, b_e = analytic.einstein_ring_angle_deg(1.0, 0.0, 0.0, 500.0, 500.0)
    assert b_e == pytest.approx(np.sqrt(4.0 * 1.0 * 250.0), rel=0.05)
    assert theta_e == pytest.approx(np.degrees(b_e / 500.0), rel=0.02)


def test_astrophysical_lensing_summary_sgr_a():
    s = analytic.astrophysical_lensing_summary(4.15e6, 1.11e-52, 8.18 * 3.0857e19, 8.18 * 3.0857e19, b_over_M=7.0)
    assert 1e-34 < s["lmd_geom"] < 1e-31
    assert s["delta_lambda_uas"] < 0.0
