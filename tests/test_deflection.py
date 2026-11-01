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
