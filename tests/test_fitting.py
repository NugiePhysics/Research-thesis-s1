"""Tests for the fitting utilities and the spin sweep."""
import numpy as np
import pytest

from kerr_ds import exponential_decay, fit_curve, power_law, r_squared, run_spin_sweep


def test_r_squared_perfect_prediction():
    y = np.array([1.0, 2.0, 4.0])
    assert r_squared(y, y) == 1.0


@pytest.mark.parametrize("model, x, true_params, p0", [
    (power_law, np.logspace(-10, np.log10(5e-6), 7), [2.4e4, 1.0, 43.9], [5000, 1.0, 43.9]),
    (exponential_decay, np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.99]), [32.4, 0.9, 30.6], [300, 5, 70]),
])
def test_fit_curve_recovers_parameters(model, x, true_params, p0):
    popt, r2 = fit_curve(model, x, model(x, *true_params), p0=p0, maxfev=20000)
    np.testing.assert_allclose(popt, true_params, rtol=1e-6)
    assert r2 == pytest.approx(1.0)


def test_spin_sweep_matches_main_notebook():
    # a = 0.99, Lambda = 0, b = 7: same value as in the main notebook's table
    res = run_spin_sweep(1.0, [0.99], [0.0], 500.0, np.pi/2, 0.0, 1.0, 7.0, n_jobs=1, verbose=False)
    assert list(res) == [(0.99, 0.0)]
    assert f"{res[0.99, 0.0].deflection_deg:.4f}" == "43.9149"
    assert res[0.99, 0.0].steps == 20140
