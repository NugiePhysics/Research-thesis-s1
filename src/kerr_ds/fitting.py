"""Nonlinear fits of the deflection angle (scipy curve_fit) and the coefficient of determination R^2."""
import numpy as np
from scipy.optimize import curve_fit


def power_law(x, A, B, C):
    """Deflection-versus-Lambda model: A * x^B + C.

    x must be positive: for Lambda < 0 fit against |Lambda| (a negative number
    raised to a non-integer power is undefined).
    """
    return A * (x**B) + C


def linear(x, A, C):
    """Deflection-versus-Lambda model A * x + C (adequate while Lambda r0^2 << 1)."""
    return A * x + C


def exponential_decay(x, A, B, C):
    """Deflection-versus-spin model: A * exp(-B x) + C."""
    return A * np.exp(-B * x) + C


def r_squared(y, y_pred):
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot."""
    y = np.asarray(y)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot


def fit_curve(model, x, y, p0, maxfev=10000, return_errors=False):
    """Fit `model` to the data (x, y); return (optimal parameters, R^2).

    With `return_errors=True`, also return the 1-sigma parameter uncertainties
    (square root of the diagonal of the covariance matrix from curve_fit), so
    that e.g. an exponent B can be compared with 1.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    popt, pcov = curve_fit(model, x, y, p0=p0, maxfev=maxfev)
    r2 = r_squared(y, model(x, *popt))
    if return_errors:
        return popt, r2, np.sqrt(np.diag(pcov))
    return popt, r2
