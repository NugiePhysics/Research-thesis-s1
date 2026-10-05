"""Nonlinear fits of the deflection angle (scipy curve_fit) and the coefficient of determination R^2."""
import numpy as np
from scipy.optimize import curve_fit


def power_law(x, A, B, C):
    """Deflection-versus-Lambda model: A * x^B + C."""
    return A * (x**B) + C


def exponential_decay(x, A, B, C):
    """Deflection-versus-spin model: A * exp(-B x) + C."""
    return A * np.exp(-B * x) + C


def r_squared(y, y_pred):
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot."""
    y = np.asarray(y)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot


def fit_curve(model, x, y, p0, maxfev=10000):
    """Fit `model` to the data (x, y); return (optimal parameters, R^2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    popt, _ = curve_fit(model, x, y, p0=p0, maxfev=maxfev)
    return popt, r_squared(y, model(x, *popt))
