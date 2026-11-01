"""Kerr-de Sitter spacetime: numerical evaluation of the Hamiltonian and its derivatives."""
import functools

import numpy as np
import sympy as sp

from . import symbolic
from .base import Spacetime

_ARGS = symbolic.COORDS + symbolic.MOMENTA + symbolic.PARAMS


@functools.cache
def _numeric_functions():
    """Lambdify the symbolic expressions (once per process, shared by all instances).

    dH/dx and dH/dp are combined into a single function with `cse=True` because
    they share many subexpressions, and use the `math` module (faster than numpy
    for scalars).
    """
    exprs = symbolic.derive_hamiltonian()
    f_H = sp.lambdify(_ARGS, exprs['H'], 'math', cse=True)
    f_grad = sp.lambdify(_ARGS, [exprs['dHdx'], exprs['dHdp']], 'math', cse=True)
    f_g_inv = sp.lambdify(symbolic.COORDS + symbolic.PARAMS, exprs['g_inv'], 'numpy')  # For the initial conditions
    return f_H, f_grad, f_g_inv


@functools.cache
def _metric_function():
    """Covariant metric g_uv as a numerical function (used for the observer angles)."""
    return sp.lambdify(symbolic.COORDS + symbolic.PARAMS, symbolic.metric(), 'numpy')


class KerrDeSitter(Spacetime):
    def __init__(self, M, a, lmd):
        self.M_val = M
        self.a_val = a
        self.lmd_val = lmd
        self.f_H, self.f_grad, self.f_g_inv = _numeric_functions()

    def _args(self, x, p):
        # tolist() -> Python floats: scalar arithmetic is much faster than with np.float64
        return (*x.tolist(), *p.tolist(), self.M_val, self.a_val, self.lmd_val)

    def H(self, x, p):
        return self.f_H(*self._args(x, p))

    def grad(self, x, p):
        dHdx, dHdp = self.f_grad(*self._args(x, p))
        return np.array(dHdx), np.array(dHdp)

    def dHdx(self, x, p):
        return self.grad(x, p)[0]

    def dHdp(self, x, p):
        return self.grad(x, p)[1]

    def get_initial_pr(self, x, E, L_ang):
        t, r, th, ph = x
        args = (t, r, th, ph, self.M_val, self.a_val, self.lmd_val)
        g_inv = self.f_g_inv(*args)

        gUU_tt = g_inv[0,0]
        gUU_rr = g_inv[1,1]
        gUU_thth = g_inv[2,2]
        gUU_pp = g_inv[3,3]
        gUU_tp = g_inv[0,3]

        if abs(gUU_rr) < 1e-10:
            raise ValueError(f"g^rr too small ({gUU_rr}) at r={r}")

        residue = gUU_tt * E**2 + gUU_pp * L_ang**2 + 2 * gUU_tp * (-E) * L_ang
        pr_sq = -residue / gUU_rr

        if pr_sq < 0:
            raise ValueError(f"Forbidden region: pr^2={pr_sq} < 0 at r={r}, with E={E}, L={L_ang}")

        return -np.sqrt(pr_sq)

    def metric(self, x):
        """Covariant metric g_uv at the point x."""
        t, r, th, ph = x
        return np.array(_metric_function()(t, r, th, ph, self.M_val, self.a_val, self.lmd_val), dtype=float)

    def static_observer_angle(self, x, p):
        """Angle Psi (rad) between the photon direction and the outward radial direction.

        Measured by a static observer (4-velocity along d/dt) at x, as in the
        finite-distance deflection angle of Ishihara et al. (2016), used by
        Belhaj et al. (2022, Eq. 2.1) and Huang et al. (2025, Eq. 12). Psi is
        close to 0 for an outgoing photon and close to pi for an ingoing one.
        Equatorial motion is assumed (p_theta = 0). Returns NaN where no static
        observer exists (g_tt >= 0: ergoregion or beyond the static limit).
        """
        g = self.metric(x)
        g_tt, g_tp, g_rr, g_pp = g[0, 0], g[0, 3], g[1, 1], g[3, 3]
        if not g_tt < 0:
            return np.nan
        p_t, p_r, _, p_ph = p
        # Components in the static observer's orthonormal frame (the common factor 1/E_obs cancels)
        p_r_hat = p_r / np.sqrt(g_rr)
        p_ph_hat = (p_ph - g_tp / g_tt * p_t) / np.sqrt(g_pp - g_tp**2 / g_tt)
        # Measure the angle in the photon's sense of rotation (sign of L = p_phi)
        sense = 1.0 if p_ph >= 0 else -1.0
        return float(np.arctan2(sense * p_ph_hat, p_r_hat))


def kerr_horizon_radius(M, a):
    """Kerr event-horizon radius (Lambda = 0), used for plotting and the capture condition."""
    return M + np.sqrt(M**2 - a**2)
