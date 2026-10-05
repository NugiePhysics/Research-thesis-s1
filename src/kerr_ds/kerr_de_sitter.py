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


def kerr_horizon_radius(M, a):
    """Kerr event-horizon radius (Lambda = 0), used for plotting and the capture condition."""
    return M + np.sqrt(M**2 - a**2)
