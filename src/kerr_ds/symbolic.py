"""Symbolic derivation of the geodesic Hamiltonian in Kerr-de Sitter spacetime.

M, a, and Lambda are treated as symbols, so the expressions only need to be
derived ONCE and can be reused for every parameter value. Because `sp.simplify`
is still expensive (about a minute, sometimes much longer depending on Python's
hash randomization), the result is also stored on disk (pickle) and
reloaded on subsequent runs. The cache file name contains a hash of the
derivation code (excluding docstrings and comments) and the SymPy version, so a
stale cache is ignored automatically whenever the metric or the derivation
changes.

Cache location: ~/.cache/kerr_ds (override with the KERR_DS_CACHE_DIR env var).
"""
import ast
import functools
import hashlib
import inspect
import os
import pickle
import textwrap
import warnings
from pathlib import Path

import sympy as sp

CACHE_DIR = Path(os.environ.get("KERR_DS_CACHE_DIR", Path.home() / ".cache" / "kerr_ds"))

# Symbols
t, r, th, ph = sp.symbols('t r theta phi')
pt, pr, pth, pph = sp.symbols('pt pr ptheta pphi')
M, a, L = sp.symbols('M a Lambda')

COORDS = (t, r, th, ph)
MOMENTA = (pt, pr, pth, pph)
PARAMS = (M, a, L)


def metric():
    """Covariant Kerr-de Sitter metric g_uv (Boyer-Lindquist coordinates)."""
    Sigma = r**2 + (a * sp.cos(th))**2
    Delta_r = (r**2 + a**2) * (1 - (L * r**2)/3) - 2*M*r
    Delta_th = 1 + (L * a**2 * sp.cos(th)**2)/3
    Xi = 1 + (L * a**2)/3

    g_tt = - (Delta_r / Sigma) + (Delta_th * sp.sin(th)**2 / Sigma) * a**2
    g_rr = Sigma / Delta_r
    g_thth = Sigma / Delta_th
    g_pp = -(Delta_r/Sigma) * (a * sp.sin(th)**2 / Xi)**2 + \
           (Delta_th * sp.sin(th)**2 / Sigma) * ((r**2 + a**2)/Xi)**2
    g_tp = -(Delta_r/Sigma) * (-a * sp.sin(th)**2 / Xi) + \
           (Delta_th * sp.sin(th)**2 / Sigma) * (-a * (r**2 + a**2)/Xi)

    return sp.Matrix([
        [g_tt, 0, 0, g_tp],
        [0, g_rr, 0, 0],
        [0, 0, g_thth, 0],
        [g_tp, 0, 0, g_pp]
    ])


def inverse_metric(g):
    """Exact inverse metric g^uv, exploiting the block structure.

    Only t and phi are coupled, so the inverse is the inverse of the 2x2 (t, phi)
    block plus 1/g_rr and 1/g_thth. Much faster than a generic `g.inv()`
    (~7 minutes) and yields more compact expressions.
    """
    g_tt, g_tp, g_pp = g[0, 0], g[0, 3], g[3, 3]
    D = g_tt * g_pp - g_tp**2

    g_inv = sp.zeros(4, 4)
    g_inv[0, 0] = g_pp / D
    g_inv[0, 3] = g_inv[3, 0] = -g_tp / D
    g_inv[3, 3] = g_tt / D
    g_inv[1, 1] = 1 / g[1, 1]
    g_inv[2, 2] = 1 / g[2, 2]
    return g_inv


def _derive():
    g_inv = inverse_metric(metric())

    # Hamiltonian H = 1/2 g^uv p_u p_v
    p_vec = sp.Matrix(MOMENTA)
    Ham = sp.simplify(0.5 * (p_vec.T * g_inv * p_vec)[0])

    return {
        'H': Ham,
        'dHdx': [sp.diff(Ham, v) for v in COORDS],
        'dHdp': [sp.diff(Ham, v) for v in MOMENTA],
        'g_inv': g_inv,
    }


def _code_fingerprint(func):
    """AST dump of `func` without its docstring (comments are not part of the AST)."""
    node = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        node.body = node.body[1:]
    return ast.dump(node)


def _cache_path():
    # Hash only the code, so editing docstrings or comments does not trigger a
    # (slow) re-derivation.
    code = ''.join(_code_fingerprint(f) for f in (metric, inverse_metric, _derive))
    key = hashlib.sha256((code + sp.__version__).encode()).hexdigest()[:16]
    return CACHE_DIR / f"hamiltonian_{key}.pkl"


@functools.cache
def derive_hamiltonian():
    """SymPy expressions {'H', 'dHdx', 'dHdp', 'g_inv'}, loaded from the disk cache if available."""
    path = _cache_path()
    if path.exists():
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass  # corrupt cache -> derive again

    exprs = _derive()
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(exprs, f)
        tmp.replace(path)
    except OSError as e:
        warnings.warn(f"Could not save the Hamiltonian cache to {path}: {e}")
    return exprs
