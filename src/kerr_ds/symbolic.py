"""Penurunan simbolik Hamiltonian geodesik di ruang-waktu Kerr-de Sitter.

M, a, dan Lambda diperlakukan sebagai simbol, sehingga ekspresi cukup diturunkan
SEKALI dan dipakai untuk semua nilai parameter. Karena `sp.simplify` tetap mahal
(~2 menit), hasilnya juga disimpan ke disk (pickle) dan dimuat ulang pada run
berikutnya. Nama file cache memuat hash dari kode derivasi + versi sympy, jadi
cache lama otomatis tidak dipakai bila metrik/derivasi diubah.

Lokasi cache: ~/.cache/kerr_ds (bisa diganti lewat env var KERR_DS_CACHE_DIR).
"""
import functools
import hashlib
import inspect
import os
import pickle
import warnings
from pathlib import Path

import sympy as sp

CACHE_DIR = Path(os.environ.get("KERR_DS_CACHE_DIR", Path.home() / ".cache" / "kerr_ds"))

# 1. Definisi Simbol
t, r, th, ph = sp.symbols('t r theta phi')
pt, pr, pth, pph = sp.symbols('pt pr ptheta pphi')
M, a, L = sp.symbols('M a Lambda')

COORDS = (t, r, th, ph)
MOMENTA = (pt, pr, pth, pph)
PARAMS = (M, a, L)


def metric():
    """Metrik kovarian g_uv Kerr-de Sitter (Boyer-Lindquist)."""
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
    """Invers metrik g^uv, eksak, memanfaatkan struktur blok.

    Hanya t dan phi yang terkopel, jadi invers = invers blok 2x2 (t, phi)
    ditambah 1/g_rr dan 1/g_thth. Jauh lebih cepat daripada `g.inv()` generik
    (~7 menit) dan menghasilkan ekspresi yang lebih ringkas.
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


def _cache_path():
    source = ''.join(inspect.getsource(f) for f in (metric, inverse_metric, _derive))
    key = hashlib.sha256((source + sp.__version__).encode()).hexdigest()[:16]
    return CACHE_DIR / f"hamiltonian_{key}.pkl"


@functools.cache
def derive_hamiltonian():
    """Ekspresi sympy {'H', 'dHdx', 'dHdp', 'g_inv'}; dari cache disk bila ada."""
    path = _cache_path()
    if path.exists():
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass  # cache rusak -> turunkan ulang

    exprs = _derive()
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(exprs, f)
        tmp.replace(path)
    except OSError as e:
        warnings.warn(f"Gagal menyimpan cache Hamiltonian ke {path}: {e}")
    return exprs
