"""Integrasi lintasan foton dan sapuan (sweep) nilai Lambda."""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .integrators import RungeKutta4
from .kerr_de_sitter import KerrDeSitter, _numeric_functions


@dataclass
class TrajectoryResult:
    lmd: float
    X: np.ndarray        # r cos(phi) tiap langkah
    Y: np.ndarray        # r sin(phi) tiap langkah
    phi: np.ndarray
    H: np.ndarray        # nilai Hamiltonian tiap langkah
    steps: int
    deflection_deg: float

    @property
    def r(self):
        return np.sqrt(self.X**2 + self.Y**2)


def trace_photon(sptm, x, p, r0, r_horizon, phi0=0.0, dtau=0.05, max_steps=30000):
    """Integrasi RK4 sampai foton jatuh ke horizon atau kembali keluar (scattering)."""
    intg = RungeKutta4(sptm)

    X_vals = []
    Y_vals = []
    Phi_vals = []
    hamiltonian_values = []

    steps = 0
    while steps < max_steps:
        x, p = intg.step(x, p, dtau)

        r_curr = x[1]
        phi_curr = x[3]

        # Simpan nilai Hamiltonian di setiap langkah
        hamiltonian_values.append(sptm.H(x, p))

        # Konversi ke Cartesian untuk plot (projected)
        X_vals.append(r_curr * np.cos(phi_curr))
        Y_vals.append(r_curr * np.sin(phi_curr))
        Phi_vals.append(phi_curr)

        # Stop Conditions
        if r_curr < r_horizon * 1.1: # Jatuh ke lubang hitam
            break
        if r_curr > r0 + 5 and steps > 10: # Keluar kembali (scattering)
            break

        steps += 1

    # Analisis Defleksi: total perubahan sudut mutlak dikurangi pi
    deflection = abs(abs(Phi_vals[-1] - phi0) - np.pi)

    return TrajectoryResult(
        lmd=sptm.lmd_val,
        X=np.array(X_vals),
        Y=np.array(Y_vals),
        phi=np.array(Phi_vals),
        H=np.array(hamiltonian_values),
        steps=steps,
        deflection_deg=np.degrees(deflection),
    )


def initial_state(sptm, r0, theta0, phi0, E, L_ang):
    """State awal (x, p) foton; ValueError bila kondisi awal tidak valid."""
    x = np.array([0.0, r0, theta0, phi0])
    pr_init = sptm.get_initial_pr(x, E, L_ang)
    p = np.array([-E, pr_init, 0.0, L_ang])
    return x, p


def _run_one(job):
    # Fungsi top-level agar bisa dikirim ke worker process. None = kondisi awal invalid.
    sptm = KerrDeSitter(job['M'], job['a'], job['lmd'])
    try:
        x, p = initial_state(sptm, job['r0'], job['theta0'], job['phi0'], job['E'], job['L_ang'])
    except ValueError:
        return None
    return trace_photon(sptm, x, p, job['r0'], job['r_horizon'], phi0=job['phi0'],
                        dtau=job['dtau'], max_steps=job['max_steps'])


def table_header():
    return f"{'a':<10} | {'Lambda':<10} | {'Deflection (deg)':<20} | {'Status'}\n" + "-" * 55


def table_row(a, res):
    return f"{a:<10} | {res.lmd:<10} | {res.deflection_deg:<20.4f} | Steps: {res.steps}"


def format_table(a, results):
    return "\n".join([table_header()] + [table_row(a, res) for res in results]) + "\n"


def run_lambda_sweep(M, a, lambdas, r0, theta0, phi0, E, L_ang, r_horizon,
                     dtau=0.05, max_steps=30000, n_jobs=None, verbose=True):
    """Hitung lintasan & defleksi untuk setiap Lambda; urutan hasil mengikuti `lambdas`.

    Setiap Lambda independen, jadi dijalankan paralel di `n_jobs` proses
    (default: jumlah core). `n_jobs=1` menjalankan semuanya di proses ini.
    Bila dipanggil dari script .py (bukan notebook), panggil di dalam blok
    `if __name__ == "__main__":` (syarat multiprocessing mode "spawn").
    """
    jobs = [dict(M=M, a=a, lmd=lmd, r0=r0, theta0=theta0, phi0=phi0, E=E, L_ang=L_ang,
                 r_horizon=r_horizon, dtau=dtau, max_steps=max_steps) for lmd in lambdas]

    # Turunkan/muat Hamiltonian di proses utama dulu supaya cache disk sudah ada
    # sebelum worker dibuat (worker cukup memuat file cache).
    _numeric_functions()

    if n_jobs is None:
        n_jobs = max(1, min(len(jobs), os.cpu_count() or 1))

    if verbose:
        print(table_header())

    if n_jobs == 1:
        outputs = map(_run_one, jobs)
        executor = None
    else:
        # "spawn" aman dipakai dari kernel Jupyter (fork pada proses multi-thread bisa deadlock).
        executor = ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn"))
        outputs = executor.map(_run_one, jobs)

    results = []
    try:
        for job, res in zip(jobs, outputs):
            if res is None:
                if verbose:
                    print(f"Skipping L={job['lmd']}: Initial condition invalid")
                continue
            results.append(res)
            if verbose:
                print(table_row(a, res), flush=True)
    finally:
        if executor is not None:
            executor.shutdown()

    return results
