"""Photon trajectory integration and parameter sweeps over Lambda and spin."""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .integrators import RungeKutta4
from .kerr_de_sitter import KerrDeSitter, _numeric_functions, kerr_horizon_radius


@dataclass
class TrajectoryResult:
    lmd: float
    X: np.ndarray        # r cos(phi) at each step
    Y: np.ndarray        # r sin(phi) at each step
    phi: np.ndarray
    H: np.ndarray        # Hamiltonian value at each step
    steps: int
    deflection_deg: float

    @property
    def r(self):
        return np.sqrt(self.X**2 + self.Y**2)


def trace_photon(sptm, x, p, r0, r_horizon, phi0=0.0, dtau=0.05, max_steps=30000):
    """Integrate with RK4 until the photon is captured or scattered back out."""
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

        # Record the Hamiltonian at every step
        hamiltonian_values.append(sptm.H(x, p))

        # Convert to Cartesian coordinates for plotting (projected)
        X_vals.append(r_curr * np.cos(phi_curr))
        Y_vals.append(r_curr * np.sin(phi_curr))
        Phi_vals.append(phi_curr)

        # Stop Conditions
        if r_curr < r_horizon * 1.1: # Captured by the black hole
            break
        if r_curr > r0 + 5 and steps > 10: # Scattered back out
            break

        steps += 1

    # Deflection: total absolute change in azimuth minus pi
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
    """Initial photon state (x, p); raises ValueError for invalid initial conditions."""
    x = np.array([0.0, r0, theta0, phi0])
    pr_init = sptm.get_initial_pr(x, E, L_ang)
    p = np.array([-E, pr_init, 0.0, L_ang])
    return x, p


def _run_one(job):
    # Top-level function so it can be sent to worker processes. None = invalid initial conditions.
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


def _run_jobs(jobs, n_jobs, verbose):
    """Run all jobs; return [(job, result)] in the order of `jobs`.

    Jobs with invalid initial conditions are skipped. Jobs are independent, so
    they run in parallel on `n_jobs` processes (default: number of cores);
    `n_jobs=1` runs everything in the current process.
    """
    # Derive/load the Hamiltonian in the main process first so the disk cache
    # exists before the workers start (workers then only load the cache file).
    _numeric_functions()

    if n_jobs is None:
        n_jobs = max(1, min(len(jobs), os.cpu_count() or 1))

    if verbose:
        print(table_header())

    if n_jobs == 1:
        outputs = map(_run_one, jobs)
        executor = None
    else:
        # "spawn" is safe inside a Jupyter kernel (fork in a multithreaded process can deadlock).
        executor = ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn"))
        outputs = executor.map(_run_one, jobs)

    done = []
    try:
        for job, res in zip(jobs, outputs):
            if res is None:
                if verbose:
                    print(f"Skipping L={job['lmd']}: Initial condition invalid")
                continue
            done.append((job, res))
            if verbose:
                print(table_row(job['a'], res), flush=True)
    finally:
        if executor is not None:
            executor.shutdown()

    return done


def run_lambda_sweep(M, a, lambdas, r0, theta0, phi0, E, L_ang, r_horizon,
                     dtau=0.05, max_steps=30000, n_jobs=None, verbose=True):
    """Compute the trajectory and deflection for each Lambda, in the order of `lambdas`.

    Each Lambda is independent, so they run in parallel on `n_jobs` processes
    (default: number of cores); `n_jobs=1` runs everything in this process.
    When called from a .py script (rather than a notebook), call it inside an
    `if __name__ == "__main__":` block (required by the "spawn" start method).
    """
    jobs = [dict(M=M, a=a, lmd=lmd, r0=r0, theta0=theta0, phi0=phi0, E=E, L_ang=L_ang,
                 r_horizon=r_horizon, dtau=dtau, max_steps=max_steps) for lmd in lambdas]
    return [res for _, res in _run_jobs(jobs, n_jobs, verbose)]


def run_spin_sweep(M, spins, lambdas, r0, theta0, phi0, E, L_ang,
                   dtau=0.05, max_steps=30000, n_jobs=None, verbose=True):
    """Compute the trajectory and deflection for every combination of spin `a` and Lambda.

    Returns a dict {(a, Lambda): TrajectoryResult}; combinations with invalid
    initial conditions are omitted. The horizon radius used for the capture
    condition is computed per spin with `kerr_horizon_radius(M, a)`. The whole
    grid runs in a single process pool (see `run_lambda_sweep` for `n_jobs` and
    for usage from .py scripts).
    """
    jobs = [dict(M=M, a=a, lmd=lmd, r0=r0, theta0=theta0, phi0=phi0, E=E, L_ang=L_ang,
                 r_horizon=kerr_horizon_radius(M, a), dtau=dtau, max_steps=max_steps)
            for a in spins for lmd in lambdas]
    return {(job['a'], job['lmd']): res for job, res in _run_jobs(jobs, n_jobs, verbose)}
