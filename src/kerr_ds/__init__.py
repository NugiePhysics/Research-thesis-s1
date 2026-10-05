"""Light deflection in Kerr-de Sitter spacetime (undergraduate thesis research)."""
from .base import Integrator, Spacetime
from .fitting import exponential_decay, fit_curve, power_law, r_squared
from .integrators import RungeKutta4
from .kerr_de_sitter import KerrDeSitter, kerr_horizon_radius
from .plotting import (
    plot_hamiltonian_vs_r,
    plot_hamiltonian_vs_step,
    plot_lambda_fit,
    plot_spin_fit,
    plot_trajectories,
)
from .simulation import (
    TrajectoryResult,
    format_table,
    initial_state,
    run_lambda_sweep,
    run_spin_sweep,
    trace_photon,
)

__all__ = [
    "Integrator",
    "KerrDeSitter",
    "RungeKutta4",
    "Spacetime",
    "TrajectoryResult",
    "exponential_decay",
    "fit_curve",
    "format_table",
    "initial_state",
    "kerr_horizon_radius",
    "plot_hamiltonian_vs_r",
    "plot_hamiltonian_vs_step",
    "plot_lambda_fit",
    "plot_spin_fit",
    "plot_trajectories",
    "power_law",
    "r_squared",
    "run_lambda_sweep",
    "run_spin_sweep",
    "trace_photon",
]
