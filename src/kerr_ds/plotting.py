"""Plots of photon trajectories, Hamiltonian conservation, and deflection fits."""
import matplotlib.pyplot as plt
import numpy as np

from .fitting import exponential_decay, power_law

COLORS = ['black', 'blue', 'red', 'green', 'orange', 'purple', 'brown']


def plot_trajectories(results, M, a, b, rh, limit=10, save_path=None):
    fig, ax = plt.subplots(figsize=(8,8))

    # Event horizon
    eventhorizon = plt.Circle((0,0), rh, color='gray', alpha=0.3, label="Horizon (Kerr)")
    ax.add_artist(eventhorizon)
    ax.scatter(0, 0, s=20, c='k', label='Singularity')

    # Computed trajectories
    for i, res in enumerate(results):
        label_txt = f'$\\Lambda = {res.lmd}$'
        # Cycle through the colors if there are more Lambda values than colors
        ax.plot(res.X, res.Y, color=COLORS[i % len(COLORS)], label=label_txt, linewidth=1.5)

    # Formatting Plot
    ax.set_title(f"M={M}, a={a}, b={b}", fontsize=14)
    ax.set_xlabel(r'$X = r \cos(\phi)$')
    ax.set_ylabel(r'$Y = r \sin(\phi)$')
    ax.set_aspect('equal')

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    _save(fig, save_path)
    return fig, ax


def plot_hamiltonian_vs_step(res, save_path=None):
    """Hamiltonian conservation check versus integration step."""
    steps_arr = np.arange(len(res.H))
    hamiltonian_arr = res.H

    # Fluctuation (max - min)
    H_max = np.max(hamiltonian_arr)
    H_min = np.min(hamiltonian_arr)
    fluctuation = H_max - H_min

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps_arr, hamiltonian_arr, color='purple', label='Hamiltonian $H$')

    # Symmetric log scale on the y axis if the fluctuation is very small
    if fluctuation > 0 and fluctuation < 1e-3:
        # symlog handles small variations around both negative and positive values
        ax.set_yscale('symlog', linthresh=0.001)

    ax.set_title("Hamiltonian Conservation During Integration (Last Trajectory)")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Hamiltonian $H$", fontsize=12)

    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend()

    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_hamiltonian_vs_r(res, save_path=None):
    """Hamiltonian conservation check versus radial coordinate r."""
    r_vals = res.r
    H_vals = res.H

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot H_vals vs r_vals
    ax.plot(r_vals, H_vals, color='teal', linewidth=1.5, label=r'Hamiltonian $\mathcal{H}$')

    # Dashed reference line at the ideal value H = 0
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label=r'Ideal $\mathcal{H} = 0$')

    # symlog scale so that tiny fluctuations (1e-14 to 1e-11) are visible;
    # H fluctuates slightly around 0 in both directions
    ax.set_yscale('symlog', linthresh=1e-14)

    # Periapsis (smallest r)
    min_r_idx = np.argmin(r_vals)
    min_r = r_vals[min_r_idx]
    H_at_min_r = H_vals[min_r_idx]

    # Annotate the periapsis
    ax.annotate(f'Periapsis\n$r \\approx {min_r:.2f}$\n$\\mathcal{{H}} \\approx {H_at_min_r:.2e}$',
                xy=(min_r, H_at_min_r),
                xytext=(min_r + max(r_vals)*0.1, H_at_min_r + 1e-12),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Title and axis labels
    ax.set_title("Hamiltonian Conservation vs Radial Coordinate ($r$)", fontsize=14, pad=15)
    ax.set_xlabel("Radial coordinate ($r$)", fontsize=12)
    ax.set_ylabel(r"Hamiltonian ($\mathcal{H}$)", fontsize=12)

    # Grid and legend
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_lambda_fit(lambdas, deflections, popt, r2, save_path=None):
    """Deflection versus Lambda with the fitted curve A * Lambda^B + C (log x axis)."""
    lambdas = np.asarray(lambdas)
    A, B, C = popt
    lmd_fit = np.logspace(np.log10(lambdas.min()), np.log10(lambdas.max()), 1000)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(lambdas, deflections, color='red', s=60, zorder=5, label='Simulation data')
    label_fitting = rf'Fit: $y = {A:.2f} \cdot \Lambda^{{{B:.4f}}} + {C:.4f}$ ($R^2 = {r2:.4f}$)'
    ax.plot(lmd_fit, power_law(lmd_fit, *popt), color='blue', linestyle='-', linewidth=2, label=label_fitting)
    ax.set_xscale('log')

    ax.set_xlabel(r'Cosmological constant $\Lambda$ ($M^{-2}$)', fontsize=12)
    ax.set_ylabel('Deflection angle (deg)', fontsize=12)
    ax.legend(loc='upper left')
    _save(fig, save_path)
    return fig, ax


def plot_spin_fit(spins, deflections, fits, save_path=None):
    """Deflection versus spin a for each Lambda, with fitted curves A * exp(-B a) + C.

    `deflections` = {Lambda: deflection for each spin}, `fits` = {Lambda: (popt, R^2)}.
    """
    x_fit = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(11, 7))
    for lmd, y_vals in deflections.items():
        popt, r2 = fits[lmd]
        label = f'Λ = {lmd:g}'
        p = ax.plot(spins, y_vals, 'o', label=f'Data {label}')
        formula_label = (f"Fit {label}: y = {popt[0]:.2f} * e^(-{popt[1]:.2f} * x) + {popt[2]:.2f} "
                         f"($R^2$={r2:.4f})")
        ax.plot(x_fit, exponential_decay(x_fit, *popt), linestyle='--', color=p[0].get_color(),
                label=formula_label)

    ax.set_xlabel('Spin (a)')
    ax.set_ylabel('Deflection angle (deg)')
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def _save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
