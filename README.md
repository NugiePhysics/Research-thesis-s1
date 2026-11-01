# Light Deflection in Kerr–de Sitter Spacetime

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Numerical study of gravitational light bending around a rotating black hole in a universe with a
cosmological constant. Null geodesics are integrated in the equatorial plane of the
Kerr–de Sitter spacetime, using a symbolically derived Hamiltonian and a fourth-order Runge–Kutta
integrator. The resulting deflection angle, defined as in the literature by the angles measured by
static observers at the source and the observer, is analysed as a function of the cosmological
constant $\Lambda$ and the black-hole spin $a$, and compared with the analytic results of
Belhaj et al. (2022) and Huang et al. (2025).

This repository contains the research code for an undergraduate (S1) physics thesis.

📄 **Full methodology:** [docs/METHODOLOGY.md](docs/METHODOLOGY.md)

## Highlights

- **Symbolic Hamiltonian.** The metric, its exact inverse, the Hamiltonian, and all partial
  derivatives are derived once with SymPy, cached on disk, and compiled to fast numerical functions.
- **Parallel parameter sweeps** over $\Lambda$ and over a spin × $\Lambda$ grid on all CPU cores.
- **Deflection angle as in the literature.** The finite-distance definition of Ishihara et al. (2016),
  used by Belhaj et al. (2022) and Huang et al. (2025): $\delta = \Psi_O - \Psi_S + \phi_{OS}$, with
  angles measured by static observers. A positive $\Lambda$ reduces the deflection angle, as in these
  articles.
- **Validated.** The null constraint is monitored at every step
  ($\max\lvert H\rvert \approx 2\times10^{-11}$). The deflection angle agrees with an exact quadrature benchmark
  to $\lesssim 10^{-6}$ degrees for all spins and both signs of $\Lambda$, and with the analytic formulas of the
  articles in the weak field. Regression tests guarantee trajectories identical to the original
  research code.
- **Reproducible analysis.** Curve fits, their uncertainties, and $R^2$ are computed directly from the
  simulation output, with no manually transcribed data.

## Methodology at a glance

| Step | Description |
|---|---|
| Metric | Kerr–de Sitter in Boyer–Lindquist coordinates, geometrized units $G = c = 1$ with $M = 1$, and $\Delta_r = (r^2+a^2)(1 - \Lambda r^2/3) - 2Mr$. The inverse metric is computed exactly from the $t$–$\phi$ block. |
| Hamiltonian | $H = \tfrac12 g^{\mu\nu}p_\mu p_\nu$ with Hamilton's equations $dx^\mu/d\tau = \partial H/\partial p_\mu$, $dp_\mu/d\tau = -\partial H/\partial x^\mu$. $E = -p_t$ and $L = p_\phi$ are conserved exactly; the impact parameter is $b = L/E$. |
| Initial conditions | Equatorial photon at $x_0 = (0, r_0, \pi/2, 0)$ with $p_0 = (-E, p_r, 0, bE)$, where the ingoing $p_r < 0$ is fixed by the null constraint $H = 0$. |
| Integration | Classical RK4 with fixed step $\Delta\tau = 0.05$, at most 30,000 steps. |
| Termination | Captured if $r < 1.1\,r_+$ (Kerr horizon); scattered if $r > r_0 + 5$. |
| Deflection angle | $\delta = \Psi_O - \Psi_S + \phi_{OS}$ in degrees, where $\phi_{OS} = \phi_\text{final} - \phi_0$ and $\Psi_S$, $\Psi_O$ are the angles between the ray and the outward radial direction measured by static observers at the start (source) and end (observer) of the trajectory. The coordinate value $\phi_{OS} - \pi$ and the legacy value $\lvert\phi_{OS} - \pi\rvert$ of the original code are also stored. |
| Validation | Hamiltonian constraint, exact quadrature benchmark, flat space, literature values (Hasanuddin 2012), analytic formulas (Huang et al. 2025; Belhaj et al. 2022), regression and unit tests. |
| Curve fitting | $\delta = A\lvert\Lambda\rvert^B + C$ (each sign of $\Lambda$) and, for each $\Lambda$, $\delta(a) = A e^{-Ba} + C$ (prograde and retrograde), via `scipy.optimize.curve_fit`, with parameter uncertainties; fit quality $R^2 = 1 - SS_\text{res}/SS_\text{tot}$. |

**Default parameters:** $M = E = 1$, $b = 7$, $r_0 = 500$,
$\Lambda \in \{0,\ \pm10^{-8},\ \pm5\times10^{-8},\ \pm10^{-7},\ \pm5\times10^{-7},\ \pm10^{-6},\ \pm5\times10^{-6}\}$
and $a = 0.99$ (main notebook), or $a \in \{0, \pm0.2, \pm0.4, \pm0.6, \pm0.8, \pm0.99\}$ and $\Lambda \ge 0$
(spin notebook).

> [!NOTE]
> The deflection angle is evaluated at finite distances ($r_0 = 500 \to r \approx 505$), as in the
> articles it is compared with. The original thesis tables used the coordinate value
> $\lvert\phi_{OS} - \pi\rvert$, which responds to $\Lambda$ with the opposite sign; see
> [Section 8](docs/METHODOLOGY.md#8-deflection-angle) and
> [Limitations](docs/METHODOLOGY.md#13-limitations).

## Example output

Deflection angles for $a = 0.99$, $b = 7$ from the main notebook ($\Lambda \ge 0$ shown):

```
a          | Lambda     | Deflection (deg)   | dphi - pi (deg)    | Status
-------------------------------------------------------------------------------
0.99       | 0.0        | 45.508988          | 43.914885          | Steps: 20140
0.99       | 1e-08      | 45.508710          | 43.915089          | Steps: 20140
0.99       | 5e-08      | 45.507596          | 43.915907          | Steps: 20140
0.99       | 1e-07      | 45.506203          | 43.916931          | Steps: 20140
0.99       | 5e-07      | 45.495051          | 43.925228          | Steps: 20140
0.99       | 1e-06      | 45.481094          | 43.935865          | Steps: 20140
0.99       | 5e-06      | 45.372598          | 44.035493          | Steps: 20140
```

Change of the deflection angle with $\Lambda$, $\delta(\Lambda) - \delta(0)$ in degrees:

| $\Lambda$ | Simulation | Exact (quadrature) | Huang et al. (2025), Eq. 24 | Belhaj et al. (2022), Eq. 5.1 |
|---|---|---|---|---|
| $-10^{-6}$ | +0.02775 | +0.02775 | +0.02764 | +0.02905 |
| $+10^{-6}$ | −0.02789 | −0.02789 | −0.03047 | −0.02905 |
| $+5\times10^{-6}$ | −0.13639 | −0.13639 | −0.18062 | −0.14525 |

Power-law fit ($\Lambda > 0$): $\delta = -23250.55\,\Lambda^{0.9869 \pm 0.0015} + 45.5090$ with $R^2 = 0.999999$.

## Final report

📑 **[Laporan akhir (PDF, in Indonesian)](docs/laporan_akhir/laporan_akhir.pdf)** covers 11 spins ×
13 values of $\Lambda$ ($b = 7M$, $r_0 = 500M$). Its main results:

- A positive $\Lambda$ **reduces** the deflection angle and a negative $\Lambda$ increases it, as in
  Rindler & Ishak (2007), Belhaj et al. (2022), and Huang et al. (2025). At $\Lambda = 10^{-6}$ the change
  agrees with the analytic formulas of Belhaj et al. and Huang et al. to within 10%.
- The reduction is weaker for prograde and stronger for retrograde spins, as predicted by the $a\Lambda$
  term of these formulas.
- For the extreme retrograde spin $a = -0.99$, the photon passes just $0.017M$ outside its critical impact
  parameter. There the shift of $b_c$ with $\Lambda$ dominates and reverses the trend (strong-field limit,
  Bozza 2002).
- The simulation agrees with an exact quadrature benchmark to $5.5\times10^{-7}$ degrees.

The report is built from [`docs/laporan_akhir/laporan_akhir.tex`](docs/laporan_akhir/laporan_akhir.tex).
Its data, figures, and tables are produced by
[`docs/laporan_akhir/generate_results.py`](docs/laporan_akhir/generate_results.py).

## Versions

| Version | Deflection angle | Use |
|---|---|---|
| Tag [`skripsi-v1.0`](https://github.com/NugiePhysics/Research-thesis-s1/tree/skripsi-v1.0) | Coordinate value $\lvert\phi_{OS} - \pi\rvert$ | Code behind the tables of the thesis manuscript |
| `main` | Finite-distance angle of the articles, $\delta = \Psi_O - \Psi_S + \phi_{OS}$ | Current results and the final report |

The two definitions respond to $\Lambda$ with opposite signs; see
[Section 8 of the methodology](docs/METHODOLOGY.md#8-deflection-angle). The trajectories are identical in
both versions (regression test), and the current code still provides the original value as
`deflection_legacy_deg` and the original table as `format_legacy_table`.

## Repository structure

```
Research-thesis-S1/
├── src/kerr_ds/                 # physics package
│   ├── base.py                  # abstract Spacetime and Integrator interfaces
│   ├── integrators.py           # RungeKutta4
│   ├── symbolic.py              # metric, inverse metric, Hamiltonian and derivatives (SymPy, cached)
│   ├── kerr_de_sitter.py        # KerrDeSitter: numerical H, dH/dx, dH/dp, initial conditions
│   ├── simulation.py            # photon integration, deflection angles, Λ sweep, spin × Λ grid sweep
│   ├── analytic.py              # exact benchmark (quadrature), b_c, formulas of Huang 2025 and Belhaj 2022
│   ├── fitting.py               # fit models (A·Λ^B + C, A·Λ + C, A·e^(−Ba) + C), uncertainties and R²
│   └── plotting.py              # trajectory, Hamiltonian-conservation, comparison, and fit plots
├── notebooks/
│   ├── Lambda_Deflection.ipynb  # main study: simulation → table → plots → comparison with articles → fits
│   ├── Spin_Deflection.ipynb    # deflection vs spin (prograde and retrograde) for each Λ → exponential fits
│   └── Validation.ipynb         # literature values (Λ = 0) and exact benchmark (Λ ≠ 0)
├── docs/
│   ├── METHODOLOGY.md           # full methodology
│   └── laporan_akhir/           # final report in LaTeX (Indonesian), with the script that generates its data and figures
├── tests/                       # regression, deflection-angle, analytic and fitting tests
└── pyproject.toml
```

## Installation

Requires Python 3.10 or newer. Using [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

or with plain `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

When opening the notebooks in VS Code or Jupyter, select the `.venv` kernel.

## Usage

### Notebooks

| Notebook | Content | Run time* |
|---|---|---|
| [`notebooks/Lambda_Deflection.ipynb`](notebooks/Lambda_Deflection.ipynb) | Deflection versus $\Lambda$ (both signs) for fixed spin: trajectories, Hamiltonian-conservation plots, comparison with the articles, and fits | ~1 min |
| [`notebooks/Spin_Deflection.ipynb`](notebooks/Spin_Deflection.ipynb) | Deflection versus spin (prograde and retrograde) for each $\Lambda$ (77 trajectories) and exponential fits | ~5 min |
| [`notebooks/Validation.ipynb`](notebooks/Validation.ipynb) | Comparison with literature values ($\Lambda = 0$) and with the exact benchmark ($\Lambda \ne 0$) | ~2.5 min |

\*On a 4-core laptop CPU, after the Hamiltonian cache has been created. The first run on a new
machine also derives the Hamiltonian symbolically and caches it. This usually takes about a minute,
but the runtime of SymPy's `simplify` varies between Python processes (hash randomization) and can
exceed 10 minutes; the derived expression is identical either way.

Open a notebook and select **Run All**. All physical and numerical parameters are defined in the
*Parameters* cell.

### Python API

```python
import numpy as np
from kerr_ds import kerr_horizon_radius, run_lambda_sweep

if __name__ == "__main__":  # required for multiprocessing ("spawn") in scripts
    M, a, b, E = 1.0, 0.99, 7.0, 1.0
    results = run_lambda_sweep(
        M, a, lambdas=[0.0, 1e-6, 5e-6],
        r0=500.0, theta0=np.pi / 2, phi0=0.0, E=E, L_ang=b * E,
        r_horizon=kerr_horizon_radius(M, a),
    )
    for res in results:
        print(res.lmd, res.deflection_deg, res.deflection_coord_deg)
```

Each `TrajectoryResult` contains the deflection angle (`deflection_deg`, definition of the articles),
the coordinate value $\phi_{OS} - \pi$ (`deflection_coord_deg`), the original value $\lvert\phi_{OS} - \pi\rvert$
(`deflection_legacy_deg`), the static-observer angles (`psi_source`, `psi_observer`), the step count,
the projected trajectory (`X`, `Y`, `r`), the azimuth `phi`, and the Hamiltonian at every step (`H`).
`run_spin_sweep` runs a full spin × $\Lambda$ grid in a single process pool and returns a dictionary
keyed by `(a, Lambda)`. `format_table` prints the new table and `format_legacy_table` the table of the
original code.

The `kerr_ds.analytic` module provides the exact deflection angle by quadrature
(`deflection_exact_deg`, same definition and radii as the simulation), the asymptotic Kerr angle, the
critical impact parameter, and the formulas of the articles:

```python
from kerr_ds import analytic

r_obs = results[-1].r[-1]  # observer radius = end point of the trajectory
analytic.deflection_exact_deg(M, a, 5e-6, b, 500.0, r_obs)       # exact benchmark
analytic.huang2025_deflection_deg(M, a, 5e-6, b, 500.0, r_obs)   # Huang et al. (2025), Eq. (24)
analytic.belhaj2022_deflection_deg(M, a, 5e-6, b, 500.0, r_obs)  # Belhaj et al. (2022), Eq. (5.1)
```

### Tests

```bash
pytest
```

The test suite checks that the trajectories, step counts, Hamiltonian values, and legacy table are
identical to those of the original research code; that the deflection angle vanishes in flat space
and matches the exact benchmark; that it decreases for $\Lambda > 0$ and agrees with the formulas of
Huang et al. and Belhaj et al. in the weak field; that the critical impact parameter matches Congdon
et al. (2022); that the exact inverse metric matches a numerical inverse; and that the fitting
utilities and spin sweep behave as expected.

## Performance

The original notebook (7 values of $\Lambda$) took over an hour to run; the same computation now takes
about 30 seconds. The main causes and fixes:

| Cause | Fix |
|---|---|
| Symbolic derivation (inverse metric + `simplify`, ~9 min) repeated for **every** $\Lambda$, although $\Lambda$ was already a symbol | Derived **once** and cached on disk |
| Generic `Matrix.inv()` on the 4×4 metric (~7 min, large expressions) | Exact inverse exploiting the block structure (only $t$ and $\phi$ are coupled) |
| `dH/dx` and `dH/dp` evaluated separately with NumPy on scalars | Single combined function (`lambdify(..., 'math', cse=True)`) |
| Values of $\Lambda$ run sequentially | Run in parallel on all CPU cores |

The Hamiltonian cache is stored in `~/.cache/kerr_ds/` (override with the `KERR_DS_CACHE_DIR`
environment variable). It is invalidated automatically when `symbolic.py` or the SymPy version
changes, and is safe to delete.

## License

The code is released under the [MIT License](LICENSE). The final report, its figures, tables, and data
in [`docs/laporan_akhir/`](docs/laporan_akhir/) are licensed under
[CC BY 4.0](docs/laporan_akhir/LICENSE).

## Author

**NugiePhysics** (Nugie Saputra) — undergraduate (S1) thesis in physics, Universitas Tanjungpura.
