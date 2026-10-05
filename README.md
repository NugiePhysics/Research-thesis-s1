# Light Deflection in Kerr–de Sitter Spacetime

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Numerical study of gravitational light bending around a rotating black hole in a universe with a
positive cosmological constant. Null geodesics are integrated in the equatorial plane of the
Kerr–de Sitter spacetime, using a symbolically derived Hamiltonian and a fourth-order Runge–Kutta
integrator, and the resulting deflection angle is analysed as a function of the cosmological
constant $\Lambda$ and the black-hole spin $a$.

This repository contains the research code for an undergraduate (S1) physics thesis.

📄 **Full methodology:** [docs/METHODOLOGY.md](docs/METHODOLOGY.md)

## Highlights

- **Symbolic Hamiltonian.** The metric, its exact inverse, the Hamiltonian, and all partial
  derivatives are derived once with SymPy, cached on disk, and compiled to fast numerical functions.
- **Parallel parameter sweeps** over $\Lambda$ and over a spin × $\Lambda$ grid on all CPU cores.
- **Validated.** The null constraint is monitored at every step
  ($\max\lvert H\rvert \approx 2\times10^{-11}$), results agree with an analytic Schwarzschild benchmark to
  four decimal places, and regression tests guarantee output identical to the original research code.
- **Reproducible analysis.** Curve fits and their $R^2$ are computed directly from the simulation
  output, with no manually transcribed data.

## Methodology at a glance

| Step | Description |
|---|---|
| Metric | Kerr–de Sitter in Boyer–Lindquist coordinates, geometrized units $G = c = 1$ with $M = 1$, and $\Delta_r = (r^2+a^2)(1 - \Lambda r^2/3) - 2Mr$. The inverse metric is computed exactly from the $t$–$\phi$ block. |
| Hamiltonian | $H = \tfrac12 g^{\mu\nu}p_\mu p_\nu$ with Hamilton's equations $dx^\mu/d\tau = \partial H/\partial p_\mu$, $dp_\mu/d\tau = -\partial H/\partial x^\mu$. $E = -p_t$ and $L = p_\phi$ are conserved exactly; the impact parameter is $b = L/E$. |
| Initial conditions | Equatorial photon at $x_0 = (0, r_0, \pi/2, 0)$ with $p_0 = (-E, p_r, 0, bE)$, where the ingoing $p_r < 0$ is fixed by the null constraint $H = 0$. |
| Integration | Classical RK4 with fixed step $\Delta\tau = 0.05$, at most 30,000 steps. |
| Termination | Captured if $r < 1.1\,r_+$ (Kerr horizon); scattered if $r > r_0 + 5$. |
| Deflection angle | $\delta = \bigl\lvert\, \lvert\phi_\text{final} - \phi_0\rvert - \pi \,\bigr\rvert$, in degrees. |
| Validation | Hamiltonian constraint, analytic Schwarzschild and flat-space benchmarks, regression and unit tests. |
| Curve fitting | $\delta(\Lambda) = A\Lambda^B + C$ and, for each $\Lambda$, $\delta(a) = A e^{-Ba} + C$, via `scipy.optimize.curve_fit`; fit quality $R^2 = 1 - SS_\text{res}/SS_\text{tot}$. |

**Default parameters:** $M = E = 1$, $b = 7$, $r_0 = 500$,
$\Lambda \in \{0,\ 10^{-8},\ 5\times10^{-8},\ 10^{-7},\ 5\times10^{-7},\ 10^{-6},\ 5\times10^{-6}\}$,
$a = 0.99$ (main notebook) or $a \in \{0, 0.2, 0.4, 0.6, 0.8, 0.99\}$ (spin notebook).

> [!NOTE]
> The deflection angle is evaluated at a finite radius ($r_0 = 500 \to r \approx 505$), not
> asymptotically. See [Limitations](docs/METHODOLOGY.md#13-limitations) for the implications.

## Example output

Deflection angles for $a = 0.99$, $b = 7$ from the main notebook:

```
a          | Lambda     | Deflection (deg)     | Status
-------------------------------------------------------
0.99       | 0.0        | 43.9149              | Steps: 20140
0.99       | 1e-08      | 43.9151              | Steps: 20140
0.99       | 5e-08      | 43.9159              | Steps: 20140
0.99       | 1e-07      | 43.9169              | Steps: 20140
0.99       | 5e-07      | 43.9252              | Steps: 20140
0.99       | 1e-06      | 43.9359              | Steps: 20140
0.99       | 5e-06      | 44.0355              | Steps: 20140
```

Power-law fit: $\delta(\Lambda) = 67445.75\,\Lambda^{1.0844} + 43.9150$ with $R^2 = 0.999985$.

## Repository structure

```
Research-thesis-S1/
├── src/kerr_ds/                 # physics package
│   ├── base.py                  # abstract Spacetime and Integrator interfaces
│   ├── integrators.py           # RungeKutta4
│   ├── symbolic.py              # metric, inverse metric, Hamiltonian and derivatives (SymPy, cached)
│   ├── kerr_de_sitter.py        # KerrDeSitter: numerical H, dH/dx, dH/dp, initial conditions
│   ├── simulation.py            # photon integration, Λ sweep, spin × Λ grid sweep
│   ├── fitting.py               # fit models (A·Λ^B + C, A·e^(−Ba) + C) and R²
│   └── plotting.py              # trajectory, Hamiltonian-conservation, and fit plots
├── notebooks/
│   ├── Lambda_Deflection.ipynb  # main study: parameters → simulation → table → plots → fit vs Λ
│   └── Spin_Deflection.ipynb    # deflection vs spin for each Λ → exponential fits
├── docs/
│   └── METHODOLOGY.md           # full methodology
├── tests/                       # regression tests against the original code, fitting tests
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
| [`notebooks/Lambda_Deflection.ipynb`](notebooks/Lambda_Deflection.ipynb) | Deflection versus $\Lambda$ for fixed spin: trajectories, Hamiltonian-conservation plots, and power-law fit | ~35 s |
| [`notebooks/Spin_Deflection.ipynb`](notebooks/Spin_Deflection.ipynb) | Deflection versus spin for each $\Lambda$ (42 trajectories) and exponential fits | ~2.5 min |

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
        print(res.lmd, res.deflection_deg)
```

Each `TrajectoryResult` contains the deflection angle (`deflection_deg`), the step count, the
projected trajectory (`X`, `Y`, `r`), the azimuth `phi`, and the Hamiltonian at every step (`H`).
`run_spin_sweep` runs a full spin × $\Lambda$ grid in a single process pool and returns a dictionary
keyed by `(a, Lambda)`.

### Tests

```bash
pytest
```

The test suite checks that the deflection table, step counts, trajectories, and Hamiltonian values
are identical to those of the original research code, that the exact inverse metric matches a
numerical inverse, and that the fitting utilities and spin sweep behave as expected.

## Performance

The original notebook took over an hour to run; it now takes about 30 seconds. The main causes and
fixes:

| Cause | Fix |
|---|---|
| Symbolic derivation (inverse metric + `simplify`, ~9 min) repeated for **every** $\Lambda$, although $\Lambda$ was already a symbol | Derived **once** and cached on disk |
| Generic `Matrix.inv()` on the 4×4 metric (~7 min, large expressions) | Exact inverse exploiting the block structure (only $t$ and $\phi$ are coupled) |
| `dH/dx` and `dH/dp` evaluated separately with NumPy on scalars | Single combined function (`lambdify(..., 'math', cse=True)`) |
| Values of $\Lambda$ run sequentially | Run in parallel on all CPU cores |

The Hamiltonian cache is stored in `~/.cache/kerr_ds/` (override with the `KERR_DS_CACHE_DIR`
environment variable). It is invalidated automatically when `symbolic.py` or the SymPy version
changes, and is safe to delete.

## Author

**NugiePhysics** — undergraduate (S1) thesis in physics.
