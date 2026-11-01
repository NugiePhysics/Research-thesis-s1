"""Data, figures, and tables for the final report (`laporan_akhir.tex`).

Run from anywhere (the repository's `src/` is added to the import path):

    python docs/laporan_akhir/generate_results.py

The simulations (143 + 10 trajectories) take about 15 minutes on a 4-core CPU.
Their results are cached in `data/`; delete that folder to recompute. Figures
are written to `figures/` and LaTeX tables to `tabel/`.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from kerr_ds import (  # noqa: E402
    analytic,
    exponential_decay,
    fit_curve,
    linear,
    power_law,
    run_spin_sweep,
)

DATA, FIG, TAB = HERE / "data", HERE / "figures", HERE / "tabel"

# Simulation parameters (as in the thesis)
M, E, B_IMPACT, R0 = 1.0, 1.0, 7.0, 500.0
THETA0, PHI0 = np.pi / 2, 0.0
SPINS = [-0.99, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.99]
LAMBDAS_POS = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6]
LAMBDAS = sorted([-x for x in LAMBDAS_POS] + [0.0] + LAMBDAS_POS)
TRAJ_SPINS = [0.99, 0.0, -0.99]
SPIN_TICKS = [-1.0, -0.5, 0.0, 0.5, 1.0]

# Validation against Hasanuddin (2012), Tables 3-4 (radians)
VAL_SPINS = [0.52, 0.9939]
VAL_IMPACTS = [5.0, 10.0, 15.0, 20.0, 40.0]
HASANUDDIN = {0.52: [1.847027, 0.532207, 0.315107, 0.225507, 0.103067],
              0.9939: [1.315827, 0.494747, 0.303387, 0.219707, 0.101867]}

# Chart style: validated categorical slots 1-3, near-black ink for simulation points
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.5, "savefig.bbox": "tight", "pdf.fonttype": 42, "mathtext.fontset": "cm",
})


def comma(fmt):
    """Tick formatter with a decimal comma (Indonesian)."""
    return FuncFormatter(lambda x, _: (fmt.format(x)).replace("-", "−").replace(".", ","))


def num(x, nd, sign=False):
    """Number for LaTeX tables: decimal comma and a proper minus sign."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$").replace("+", "$+$").replace(".", "{,}")


def lam_tex(x):
    """Lambda value in LaTeX (e.g. 5\\times10^{-6})."""
    if x == 0:
        return "$0$"
    m, e = f"{abs(x):.0e}".split("e")
    sign = "-" if x < 0 else ""
    mant = "" if m == "1" else rf"{m}\times"
    return rf"${sign}{mant}10^{{{int(e)}}}$"


# ----------------------------------------------------------------------------- simulation
def simulate():
    grid_file, val_file, traj_file = DATA / "hasil_grid.csv", DATA / "validasi.csv", DATA / "lintasan.npz"
    if grid_file.exists() and val_file.exists() and traj_file.exists():
        return
    DATA.mkdir(exist_ok=True)

    res = run_spin_sweep(M, SPINS, LAMBDAS, R0, THETA0, PHI0, E, B_IMPACT * E, verbose=False)
    header = ("a,Lambda,deflection_deg,coord_deg,legacy_deg,psi_source,psi_observer,r_observer,steps,"
              "H_max,exact_deg,exact_coord_deg,huang2025_deg,belhaj2022_deg")
    rows = []
    for a in SPINS:
        for lmd in LAMBDAS:
            r = res[a, lmd]
            r_obs = float(r.r[-1])
            rows.append([a, lmd, r.deflection_deg, r.deflection_coord_deg, r.deflection_legacy_deg,
                         r.psi_source, r.psi_observer, r_obs, r.steps, float(np.max(np.abs(r.H))),
                         analytic.deflection_exact_deg(M, a, lmd, B_IMPACT, R0, r_obs),
                         analytic.coordinate_deflection_exact_deg(M, a, lmd, B_IMPACT, R0, r_obs),
                         analytic.huang2025_deflection_deg(M, a, lmd, B_IMPACT, R0, r_obs),
                         analytic.belhaj2022_deflection_deg(M, a, lmd, B_IMPACT, R0, r_obs)])
    np.savetxt(grid_file, np.array(rows), delimiter=",", header=header, comments="", fmt="%.12g")

    traj = {}
    for a in TRAJ_SPINS:
        r = res[a, 0.0]
        keep = r.r < 40
        traj[f"X_{a}"], traj[f"Y_{a}"] = r.X[keep], r.Y[keep]
    h = res[0.99, 5e-6]
    traj["H_r"], traj["H"] = h.r, h.H
    np.savez_compressed(traj_file, **traj)

    vrows = []
    for b in VAL_IMPACTS:
        vres = run_spin_sweep(M, VAL_SPINS, [0.0], R0, THETA0, PHI0, E, b * E, verbose=False)
        for a in VAL_SPINS:
            r = vres[a, 0.0]
            r_obs = float(r.r[-1])
            vrows.append([a, b, np.degrees(HASANUDDIN[a][VAL_IMPACTS.index(b)]),
                          analytic.asymptotic_kerr_deflection_deg(M, a, b), r.deflection_deg,
                          analytic.deflection_exact_deg(M, a, 0.0, b, R0, r_obs), r.deflection_coord_deg,
                          r.deflection_legacy_deg, r_obs])
    np.savetxt(val_file, np.array(vrows), delimiter=",", comments="", fmt="%.12g",
               header="a,b,hasanuddin_deg,exact_asymptotic_deg,deflection_deg,exact_deg,coord_deg,legacy_deg,r_observer")


def load():
    g = np.genfromtxt(DATA / "hasil_grid.csv", delimiter=",", names=True)
    v = np.genfromtxt(DATA / "validasi.csv", delimiter=",", names=True)
    t = dict(np.load(DATA / "lintasan.npz"))
    return g, v, t


def column(g, a, name):
    """Values of `name` for spin a, ordered by Lambda (LAMBDAS)."""
    sel = np.isclose(g["a"], a)
    order = np.argsort(g["Lambda"][sel])
    return g[name][sel][order]


# ----------------------------------------------------------------------------- analysis
def analyse(g, v):
    out = {}
    lam = np.array(LAMBDAS)
    i0 = LAMBDAS.index(0.0)
    small = np.abs(lam) <= 1e-6

    # Lambda slopes (linear fit for |Lambda| <= 1e-6) for every spin
    slopes = {}
    for a in SPINS:
        s = {}
        for key in ("deflection_deg", "exact_deg", "huang2025_deg", "belhaj2022_deg", "coord_deg"):
            y = column(g, a, key)
            p0 = [(y[small][-1] - y[small][0]) / 2e-6, y[i0]]
            popt, _, perr = fit_curve(linear, lam[small], y[small], p0=p0, return_errors=True)
            s[key] = (popt[0], perr[0])
        slopes[a] = s
    out["slopes"] = slopes

    # delta = (phi_OS - pi) + (Psi_O - Psi_S + pi): coordinate part and observer-angle part
    dec = {}
    for a in (0.99, 0.0, -0.6):
        d = column(g, a, "deflection_deg")
        c = column(g, a, "coord_deg")
        dec[a] = {"lam": lam, "delta": d - d[i0], "coord": c - c[i0], "psi": (d - c) - (d[i0] - c[i0])}
    out["decomposition"] = dec

    # Power-law fits for a = 0.99, separately for each sign of Lambda
    fits = {}
    for sign in (+1, -1):
        sel = (np.sign(lam) == sign) | (lam == 0)
        x = np.abs(lam[sel])
        order = np.argsort(x)
        x, y = x[order], column(g, 0.99, "deflection_deg")[sel][order]
        xf = np.where(x == 0, 1e-10, x)
        popt, r2, perr = fit_curve(power_law, xf, y, p0=[np.sign(y[-1] - y[0]) * 5000, 1.0, y[0]],
                                   maxfev=20000, return_errors=True)
        fits[sign] = dict(popt=popt.tolist(), perr=perr.tolist(), r2=r2, x=xf.tolist(), y=y.tolist())
    out["powerlaw"] = fits

    # Same fit on the exact deflection with the observer much farther from the cosmological horizon
    lp = np.array([0.0] + LAMBDAS_POS)
    for r_s in (100.0, 500.0):
        y = np.array([analytic.deflection_exact_deg(M, 0.99, l, B_IMPACT, r_s, r_s + 5) for l in lp])
        xf = np.where(lp == 0, 1e-10, lp)
        popt, r2, perr = fit_curve(power_law, xf, y, p0=[-5000, 1.0, y[0]], maxfev=20000, return_errors=True)
        out[f"powerlaw_exact_r{int(r_s)}"] = dict(popt=popt.tolist(), perr=perr.tolist())

    # Spin fits at Lambda = 0 (prograde, retrograde)
    d0 = {a: column(g, a, "deflection_deg")[i0] for a in SPINS}
    spin_fits = {}
    for name, group, p0 in (("prograde", [a for a in SPINS if a >= 0], [300, 5, 70]),
                            ("retrograde", [a for a in SPINS if a <= 0], [1, 6, 70])):
        popt, r2, perr = fit_curve(exponential_decay, group, [d0[a] for a in group], p0=p0, maxfev=20000,
                                   return_errors=True)
        spin_fits[name] = dict(group=group, popt=popt.tolist(), perr=perr.tolist(), r2=r2)
    out["spin_fits"] = spin_fits
    out["delta0"] = d0

    # Critical impact parameter and the strong-field case a = -0.99
    out["b_c"] = {a: analytic.critical_impact_parameter(M, a, 0.0)[0] for a in SPINS}
    out["b_c_retro"] = {l: analytic.critical_impact_parameter(M, -0.99, l)[0] for l in LAMBDAS}
    h = 1e-4
    out["ddelta_db"] = {a: (analytic.deflection_exact_deg(M, a, 0.0, B_IMPACT + h, R0, R0 + 5)
                            - analytic.deflection_exact_deg(M, a, 0.0, B_IMPACT - h, R0, R0 + 5)) / (2 * h)
                        for a in (-0.99, 0.0, 0.99)}

    # Accuracy
    out["max_sim_exact"] = float(np.max(np.abs(g["deflection_deg"] - g["exact_deg"])))
    out["max_H"] = float(np.max(g["H_max"]))
    out["max_H_b7"] = {a: float(np.max(g["H_max"][np.isclose(g["a"], a)])) for a in SPINS}
    out["val_max_corr"] = float(np.max(np.abs(v["coord_deg"] + np.degrees(
        np.arcsin(v["b"] / R0) + np.arcsin(v["b"] / v["r_observer"])) - v["exact_asymptotic_deg"])))
    out["val_max_sim_exact"] = float(np.max(np.abs(v["deflection_deg"] - v["exact_deg"])))
    return out


# ----------------------------------------------------------------------------- figures
def fig_trajectories(t):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    styles = {0.99: (C1, "-", "prograde, $a=0{,}99$"), 0.0: (C2, "--", "Schwarzschild, $a=0$"),
              -0.99: (C3, "-.", r"retrograde, $a=-0{,}99$")}
    for a in TRAJ_SPINS:
        col, ls, lab = styles[a]
        ax.plot(t[f"X_{a}"], t[f"Y_{a}"], color=col, linestyle=ls, label=lab)
    # Event horizons (r_+ depends on a^2 only, so prograde and retrograde share one)
    ax.add_patch(plt.Circle((0, 0), M + np.sqrt(M**2 - 0.99**2), color=INK2, alpha=0.35, lw=0))
    ax.add_patch(plt.Circle((0, 0), 2.0 * M, fill=False, color=INK2, lw=0.8))
    ax.annotate("horizon:\n" + r"$r_+ = 2M$ ($a=0$)" + "\n" + r"$r_+ = 1{,}14M$ ($|a|=0{,}99$)",
                xy=(1.2, -1.6), xytext=(3.5, -9.5), fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$X = r\cos\phi$ ($M$)")
    ax.set_ylabel(r"$Y = r\sin\phi$ ($M$)")
    ax.legend(loc="upper left")
    fig.savefig(FIG / "lintasan.pdf")
    plt.close(fig)


def fig_hamiltonian(t):
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    r, H = t["H_r"], np.maximum(np.abs(t["H"]), 1e-17)
    k = int(np.argmin(r))  # periapsis splits the ingoing and outgoing legs
    ax.semilogy(r[:k + 1], H[:k + 1], color=C1, lw=1.0, label="Lintasan masuk ($r_0 \\to$ periapsis)")
    ax.semilogy(r[k:], H[k:], color=C2, lw=1.0, linestyle="--", label="Lintasan keluar (periapsis $\\to r_O$)")
    ax.set_xlabel(r"Koordinat radial $r$ ($M$)")
    ax.set_ylabel(r"$|H|$")
    ax.set_ylim(1e-17, 1e-9)
    ax.legend(loc="upper right")
    fig.savefig(FIG / "hamiltonian.pdf")
    plt.close(fig)


def fig_spin(an):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.9))
    d0 = an["delta0"]
    ax1.plot(SPINS, [d0[a] for a in SPINS], "o", color=INK, ms=4.5)
    ax1.set_yscale("log")
    ax1.set_yticks([50, 100, 200])
    ax1.yaxis.set_major_formatter(comma("{:.0f}"))
    ax1.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: ""))
    ax1.set_xlabel("Spin $a$ ($M$)")
    ax1.set_ylabel(r"$\delta$ (derajat), $\Lambda = 0$")
    ax1.xaxis.set_major_formatter(comma("{:.1f}"))
    ax1.set_title("(a) Sudut defleksi", loc="left")

    aa = np.linspace(-0.999, 0.999, 400)
    bc = [-x + 6 * np.cos(np.arccos(-x) / 3) for x in aa]  # Congdon et al. (2022), Eq. (17)
    ax2.plot(aa, bc, color=C1, label=r"$b_c$ (Congdon dkk. 2022)")
    ax2.axhline(B_IMPACT, color=C2, linestyle="--", label=r"$b = 7M$ (simulasi)")
    ax2.set_xlabel("Spin $a$ ($M$)")
    ax2.set_ylabel(r"Parameter dampak ($M$)")
    ax2.xaxis.set_major_formatter(comma("{:.1f}"))
    ax2.set_title("(b) Parameter dampak kritis", loc="left")
    ax2.legend(loc="lower left")
    for ax in (ax1, ax2):
        ax.set_xticks(SPIN_TICKS)
    fig.tight_layout()
    fig.savefig(FIG / "pengaruh_spin.pdf")
    plt.close(fig)


def fig_lambda_shift(g):
    lam = np.array(LAMBDAS)
    i0 = LAMBDAS.index(0.0)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0), sharex=True)
    for ax, a in zip(axes.flat, (0.99, 0.0, -0.6, -0.99)):
        def sh(key):
            y = column(g, a, key)
            return (y - y[i0]) * 1e3  # millidegrees
        ax.plot(lam * 1e6, sh("exact_deg"), color=C1, label="Eksak (kuadratur)")
        ax.plot(lam * 1e6, sh("huang2025_deg"), color=C2, linestyle="--", label="Huang dkk. (2025), Pers. (24)")
        ax.plot(lam * 1e6, sh("belhaj2022_deg"), color=C3, linestyle=":", lw=1.8,
                label="Belhaj dkk. (2022), Pers. (5.1)")
        ax.plot(lam * 1e6, sh("deflection_deg"), "o", color=INK, ms=3.5, label="Simulasi (RK4)", zorder=5)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_title(f"$a = {a:g}$".replace(".", "{,}"), loc="left")
        ax.xaxis.set_major_formatter(comma("{:.0f}"))
        ax.yaxis.set_major_formatter(comma("{:.0f}"))
    for ax in axes[1]:
        ax.set_xlabel(r"$\Lambda$ ($10^{-6}\,M^{-2}$)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\delta(\Lambda)-\delta(0)$ ($10^{-3}$ derajat)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "pergeseran_lambda.pdf")
    plt.close(fig)


def fig_slopes(an):
    s = an["slopes"]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    a = np.array(SPINS)
    ax.plot(a, [s[x]["huang2025_deg"][0] / 1e3 for x in SPINS], color=C2, linestyle="--", marker="s", ms=4,
            label="Huang dkk. (2025), Pers. (24)")
    ax.plot(a, [s[x]["belhaj2022_deg"][0] / 1e3 for x in SPINS], color=C3, linestyle=":", marker="^", ms=4.5,
            lw=1.8, label="Belhaj dkk. (2022), Pers. (5.1)")
    ax.plot(a, [s[x]["deflection_deg"][0] / 1e3 for x in SPINS], "o", color=INK, ms=4.5, label="Simulasi (RK4)",
            zorder=5)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Spin $a$ ($M$)")
    ax.set_ylabel(r"$d\delta/d\Lambda$ ($10^{3}$ derajat $M^2$)")
    ax.set_xticks(SPIN_TICKS)
    ax.xaxis.set_major_formatter(comma("{:.1f}"))
    ax.legend(loc="lower right")
    fig.savefig(FIG / "kemiringan_lambda.pdf")
    plt.close(fig)


def fig_definitions(an):
    d = an["decomposition"][0.99]
    lam = d["lam"] * 1e6
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(lam, d["coord"] * 1e3, color=C1, marker="o", ms=3.5, label=r"$\phi_{OS}-\pi$ (definisi koordinat)")
    ax.plot(lam, d["psi"] * 1e3, color=C2, linestyle="--", marker="s", ms=3.5,
            label=r"$\Psi_O-\Psi_S+\pi$ (sudut pengamat)")
    ax.plot(lam, d["delta"] * 1e3, color=C3, linestyle="-.", marker="^", ms=4,
            label=r"$\delta$ (definisi artikel) = jumlah keduanya")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel(r"$\Lambda$ ($10^{-6}\,M^{-2}$)")
    ax.set_ylabel(r"Perubahan terhadap $\Lambda=0$ ($10^{-3}$ derajat)")
    ax.xaxis.set_major_formatter(comma("{:.0f}"))
    ax.yaxis.set_major_formatter(comma("{:.0f}"))
    ax.legend(loc="lower left")
    fig.savefig(FIG / "dua_definisi.pdf")
    plt.close(fig)


def fig_fits(an):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.9))
    f = an["powerlaw"][1]
    x = np.array(f["x"])
    xx = np.logspace(-10, np.log10(5e-6), 300)
    ax1.plot(xx, power_law(xx, *f["popt"]), color=C1, label="Fit pangkat")
    ax1.plot(x, f["y"], "o", color=INK, ms=4, label="Simulasi", zorder=5)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\Lambda$ ($M^{-2}$), $a = 0{,}99$")
    ax1.set_ylabel(r"$\delta$ (derajat)")
    ax1.yaxis.set_major_formatter(comma("{:.2f}"))
    ax1.set_title(r"(a) $\delta$ terhadap $\Lambda > 0$", loc="left")
    ax1.legend(loc="lower left")

    d0 = an["delta0"]
    for name, col, ls in (("prograde", C1, "-"), ("retrograde", C2, "--")):
        sf = an["spin_fits"][name]
        grp = np.array(sf["group"])
        aa = np.linspace(grp.min(), grp.max(), 200)
        ax2.plot(aa, exponential_decay(aa, *sf["popt"]), color=col, linestyle=ls, label=f"Fit {name}")
    ax2.plot(SPINS, [d0[a] for a in SPINS], "o", color=INK, ms=4, label="Simulasi", zorder=5)
    ax2.set_xlabel("Spin $a$ ($M$), $\\Lambda = 0$")
    ax2.set_ylabel(r"$\delta$ (derajat)")
    ax2.set_xticks(SPIN_TICKS)
    ax2.xaxis.set_major_formatter(comma("{:.1f}"))
    ax2.set_title(r"(b) $\delta$ terhadap spin", loc="left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fitting.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- tables
def write(name, body):
    (TAB / name).write_text(body, encoding="utf-8")


def tables(g, v, an):
    i0 = LAMBDAS.index(0.0)

    # Validation (Hasanuddin), grouped by spin
    rows = []
    for k in np.lexsort((v["b"], v["a"])):
        a, b = v["a"][k], v["b"][k]
        corr = v["coord_deg"][k] + np.degrees(np.arcsin(b / R0) + np.arcsin(b / v["r_observer"][k]))
        rows.append(f"{num(a, 4 if a > 0.6 else 2)} & {b:g} & {num(v['hasanuddin_deg'][k], 3)} & "
                    f"{num(v['exact_asymptotic_deg'][k], 4)} & {num(v['deflection_deg'][k], 4)} & "
                    f"{num(v['exact_deg'][k], 4)} & {num(corr, 4)} & {num(v['legacy_deg'][k], 3)} \\\\")
    write("validasi.tex", "\n".join(rows) + "\n")

    # Full grid of deflection angles, Lambda >= 0 and Lambda <= 0
    for name, lams in (("grid_positif.tex", [0.0] + LAMBDAS_POS),
                       ("grid_negatif.tex", [0.0] + [-x for x in LAMBDAS_POS])):
        rows = []
        for a in SPINS:
            y = column(g, a, "deflection_deg")
            vals = [y[LAMBDAS.index(l)] for l in lams]
            rows.append(f"{num(a, 2)} & " + " & ".join(num(x, 4) for x in vals) + r" \\")
        write(name, "\n".join(rows) + "\n")

    # Shift at Lambda = +1e-6 and comparison with the articles
    k = LAMBDAS.index(1e-6)
    rows = []
    for a in SPINS:
        def sh(key):
            y = column(g, a, key)
            return (y[k] - y[i0]) * 1e3
        s, ex, hu, be = sh("deflection_deg"), sh("exact_deg"), sh("huang2025_deg"), sh("belhaj2022_deg")
        rel_h, rel_b = 100 * (hu - s) / abs(s), 100 * (be - s) / abs(s)
        rows.append(f"{num(a, 2)} & {num(s, 3, True)} & {num(ex, 3, True)} & {num(hu, 3, True)} & "
                    f"{num(rel_h, 1, True)} & {num(be, 3, True)} & {num(rel_b, 1, True)} \\\\")
    write("pergeseran_1e-6.tex", "\n".join(rows) + "\n")

    # Two definitions for a = 0.99
    d = an["decomposition"][0.99]
    legacy = column(g, 0.99, "legacy_deg")
    rows = []
    for j, l in enumerate(LAMBDAS):
        rows.append(f"{lam_tex(l)} & {num(legacy[j], 4)} & {num(d['coord'][j] * 1e3, 2, True)} & "
                    f"{num(d['psi'][j] * 1e3, 2, True)} & {num(d['delta'][j] * 1e3, 2, True)} \\\\")
    write("dua_definisi.tex", "\n".join(rows) + "\n")

    # Strong-field case a = -0.99
    y = column(g, -0.99, "deflection_deg")
    c = column(g, -0.99, "coord_deg")
    rows = []
    for j, l in enumerate(LAMBDAS):
        if abs(l) not in (0.0, 1e-7, 1e-6, 5e-6):
            continue
        bc = an["b_c_retro"][l]
        rows.append(f"{lam_tex(l)} & {num(bc, 6)} & {num(B_IMPACT - bc, 6)} & {num(y[j], 4)} & "
                    f"{num(y[j] - y[i0], 4, True)} & {num(c[j] - c[i0], 4, True)} \\\\")
    write("medan_kuat.tex", "\n".join(rows) + "\n")

    # Spin dependence at Lambda = 0 with b_c and the weak-field formulas
    rows = []
    for a in SPINS:
        d0 = an["delta0"][a]
        wf_h = analytic.huang2025_deflection_deg(M, a, 0.0, B_IMPACT, R0, R0 + 5)
        wf_b = analytic.belhaj2022_deflection_deg(M, a, 0.0, B_IMPACT, R0, R0 + 5)
        rows.append(f"{num(a, 2)} & {num(an['b_c'][a], 4)} & {num(d0, 4)} & {num(wf_h, 2)} & {num(wf_b, 2)} \\\\")
    write("spin.tex", "\n".join(rows) + "\n")


# ----------------------------------------------------------------------------- main
def main():
    for p in (FIG, TAB):
        p.mkdir(exist_ok=True)
    simulate()
    g, v, t = load()
    an = analyse(g, v)
    fig_trajectories(t)
    fig_hamiltonian(t)
    fig_spin(an)
    fig_lambda_shift(g)
    fig_slopes(an)
    fig_definitions(an)
    fig_fits(an)
    tables(g, v, an)

    # Key numbers quoted in the text
    summary = {k: an[k] for k in ("powerlaw_exact_r100", "powerlaw_exact_r500", "spin_fits", "max_sim_exact",
                                  "max_H", "val_max_corr", "val_max_sim_exact")}
    summary["powerlaw"] = {str(s): {k: an["powerlaw"][s][k] for k in ("popt", "perr", "r2")} for s in (1, -1)}
    summary["slopes"] = {str(a): {k: list(v) for k, v in s.items()} for a, s in an["slopes"].items()}
    summary["b_c"] = {str(a): b for a, b in an["b_c"].items()}
    summary["b_c_retro"] = {str(l): b for l, b in an["b_c_retro"].items()}
    summary["ddelta_db"] = {str(a): x for a, x in an["ddelta_db"].items()}
    summary["delta0"] = {str(a): x for a, x in an["delta0"].items()}
    summary["max_H_b7"] = {str(a): x for a, x in an["max_H_b7"].items()}
    (DATA / "ringkasan.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
