"""Plot lintasan foton dan validasi konservasi Hamiltonian."""
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['black', 'blue', 'red', 'green', 'orange', 'purple', 'brown']


def plot_trajectories(results, M, a, b, rh, limit=10, save_path=None):
    fig, ax = plt.subplots(figsize=(8,8))

    # Gambar Event Horizon
    eventhorizon = plt.Circle((0,0), rh, color='gray', alpha=0.3, label="Horizon (Kerr)")
    ax.add_artist(eventhorizon)
    ax.scatter(0, 0, s=20, c='k', label='Singularity')

    # Plot garis dari data trajektori yang sudah dihitung
    for i, res in enumerate(results):
        label_txt = f'$\\Lambda = {res.lmd}$'
        # Menggunakan modulo jika jumlah lambda melebihi jumlah warna
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
    """Validasi konservasi Hamiltonian terhadap jumlah langkah integrasi."""
    steps_arr = np.arange(len(res.H))
    hamiltonian_arr = res.H

    # Hitung fluktuasi (max - min)
    H_max = np.max(hamiltonian_arr)
    H_min = np.min(hamiltonian_arr)
    fluctuation = H_max - H_min

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps_arr, hamiltonian_arr, color='purple', label='Hamiltonian $H$')

    # Log scale pada sumbu Y jika fluktuasinya sangat kecil
    if fluctuation > 0 and fluctuation < 1e-3:
        # symlog untuk variasi kecil di sekitar angka negatif/positif
        ax.set_yscale('symlog', linthresh=0.001)

    ax.set_title("Validasi Konservasi Hamiltonian Selama Integrasi (Trayektori Terakhir)")
    ax.set_xlabel("Jumlah Langkah", fontsize=12)
    ax.set_ylabel("Nilai Hamiltonian", fontsize=12)

    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend()

    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_hamiltonian_vs_r(res, save_path=None):
    """Validasi konservasi Hamiltonian terhadap jarak radial r."""
    r_vals = res.r
    H_vals = res.H

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot H_vals vs r_vals
    ax.plot(r_vals, H_vals, color='teal', linewidth=1.5, label=r'Hamiltonian $\mathcal{H}$')

    # Garis horizontal putus-putus pada y=0 sebagai referensi nilai ideal
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label=r'Ideal $\mathcal{H} = 0$')

    # Skala symlog agar fluktuasi orde kecil (1e-11 s/d 1e-14) terlihat;
    # nilai H bisa sedikit berfluktuasi ke negatif dan positif di sekitar 0
    ax.set_yscale('symlog', linthresh=1e-14)

    # Mencari titik periapsis (r terkecil)
    min_r_idx = np.argmin(r_vals)
    min_r = r_vals[min_r_idx]
    H_at_min_r = H_vals[min_r_idx]

    # Memberikan anotasi otomatis pada r terkecil (periapsis)
    ax.annotate(f'Periapsis\n$r \\approx {min_r:.2f}$\n$\\mathcal{{H}} \\approx {H_at_min_r:.2e}$',
                xy=(min_r, H_at_min_r),
                xytext=(min_r + max(r_vals)*0.1, H_at_min_r + 1e-12),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Pengaturan Judul dan Label Sumbu
    ax.set_title("Validasi Konservasi Hamiltonian vs Jarak Radial ($r$)", fontsize=14, pad=15)
    ax.set_xlabel("Koordinat Radial ($r$)", fontsize=12)
    ax.set_ylabel(r"Nilai Hamiltonian ($\mathcal{H}$)", fontsize=12)

    # Grid dan Legend
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def _save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
