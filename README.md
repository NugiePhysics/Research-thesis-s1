# Defleksi Cahaya di Ruang-waktu Kerr–de Sitter

Kode penelitian S1: menghitung sudut defleksi cahaya (geodesik null) di sekitar lubang hitam
Kerr–de Sitter untuk beberapa nilai konstanta kosmologi Λ. Persamaan Hamilton diturunkan secara
simbolik (SymPy) lalu diintegrasikan secara numerik dengan Runge–Kutta orde 4.

## Struktur folder

```
Research-thesis-S1/
├── src/kerr_ds/              # package kode fisika
│   ├── base.py               # kelas abstrak Spacetime & Integrator
│   ├── integrators.py        # RungeKutta4
│   ├── symbolic.py           # metrik, invers metrik, Hamiltonian & turunannya (SymPy, di-cache)
│   ├── kerr_de_sitter.py     # KerrDeSitter: evaluasi numerik H, dH/dx, dH/dp, kondisi awal
│   ├── simulation.py         # integrasi lintasan foton & sweep nilai Λ
│   └── plotting.py           # plot lintasan & validasi konservasi Hamiltonian
├── notebooks/
│   └── Penelitian_S1.ipynb   # notebook utama: parameter → simulasi → tabel → plot
├── analysis/                 # fitting hasil defleksi
│   ├── lambda_defleksi.py    # defleksi vs Λ (model a·Λ^b + c)
│   └── spin_defleksi.py      # defleksi vs spin a (model eksponensial)
├── tests/                    # uji regresi terhadap hasil kode asli
└── pyproject.toml
```

## Instalasi

Membutuhkan [uv](https://docs.astral.sh/uv/) (atau `pip` biasa).

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Di VS Code, pilih kernel `.venv` (Python 3.12) saat membuka notebook.

## Menjalankan

- **Notebook**: buka `notebooks/Penelitian_S1.ipynb`, lalu *Run All*.
- **Script fitting**: `python analysis/lambda_defleksi.py` atau `python analysis/spin_defleksi.py`.
- **Uji regresi**: `pytest` — memastikan tabel defleksi, jumlah step, lintasan, dan nilai H
  tetap sama dengan hasil kode asli.

## Catatan performa

Versi awal notebook butuh ±1–2 jam; sekarang ±2,5 menit untuk run pertama dan ±20–30 detik
untuk run berikutnya. Penyebab dan perbaikannya:

| Penyebab | Perbaikan |
|---|---|
| Penurunan simbolik (invers metrik + `simplify`, ±9 menit) diulang untuk **setiap** Λ, padahal Λ sudah berupa simbol | Diturunkan **sekali**, lalu disimpan ke cache disk |
| `Matrix.inv()` generik pada metrik 4×4 (±7 menit, ekspresi besar) | Invers eksak memanfaatkan struktur blok (hanya t–φ terkopel) |
| `dH/dx` dan `dH/dp` dievaluasi terpisah dengan numpy pada skalar | Satu fungsi gabungan (`lambdify(..., 'math', cse=True)`) |
| 7 nilai Λ dijalankan berurutan | Dijalankan paralel di semua core CPU |

Cache Hamiltonian disimpan di `~/.cache/kerr_ds/` (ubah lewat env var `KERR_DS_CACHE_DIR`).
Cache otomatis diperbarui bila kode di `symbolic.py` atau versi SymPy berubah; aman untuk dihapus.
