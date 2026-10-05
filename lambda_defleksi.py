import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data Pengamatan Nugie
lamda = np.array([1e-10, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6]) 
sudut_defleksi = np.array([43.9149, 43.9151, 43.9156, 43.9169, 43.9252, 43.9359, 44.0355])

# Model Fungsi Non-Linear
def model_fungsi(x, a, b, c):
    return a * (x**b) + c

# PERBAIKAN 1: Optimasi tebakan awal (p0) agar curve_fit konvergen dengan benar
# c didekatkan ke 57.61 (baseline), b diset mendekati 1 (linear/power-law trend)
tebakan_awal = [5000, 1.0, 43.91] 
popt, pcov = curve_fit(model_fungsi, lamda, sudut_defleksi, p0=tebakan_awal, maxfev=20000)
a_opt, b_opt, c_opt = popt

# Membuat data untuk kurva fitting halus
lamda_fit = np.logspace(np.log10(lamda.min()), np.log10(lamda.max()), 1000)
sudut_fit = model_fungsi(lamda_fit, a_opt, b_opt, c_opt)

# Menghitung R-squared
y_prediksi = model_fungsi(lamda, a_opt, b_opt, c_opt)
residuals = sudut_defleksi - y_prediksi
ss_res = np.sum(residuals**2)
ss_tot = np.sum((sudut_defleksi - np.mean(sudut_defleksi))**2)
r_squared = 1 - (ss_res / ss_tot)

# Plot Grafik
plt.figure(figsize=(10, 6))

# Plot titik data asli
plt.scatter(lamda, sudut_defleksi, color='red', s=60, zorder=5, label='Data Pengamatan (Asli)')

# Plot kurva hasil fitting
label_fitting = rf'Kurva Fitting: $y = {a_opt:.2f} \cdot \Lambda^{{{b_opt:.4f}}} + {c_opt:.4f}$ ($R^2 = {r_squared:.4f}$)'
plt.plot(lamda_fit, sudut_fit, color='blue', linestyle='-', linewidth=2, label=label_fitting)
plt.xscale('log')

plt.xlabel(r'Konstanta Kosmologi ($\log_{10}$)', fontsize=12)
plt.ylabel('Sudut Defleksi', fontsize=12)
plt.legend(loc='upper left')
plt.show()
# plt.savefig('lambda_defleksi_fitting.png', dpi=300) # Simpan gambar dengan resolusi tinggi

# Output untuk laporan skripsi
print("--- HASIL FITTING NON-LINEAR UNTUK LAPORAN ---")
print(f"Persamaan Fungsi: y = {a_opt:.4f} * (Lambda^{b_opt:.4f}) + {c_opt:.4f}")
print(f"Nilai R² (Koefisien Determinasi) = {r_squared:.4f}")