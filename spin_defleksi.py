import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

spin = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.99])
konstanta = {
    'Λ = 0':    [389.0197, 136.5261, 105.2794, 88.5366, 77.4697, 69.7456],
    'Λ = 1e-8': [389.0232, 136.5262, 105.2795, 88.5367, 77.4699, 69.7458],
    'Λ = 5e-8': [389.0371, 136.5264, 105.2799, 88.5372, 77.4706, 69.7467],
    'Λ = 1e-7': [389.0545, 136.5268, 105.2804, 88.5379, 77.4714, 69.7477],
    'Λ = 5e-7': [389.1941, 136.5297, 105.2843, 88.5432, 77.4883, 69.7560],
    'Λ = 1e-6': [389.3690, 136.5334, 105.2893, 88.5499, 77.4871, 69.7667],
    'Λ = 5e-6': [390.7879, 136.5656, 105.3349, 88.6132, 77.5690, 69.8667]
}

def model_eksponensial(x, A, b, C):
    return A * np.exp(-b * x) + C

plt.figure(figsize=(11, 7))

for label, y_vals in konstanta.items():
    y_vals_np = np.array(y_vals)
    
    popt, pcov = curve_fit(model_eksponensial, spin, y_vals_np, p0=[300, 5, 70], maxfev=10000)
    
    residuals = y_vals_np - model_eksponensial(spin, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_vals_np - np.mean(y_vals_np))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    p = plt.plot(spin, y_vals_np, 'o', label=f'Data {label}')
    color = p[0].get_color()
    
    x_fit = np.linspace(0, 1, 100)
    y_fit = model_eksponensial(x_fit, *popt)
    
    formula_label = f"Fit {label}: y = {popt[0]:.2f} * e^(-{popt[1]:.2f} * x) + {popt[2]:.2f} ($R^2$={r_squared:.4f})"
    
    plt.plot(x_fit, y_fit, linestyle='--', color=color, label=formula_label)

plt.xlabel('Spin (a)')
plt.ylabel('Sudut Defleksi')
# plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
plt.legend(loc="upper right", fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()
# plt.savefig('spin_defleksi_fitting.png', dpi=300) # Simpan gambar dengan resolusi tinggi