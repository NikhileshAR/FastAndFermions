import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

def gaussian(x, A, sigma):
    return A * np.exp(-(x**2) / (2 * sigma**2))

input_file = "consolidated_data.csv"
thicknesses, sigmas, sigma_errors = [], [], []

with open(input_file, "r") as f:
    lines = f.readlines()

current_theta, current_counts, current_t = [], [], None

def get_core_cut(t):
    if t <= 0.10:   return 1.5
    elif t <= 0.25: return 2.5
    else:           return 3.0

def fit_block(t, theta_arr, counts_arr):
    theta  = np.array(theta_arr)
    counts = np.array(counts_arr)
    cut    = get_core_cut(t)
    core_mask   = theta <= cut
    theta_core  = theta[core_mask]
    counts_core = counts[core_mask]
    if len(theta_core) < 4:
        return None, None
    theta_full  = np.concatenate((-theta_core[::-1], theta_core))
    counts_full = np.concatenate(( counts_core[::-1], counts_core))
    weights     = np.where(counts_full > 0, np.sqrt(counts_full), 1.0)
    try:
        p0 = [np.max(counts_full), 0.5]
        popt, pcov = curve_fit(gaussian, theta_full, counts_full,
                               p0=p0, sigma=weights, absolute_sigma=True, maxfev=10000)
        return abs(popt[1]), np.sqrt(np.diag(pcov))[1]
    except Exception as e:
        print("  WARNING: fit failed for t=" + str(round(t,3)) + " — " + str(e))
        return None, None

for line in lines:
    stripped = line.strip()
    if stripped.startswith("hist_t_"):
        if current_t is not None and len(current_theta) > 0:
            sig, err = fit_block(current_t, current_theta, current_counts)
            if sig is not None:
                thicknesses.append(current_t); sigmas.append(sig); sigma_errors.append(err)
        current_t = float(stripped.split("_")[-1].replace(".csv",""))
        current_theta, current_counts = [], []
    elif stripped.startswith("theta_deg") or stripped == "":
        continue
    else:
        parts = stripped.replace(",","\t").split("\t")
        if len(parts) >= 2:
            current_theta.append(float(parts[0])); current_counts.append(float(parts[1]))

if current_t is not None and len(current_theta) > 0:
    sig, err = fit_block(current_t, current_theta, current_counts)
    if sig is not None:
        thicknesses.append(current_t); sigmas.append(sig); sigma_errors.append(err)

thicknesses  = np.array(thicknesses); sigmas = np.array(sigmas); sigma_errors = np.array(sigma_errors)
order        = np.argsort(thicknesses)
thicknesses  = thicknesses[order]; sigmas = sigmas[order]; sigma_errors = sigma_errors[order]
stable       = thicknesses <= 0.45
t_s          = thicknesses[stable]; sig_s = sigmas[stable]; err_s = sigma_errors[stable]
sqrt_t       = np.sqrt(t_s)

def highland(x, p=500.0):
    ratio = np.where(x > 0, x, 1e-9)
    return np.degrees((13.6/p) * np.sqrt(ratio) * (1 + 0.038*np.log(ratio)))
hl = highland(t_s)

# ─────────────────────────────────────────
# FIX: Force fit through zero (no intercept)
# sigma = slope * sqrt(t)  — physically correct
# ─────────────────────────────────────────
lin_mask = t_s <= 0.25
# Fit with no intercept: minimize sum((sigma - slope*sqrt(t))^2)
slope = np.dot(sqrt_t[lin_mask], sig_s[lin_mask]) / np.dot(sqrt_t[lin_mask], sqrt_t[lin_mask])
residuals = sig_s[lin_mask] - slope * sqrt_t[lin_mask]
ss_res = np.sum(residuals**2)
ss_tot = np.sum((sig_s[lin_mask] - np.mean(sig_s[lin_mask]))**2)
r2 = 1 - ss_res/ss_tot

print("Zero-intercept fit (t<=0.25): sigma = " + str(round(slope,4)) + " * sqrt(t)")
print("R^2 = " + str(round(r2, 5)))

fit_x = np.linspace(0, sqrt_t.max(), 300)
fit_y = slope * fit_x   # passes through origin

# ═══════════════════════════════════════════════
# PLOT 1 — sigma vs sqrt(t)  +  zero-intercept fit
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
ax.errorbar(sqrt_t, sig_s, yerr=err_s, fmt='o', color='royalblue',
            ms=5, elinewidth=1, capsize=3, label='Measured sigma_core', zorder=3)
ax.plot(fit_x, fit_y, 'r-', lw=2,
        label='Fit: sigma = ' + str(round(slope,3)) + ' * sqrt(t)  (R² = ' + str(round(r2,4)) + ',  t <= 0.25)')
ax.plot(sqrt_t, hl, 's--', color='darkorange', ms=4, lw=1.5,
        label='Highland (500 MeV e-)')
ax.axvline(np.sqrt(0.25), color='grey', lw=1, ls=':', alpha=0.8)
ax.text(np.sqrt(0.25)+0.005, 0.2, 't=0.25\n(linear\nlimit)', fontsize=9, color='grey')
ax.axvline(np.sqrt(0.40), color='red', lw=1, ls='--', alpha=0.6)
ax.text(np.sqrt(0.40)+0.005, 0.2, 't=0.40\n(non-Gaussian\nregime)', fontsize=9, color='red')
ax.set_xlabel('sqrt(Thickness)   sqrt(x/X0)', fontsize=12)
ax.set_ylabel('sigma_core  (degrees)', fontsize=12)
ax.set_title('Phase 1 — sigma vs sqrt(Thickness)\n'
             'Linear regime confirms sqrt(t) multiple scattering scaling',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, None)
ax.set_ylim(0, None)
plt.tight_layout()
plt.savefig('sigma_vs_sqrt_thickness.png', dpi=150)
plt.show()
print('Saved: sigma_vs_sqrt_thickness.png')

# ═══════════════════════════════════════════════
# PLOT 2 — sigma vs thickness
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
ax.errorbar(t_s, sig_s, yerr=err_s, fmt='o', color='royalblue',
            ms=5, elinewidth=1, capsize=3, label='Measured sigma_core', zorder=3)
t_smooth = np.linspace(0, 0.25, 300)
ax.plot(t_smooth, slope*np.sqrt(t_smooth), 'r-', lw=2,
        label='Fit: sigma = '+str(round(slope,3))+' * sqrt(t)   (t <= 0.25)')
ax.plot(t_s, hl, 's--', color='darkorange', ms=4, lw=1.5, label='Highland (500 MeV e-)')
ax.axvline(0.25, color='grey', lw=1, ls=':', alpha=0.8)
ax.text(0.255, 0.2, 't=0.25\n(linear\nlimit)', fontsize=9, color='grey')
ax.axvline(0.40, color='red', lw=1, ls='--', alpha=0.6)
ax.text(0.405, 0.2, 't=0.40\n(non-Gaussian\nregime)', fontsize=9, color='red')
ax.set_xlabel('Thickness   (x/X0)', fontsize=12)
ax.set_ylabel('sigma_core  (degrees)', fontsize=12)
ax.set_title('Phase 1 — sigma vs Thickness', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, None); ax.set_ylim(0, None)
plt.tight_layout()
plt.savefig('sigma_vs_thickness.png', dpi=150)
plt.show()
print('Saved: sigma_vs_thickness.png')

# ═══════════════════════════════════════════════
# PLOT 3 — Highland comparison
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
ax.errorbar(sqrt_t, sig_s, yerr=err_s, fmt='o', color='royalblue',
            ms=5, elinewidth=1, capsize=3, label='Measured sigma_core')
ax.plot(sqrt_t, hl, '-', color='darkorange', lw=2,
        label='Highland Prediction (500 MeV e-)')
ax.set_xlabel('sqrt(Thickness)   sqrt(x/X0)', fontsize=12)
ax.set_ylabel('sigma  (degrees)', fontsize=12)
ax.set_title('Phase 1 — Comparison with Highland Formula\n'
             'Measured width consistently larger than Highland prediction',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, None); ax.set_ylim(0, None)
plt.tight_layout()
plt.savefig('sigma_vs_sqrt_with_theory.png', dpi=150)
plt.show()
print('Saved: sigma_vs_sqrt_with_theory.png')

# Save results table
results = np.column_stack((t_s, sig_s, err_s))
np.savetxt("phase1_results.txt", results,
           header="Thickness    Sigma(deg)    Sigma_error(deg)", fmt="%.6f")
print("\nPhase 1 complete. Results saved.")