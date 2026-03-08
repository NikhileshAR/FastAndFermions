import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# Gaussian model (centered at 0)
# -----------------------------
def gaussian(x, A, sigma):
    return A * np.exp(-(x**2) / (2 * sigma**2))

input_file = "consolidated_data.csv"

thicknesses = []
sigmas = []
sigma_errors = []

with open(input_file, "r") as f:
    lines = f.readlines()

current_theta = []
current_counts = []
current_t = None

# -----------------------------
# Adaptive core cut
# (tighter for thin, wider for thick)
# -----------------------------
def get_core_cut(t):
    if t <= 0.10:
        return 1.5
    elif t <= 0.25:
        return 2.5
    else:
        return 3.0

# -----------------------------
# Fit one histogram block
# -----------------------------
def fit_block(t, theta_arr, counts_arr):
    theta  = np.array(theta_arr)
    counts = np.array(counts_arr)

    # Select Gaussian core with adaptive cut
    cut = get_core_cut(t)
    core_mask    = theta <= cut
    theta_core   = theta[core_mask]
    counts_core  = counts[core_mask]

    if len(theta_core) < 4:
        return None, None

    # Mirror data to make symmetric distribution
    theta_full  = np.concatenate((-theta_core[::-1], theta_core))
    counts_full = np.concatenate(( counts_core[::-1], counts_core))

    # Weight by sqrt(counts) for proper chi-square minimisation
    weights = np.where(counts_full > 0, np.sqrt(counts_full), 1.0)

    try:
        p0 = [np.max(counts_full), 0.5]
        popt, pcov = curve_fit(gaussian, theta_full, counts_full,
                               p0=p0, sigma=weights, absolute_sigma=True,
                               maxfev=10000)
        sigma_fit = abs(popt[1])
        sigma_err = np.sqrt(np.diag(pcov))[1]
        return sigma_fit, sigma_err
    except Exception as e:
        print(f"  WARNING: fit failed for t={t:.3f} — {e}")
        return None, None

# -----------------------------
# Read file and process blocks
# -----------------------------
for line in lines:
    stripped = line.strip()

    if stripped.startswith("hist_t_"):

        # Fit previous block
        if current_t is not None and len(current_theta) > 0:
            sig, err = fit_block(current_t, current_theta, current_counts)
            if sig is not None:
                thicknesses.append(current_t)
                sigmas.append(sig)
                sigma_errors.append(err)

        # Start new histogram block
        current_t      = float(stripped.split("_")[-1].replace(".csv", ""))
        current_theta  = []
        current_counts = []

    elif stripped.startswith("theta_deg") or stripped == "":
        continue

    else:
        parts = stripped.replace(",", "\t").split("\t")
        if len(parts) >= 2:
            current_theta.append(float(parts[0]))
            current_counts.append(float(parts[1]))

# Fit last block
if current_t is not None and len(current_theta) > 0:
    sig, err = fit_block(current_t, current_theta, current_counts)
    if sig is not None:
        thicknesses.append(current_t)
        sigmas.append(sig)
        sigma_errors.append(err)

# Convert and sort
thicknesses  = np.array(thicknesses)
sigmas       = np.array(sigmas)
sigma_errors = np.array(sigma_errors)

order        = np.argsort(thicknesses)
thicknesses  = thicknesses[order]
sigmas       = sigmas[order]
sigma_errors = sigma_errors[order]

# Use only stable region (avoids diverging fits at very high thickness)
stable       = thicknesses <= 0.45
t_s          = thicknesses[stable]
sig_s        = sigmas[stable]
err_s        = sigma_errors[stable]

# -----------------------------
# Plot 1: Sigma vs Thickness
# -----------------------------
plt.figure()
plt.errorbar(t_s, sig_s, yerr=err_s, fmt='o', capsize=3)
plt.xlabel("Thickness (x/X₀)")
plt.ylabel("Sigma (degrees)")
plt.title("Sigma vs Thickness")
plt.grid()
plt.tight_layout()
plt.savefig("sigma_vs_thickness.png")
plt.show()

# -----------------------------
# Plot 2: Sigma vs sqrt(Thickness)
# -----------------------------
plt.figure()
plt.errorbar(np.sqrt(t_s), sig_s, yerr=err_s, fmt='o', capsize=3)
plt.xlabel("Sqrt(Thickness)  √(x/X₀)")
plt.ylabel("Sigma (degrees)")
plt.title("Sigma vs sqrt(Thickness)")
plt.grid()
plt.tight_layout()
plt.savefig("sigma_vs_sqrt_thickness.png")
plt.show()

# -----------------------------
# Highland Formula (500 MeV electrons)
# FIX: thickness is already x/X0, do NOT divide by X0 again
# -----------------------------
p = 500.0   # MeV

def highland(x_over_X0, p=500.0):
    ratio = np.where(x_over_X0 > 0, x_over_X0, 1e-9)
    theta0_rad = (13.6 / p) * np.sqrt(ratio) * (1 + 0.038 * np.log(ratio))
    return np.degrees(theta0_rad)

highland_sigmas = highland(t_s, p=p)

# -----------------------------
# Plot 3: Comparison with Highland
# -----------------------------
plt.figure()
plt.plot(np.sqrt(t_s), sig_s,           'o',  label="Measured")
plt.plot(np.sqrt(t_s), highland_sigmas, '-',  label="Highland Prediction (500 MeV e⁻)")
plt.xlabel("Sqrt(Thickness)  √(x/X₀)")
plt.ylabel("Sigma (degrees)")
plt.title("Comparison with Highland Formula")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("sigma_vs_sqrt_with_theory.png")
plt.show()

# -----------------------------
# Save results table
# -----------------------------
results = np.column_stack((t_s, sig_s, err_s))
np.savetxt("phase1_results.txt",
           results,
           header="Thickness    Sigma(deg)    Sigma_error(deg)",
           fmt="%.6f")

print("\nPhase 1 complete.")
print("Results saved to 'phase1_results.txt'")
print("Plots saved as PNG files.")