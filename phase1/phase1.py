import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ─────────────────────────────────────────
# Phase 1 results (from phase1_results.txt)
# ─────────────────────────────────────────
all_t = np.array([
    0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,
    0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,
    0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,
    0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,
    0.41,0.42,0.43,0.44,0.45
])

all_sigma = np.array([
    0.215602,0.306250,0.377925,0.442378,0.500922,0.556174,0.612705,0.672786,
    0.735711,0.808690,0.818706,0.865963,0.911546,0.957399,1.006369,1.054101,
    1.106078,1.157114,1.216443,1.274964,1.338880,1.407353,1.483940,1.561059,
    1.660406,1.532412,1.583846,1.650284,1.711384,1.777177,1.851152,1.925136,
    2.004235,2.097066,2.189046,2.299661,2.433461,2.549819,2.713401,2.887710,
    3.078882,3.348824,3.634832,3.977902,4.537495
])

all_err = np.array([
    0.000114,0.000150,0.000187,0.000221,0.000258,0.000302,0.000360,0.000439,
    0.000540,0.000685,0.000445,0.000484,0.000527,0.000576,0.000634,0.000697,
    0.000776,0.000860,0.000969,0.001088,0.001233,0.001406,0.001619,0.001859,
    0.002208,0.001262,0.001368,0.001519,0.001666,0.001842,0.002055,0.002287,
    0.002555,0.002900,0.003275,0.003770,0.004432,0.005083,0.006087,0.007307,
    0.008831,0.011335,0.014462,0.018939,0.028063
])

sqrt_t = np.sqrt(all_t)

# Highland formula
def highland(x, p=500.0):
    return np.degrees((13.6 / p) * np.sqrt(x) * (1 + 0.038 * np.log(np.where(x>0, x, 1e-9))))

all_hl = highland(all_t)

# ─────────────────────────────────────────
# Linear fit on the LINEAR regime only
# t = 0.01 to 0.25 is clearly linear
# ─────────────────────────────────────────
linear_mask = all_t <= 0.25
slope, intercept, r, p_val, se = linregress(sqrt_t[linear_mask], all_sigma[linear_mask])

print("Linear fit (t <= 0.25):  sigma = " + str(round(slope,4)) + " * sqrt(t) + " + str(round(intercept,5)))
print("R^2 = " + str(round(r**2, 5)))

fit_x   = np.linspace(0, sqrt_t.max(), 300)
fit_y   = slope * fit_x + intercept

# ═══════════════════════════════════════════════
# PLOT 1 — sigma vs sqrt(t)  WITH linear fit
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

ax.errorbar(sqrt_t, all_sigma, yerr=all_err,
            fmt='o', color='royalblue', ms=5, lw=1, elinewidth=1,
            label='Measured sigma_core')

ax.plot(fit_x, fit_y, 'r-', lw=2,
        label='Linear fit (t <= 0.25):  sigma = ' + str(round(slope,3)) + ' * sqrt(t)  (R² = ' + str(round(r**2,4)) + ')')

ax.plot(sqrt_t, all_hl, 's--', color='darkorange', ms=4, lw=1.5,
        label='Highland prediction (500 MeV e-)')

# Mark the linear regime boundary
ax.axvline(np.sqrt(0.25), color='grey', lw=1, ls=':', alpha=0.8)
ax.text(np.sqrt(0.25)+0.005, 0.3, 't = 0.25\n(linear limit)', fontsize=9, color='grey')

# Mark where instability begins
ax.axvline(np.sqrt(0.40), color='red', lw=1, ls='--', alpha=0.6)
ax.text(np.sqrt(0.40)+0.005, 0.3, 't = 0.40\n(non-Gaussian\n regime)', fontsize=9, color='red')

ax.set_xlabel('sqrt(Thickness)   sqrt(x/X0)', fontsize=12)
ax.set_ylabel('sigma_core  (degrees)', fontsize=12)
ax.set_title('Phase 1 — sigma vs sqrt(Thickness)\n'
             'Linear regime confirms sqrt(t) multiple scattering scaling',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase1_sigma_vs_sqrtt_fixed.png', dpi=150)
plt.show()
print('Saved: phase1_sigma_vs_sqrtt_fixed.png')


# ═══════════════════════════════════════════════
# PLOT 2 — sigma vs thickness  WITH fit overlay
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

ax.errorbar(all_t, all_sigma, yerr=all_err,
            fmt='o', color='royalblue', ms=5, lw=1, elinewidth=1,
            label='Measured sigma_core')

t_smooth = np.linspace(0.001, 0.25, 300)
ax.plot(t_smooth, slope * np.sqrt(t_smooth) + intercept, 'r-', lw=2,
        label='Linear fit: sigma = ' + str(round(slope,3)) + ' * sqrt(t)  (valid for t <= 0.25)')

ax.plot(all_t, all_hl, 's--', color='darkorange', ms=4, lw=1.5,
        label='Highland prediction (500 MeV e-)')

ax.axvline(0.25, color='grey', lw=1, ls=':', alpha=0.8)
ax.text(0.255, 0.3, 't = 0.25\n(linear\nlimit)', fontsize=9, color='grey')
ax.axvline(0.40, color='red', lw=1, ls='--', alpha=0.6)
ax.text(0.405, 0.3, 't = 0.40\n(non-Gaussian\nregime)', fontsize=9, color='red')

ax.set_xlabel('Thickness   (x/X0)', fontsize=12)
ax.set_ylabel('sigma_core  (degrees)', fontsize=12)
ax.set_title('Phase 1 — sigma vs Thickness\n'
             'With sqrt(t) fit and Highland comparison',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase1_sigma_vs_thickness_fixed.png', dpi=150)
plt.show()
print('Saved: phase1_sigma_vs_thickness_fixed.png')

print('\nPhase 1 fix complete!')