import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


data_10 = np.array([
    [0.01, 7.00e-05],
    [0.02, 0.00022],
    [0.03, 0.00047],
    [0.04, 0.00069],
    [0.05, 0.00089],
    [0.06, 0.00133],
    [0.07, 0.00161],
    [0.08, 0.00224],
    [0.09, 0.00238],
    [0.10, 0.00285],
    [0.11, 0.00349],
    [0.12, 0.00401],
    [0.13, 0.00499],
    [0.14, 0.00528],
    [0.15, 0.00620],
    [0.16, 0.00726],
    [0.17, 0.00789],
    [0.18, 0.00794],
    [0.19, 0.00928],
    [0.20, 0.01034],
    [0.21, 0.01071],
    [0.22, 0.01151],
    [0.23, 0.01322],
    [0.24, 0.01370],
    [0.25, 0.01565],
    [0.26, 0.01598],
    [0.27, 0.01790],
    [0.28, 0.01844],
    [0.29, 0.01902],
    [0.30, 0.01956],
    [0.31, 0.02255],
    [0.32, 0.02325],
    [0.33, 0.02401],
    [0.34, 0.02633],
    [0.35, 0.02652],
    [0.36, 0.02897],
    [0.37, 0.03025],
    [0.38, 0.03157],
    [0.39, 0.03322],
    [0.40, 0.03405],
    [0.41, 0.03654],
    [0.42, 0.03709],
    [0.43, 0.03785],
    [0.44, 0.04010],
    [0.45, 0.04092],
    [0.46, 0.04309],
    [0.47, 0.04551],
    [0.48, 0.04709],
    [0.49, 0.04948],
    [0.50, 0.05114],
])


data_12 = np.array([
    [0.01, 8.00e-05],
    [0.05, 0.000738],
    [0.10, 0.002426],
    [0.15, 0.004802],
    [0.20, 0.008073],
    [0.25, 0.011665],
    [0.30, 0.015917],
    [0.35, 0.021126],
    [0.40, 0.027375],
    [0.45, 0.033512],
    [0.50, 0.039916],
])

data_15 = np.array([
    [0.01, 5.40e-05],
    [0.05, 0.000513],
    [0.10, 0.001721],
    [0.15, 0.003516],
    [0.20, 0.006001],
    [0.25, 0.008620],
    [0.30, 0.011768],
    [0.35, 0.015578],
    [0.40, 0.020298],
    [0.45, 0.024842],
    [0.50, 0.029690],
])

data_18 = np.array([
    [0.01, 4.20e-05],
    [0.05, 0.000411],
    [0.10, 0.001310],
    [0.15, 0.002659],
    [0.20, 0.004618],
    [0.25, 0.006677],
    [0.30, 0.009089],
    [0.35, 0.011961],
    [0.40, 0.015682],
    [0.45, 0.019284],
    [0.50, 0.022825],
])

data_20 = np.array([
    [0.01, 2.90e-05],
    [0.05, 0.000345],
    [0.10, 0.001118],
    [0.15, 0.002292],
    [0.20, 0.003972],
    [0.25, 0.005701],
    [0.30, 0.007774],
    [0.35, 0.010209],
    [0.40, 0.013436],
    [0.45, 0.016468],
    [0.50, 0.019551],
])


datasets = {
    10: data_10,
    12: data_12,
    15: data_15,
    18: data_18,
    20: data_20,
}

colors = {
    10: 'royalblue',
    12: 'darkorange',
    15: 'crimson',
    18: 'purple',
    20: 'seagreen',
}


alpha_results = {}

print("=" * 55)
print("Phase 3 — Power Law Fit Results")
print("=" * 55)

for theta0, arr in datasets.items():
    t_vals = arr[:, 0]
    p_vals = arr[:, 1]

    # Remove any zero P values before log
    mask   = p_vals > 0
    t_vals = t_vals[mask]
    p_vals = p_vals[mask]

    log_t = np.log(t_vals)
    log_p = np.log(p_vals)

    slope, intercept, r_value, p_value, std_err = linregress(log_t, log_p)

    alpha_results[theta0] = {
        'alpha':     slope,
        'alpha_err': std_err,
        'intercept': intercept,
        'r2':        r_value**2,
        't':         t_vals,
        'p':         p_vals,
    }

    print("Theta0 = {:2d} deg  -->  alpha = {:.4f} +/- {:.4f}   (R2 = {:.5f})".format(
        theta0, slope, std_err, r_value**2))

print("=" * 55)


fig, ax = plt.subplots(figsize=(10, 7))

for theta0, res in alpha_results.items():
    t_vals = res['t']
    p_vals = res['p']
    alpha  = res['alpha']
    C      = res['intercept']
    col    = colors[theta0]

    # Data points
    ax.scatter(t_vals, p_vals, s=25, color=col, zorder=3,
               label='theta0 = {:d} deg  (alpha = {:.3f})'.format(theta0, alpha))

    # Fit line
    t_smooth = np.linspace(t_vals.min(), t_vals.max(), 200)
    p_fit    = np.exp(C) * t_smooth**alpha
    ax.plot(t_smooth, p_fit, '-', color=col, lw=1.5, alpha=0.7)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Thickness  (x/X0)  [log scale]', fontsize=12)
ax.set_ylabel('P(theta > theta0)  [log scale]', fontsize=12)
ax.set_title('Phase 3 — Power Law Scaling\n'
             'P(theta > theta0) proportional to t^alpha\n'
             'Slopes > 1 confirm Moliere transition regime',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('phase3_loglog_all.png', dpi=150)
plt.show()
print('Saved: phase3_loglog_all.png')


fig, axes = plt.subplots(3, 2, figsize=(13, 14))
axes = axes.flatten()

for idx, (theta0, res) in enumerate(alpha_results.items()):
    ax     = axes[idx]
    t_vals = res['t']
    p_vals = res['p']
    alpha  = res['alpha']
    aerr   = res['alpha_err']
    C      = res['intercept']
    r2     = res['r2']
    col    = colors[theta0]

    # Data
    ax.scatter(t_vals, p_vals, s=30, color=col, zorder=3, label='Data')

    # Fit line
    t_smooth = np.linspace(t_vals.min(), t_vals.max(), 200)
    p_fit    = np.exp(C) * t_smooth**alpha
    ax.plot(t_smooth, p_fit, '-', color=col, lw=2,
            label='Fit: alpha = {:.4f} +/- {:.4f}'.format(alpha, aerr))

    # Reference line alpha=1 through first data point
    p_ref = np.exp(C) * t_smooth**1.0
    ax.plot(t_smooth, p_ref, 'k--', lw=1.5, alpha=0.6, label='alpha = 1  (single scatter)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Thickness  (x/X0)', fontsize=11)
    ax.set_ylabel('P(theta > theta0)', fontsize=11)
    ax.set_title('theta0 = {:d} deg   R2 = {:.4f}'.format(theta0, r2),
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)


axes[5].set_visible(False)

plt.suptitle('Phase 3 — Individual Log-Log Fits\n'
             '500 MeV electrons on Aluminium',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('phase3_loglog_individual.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: phase3_loglog_individual.png')


theta0_vals = np.array(sorted(alpha_results.keys()))
alpha_vals  = np.array([alpha_results[t]['alpha']     for t in theta0_vals])
alpha_errs  = np.array([alpha_results[t]['alpha_err'] for t in theta0_vals])

fig, ax = plt.subplots(figsize=(9, 6))

ax.errorbar(theta0_vals, alpha_vals, yerr=alpha_errs,
            fmt='o', color='royalblue', ms=8, elinewidth=2,
            capsize=5, capthick=2, zorder=3,
            label='Measured alpha')

# Reference lines
ax.axhline(1.0, color='red',   lw=2,   ls='--', label='alpha = 1  (pure single scattering)')
ax.axhline(0.5, color='green', lw=1.5, ls=':',  label='alpha = 0.5  (pure multiple scattering)')

# Shade the Moliere transition band
ax.axhspan(1.0, 2.0, alpha=0.08, color='gold', label='Moliere transition band  (1 < alpha < 2)')

for t0, a, ae in zip(theta0_vals, alpha_vals, alpha_errs):
    ax.annotate('alpha = {:.3f}'.format(a),
                xy=(t0, a), xytext=(5, 8),
                textcoords='offset points', fontsize=9, color='royalblue')

ax.set_xlabel('Angle threshold  theta0  (degrees)', fontsize=12)
ax.set_ylabel('Power law exponent  alpha', fontsize=12)
ax.set_title('Phase 3 & 4 — Alpha vs Threshold Angle\n'
             'All values in Moliere transition regime  (alpha approx 1.65)\n'
             'Consistent result across all thresholds confirms robust measurement',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.set_ylim(0, 2.5)
ax.set_xlim(8, 22)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase3_alpha_vs_theta0.png', dpi=150)
plt.show()
print('Saved: phase3_alpha_vs_theta0.png')


print('\nSummary Table:')
print('-' * 45)
print('{:>10}  {:>12}  {:>12}  {:>8}'.format('Theta0', 'Alpha', 'Uncertainty', 'R2'))
print('-' * 45)
for t0 in theta0_vals:
    r = alpha_results[t0]
    print('{:>8} deg  {:>12.4f}  {:>12.4f}  {:>8.5f}'.format(
        t0, r['alpha'], r['alpha_err'], r['r2']))
print('-' * 45)


out = np.column_stack((theta0_vals, alpha_vals, alpha_errs))
np.savetxt('phase3_results.txt', out,
           header='Theta0(deg)    Alpha    Alpha_err',
           fmt='%.6f')
print('\nPhase 3 complete.')
print('Saved: phase3_results.txt')
print('Plots: phase3_loglog_all.png')
print('       phase3_loglog_individual.png')
print('       phase3_alpha_vs_theta0.png  <-- this is also your Phase 4 main result')