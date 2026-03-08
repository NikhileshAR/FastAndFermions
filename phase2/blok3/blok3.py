import numpy as np
import matplotlib.pyplot as plt


def highland(x, p=500.0):
    ratio = np.where(x > 0, x, 1e-9)
    return np.degrees((13.6 / p) * np.sqrt(ratio) * (1 + 0.038 * np.log(ratio)))


all_t = np.array([
    0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,
    0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,
    0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,
    0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40
])

all_sigma_core = np.array([
    0.21560,0.30625,0.37792,0.44238,0.50092,0.55617,0.61270,0.67279,0.73571,0.80869,
    0.81871,0.86596,0.91155,0.95740,1.00637,1.05410,1.10608,1.15711,1.21644,1.27496,
    1.33888,1.40735,1.48394,1.56106,1.66041,1.53241,1.58385,1.65028,1.71138,1.77718,
    1.85115,1.92514,2.00424,2.09707,2.18905,2.29966,2.43346,2.54982,2.71340,2.88771
])

all_sigma_hl = highland(all_t)


ratio = all_sigma_core / all_sigma_hl

print("sigma_core / sigma_Highland summary:")
print("  Min ratio : " + str(round(ratio.min(), 4)) + "  at t = " + str(all_t[ratio.argmin()]))
print("  Max ratio : " + str(round(ratio.max(), 4)) + "  at t = " + str(all_t[ratio.argmax()]))
print("  Mean ratio: " + str(round(ratio.mean(), 4)))


fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(all_t, ratio, 'o-', color='purple', ms=6, lw=2, label='sigma_core / sigma_Highland')
ax.axhline(1.0, color='black', lw=1.5, ls='--', label='Ratio = 1  (perfect Highland agreement)')
ax.fill_between(all_t, 1.0, ratio, alpha=0.15, color='purple', label='Excess above Highland')


ax.axhline(ratio.mean(), color='purple', lw=1.0, ls=':', alpha=0.7,
           label='Mean ratio = ' + str(round(ratio.mean(), 3)))

ax.set_xlabel('Thickness  (x/X0)', fontsize=12)
ax.set_ylabel('sigma_core / sigma_Highland', fontsize=12)
ax.set_title('Block 3 — Core Width Ratio vs Thickness\n'
             'sigma_core always > sigma_Highland proves non-Gaussian even in core',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, None)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase2_block3_ratio_vs_thickness.png', dpi=150)
plt.show()
print('Saved: phase2_block3_ratio_vs_thickness.png')



fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(all_t, all_sigma_core, 'o-', color='royalblue', ms=5, lw=2,
        label='sigma_core  (measured, Gaussian fit)')
ax.plot(all_t, all_sigma_hl,   's-', color='red',       ms=5, lw=2,
        label='sigma_Highland  (theory, 500 MeV e-)')
ax.fill_between(all_t, all_sigma_hl, all_sigma_core,
                alpha=0.15, color='royalblue', label='Gap = tail excess contribution')

ax.set_xlabel('Thickness  (x/X0)', fontsize=12)
ax.set_ylabel('sigma  (degrees)', fontsize=12)
ax.set_title('Block 3 — Measured Core Width vs Highland Prediction\n'
             'Gap between curves grows with thickness',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase2_block3_sigma_comparison.png', dpi=150)
plt.show()
print('Saved: phase2_block3_sigma_comparison.png')



fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(np.sqrt(all_t), ratio, 'o-', color='darkorange', ms=6, lw=2,
        label='sigma_core / sigma_Highland')
ax.axhline(1.0, color='black', lw=1.5, ls='--', label='Ratio = 1')
ax.axhline(ratio.mean(), color='darkorange', lw=1.0, ls=':',
           label='Mean = ' + str(round(ratio.mean(), 3)))
ax.fill_between(np.sqrt(all_t), 1.0, ratio, alpha=0.15, color='darkorange')

ax.set_xlabel('sqrt(Thickness)  sqrt(x/X0)', fontsize=12)
ax.set_ylabel('sigma_core / sigma_Highland', fontsize=12)
ax.set_title('Block 3 — Width Ratio vs sqrt(Thickness)\n'
             'Flat ratio would mean both scale identically with sqrt(t)',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, None)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase2_block3_ratio_vs_sqrtt.png', dpi=150)
plt.show()
print('Saved: phase2_block3_ratio_vs_sqrtt.png')

print('\nBlock 3 complete!')
print('Total plots saved: 3')
print('  phase2_block3_ratio_vs_thickness.png')
print('  phase2_block3_sigma_comparison.png')
print('  phase2_block3_ratio_vs_sqrtt.png')