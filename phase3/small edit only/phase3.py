import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.stats import linregress

# ─────────────────────────────────────────
# Load data — keeping per-threshold counts separate
# ─────────────────────────────────────────
raw = {}
with open('C:\Nikhilesh\Nikhilesh\Desktop\py_sktip\phase3\small_edit_only\Beamline_-_Events.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row['theta0'].strip() or not row['t (frac of X0)'].strip() or not row['p(theta>theta0)'].strip():
            continue
        theta0 = float(row['theta0'])
        t      = float(row['t (frac of X0)'])
        p      = float(row['p(theta>theta0)'])
        if theta0 not in raw:
            raw[theta0] = []
        raw[theta0].append((t, p))

thresholds = sorted(raw.keys())
colors = ['royalblue', 'darkorange', 'green', 'crimson', 'purple']
results = {}

print("="*60)
print("PHASE 3 — Power Law:  P(theta>theta0) ∝ t^alpha")
print("Fit method: unweighted log-log linear regression")
print("="*60)
print(f"{'Theta0':>10}   {'alpha':>8}   {'±se':>8}   {'R²':>8}   {'N pts':>6}")
print("-"*60)

for th in thresholds:
    data   = sorted(raw[th], key=lambda x: x[0])
    tvals  = np.array([x[0] for x in data])
    pvals  = np.array([x[1] for x in data])
    valid  = pvals > 0
    t_v    = tvals[valid]
    p_v    = pvals[valid]
    log_t  = np.log(t_v)
    log_p  = np.log(p_v)

    slope, intercept, r, _, se = linregress(log_t, log_p)
    r2 = r**2
    results[th] = (slope, se, r2, intercept)
    print(f"  theta>{th:4.0f} deg:   {slope:7.4f}   ±{se:.4f}   {r2:.5f}   {len(t_v):5d}")

print("="*60)

# ═══════════════════════════════════════════════
# PLOT 1 — Combined log-log all thresholds
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
for th, col in zip(thresholds, colors):
    data   = sorted(raw[th], key=lambda x: x[0])
    tvals  = np.array([x[0] for x in data])
    pvals  = np.array([x[1] for x in data])
    valid  = pvals > 0
    t_v = tvals[valid]; p_v = pvals[valid]
    alpha, se, r2, C = results[th]

    ax.scatter(np.log10(t_v), np.log10(p_v), color=col, s=30, zorder=3,
               label=f'theta > {int(th)} deg  (alpha={alpha:.4f} ± {se:.4f})')
    log_t_fit = np.linspace(np.log(t_v.min()), np.log(t_v.max()), 200)
    ax.plot(np.log10(np.exp(log_t_fit)),
            np.log10(np.exp(alpha * log_t_fit + C)),
            '-', color=col, lw=1.5, alpha=0.8)

ax.set_xlabel('log10( Thickness  x/X0 )', fontsize=12)
ax.set_ylabel('log10( P(theta > theta0) )', fontsize=12)
ax.set_title('Phase 3 — Power Law: P(θ > θ₀) ∝ t^α\n'
             'Log-log plot — straight lines confirm power law scaling',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase3_loglog_all.png', dpi=150)
print('Saved: phase3_loglog_all.png')

# ═══════════════════════════════════════════════
# PLOT 2 — Individual panels
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
for idx, (th, col) in enumerate(zip(thresholds, colors)):
    ax = axes[idx]
    data  = sorted(raw[th], key=lambda x: x[0])
    tvals = np.array([x[0] for x in data])
    pvals = np.array([x[1] for x in data])
    valid = pvals > 0
    t_v   = tvals[valid]; p_v = pvals[valid]
    alpha, se, r2, C = results[th]

    log_t_fit = np.linspace(np.log(t_v.min()), np.log(t_v.max()), 200)
    ax.scatter(np.log10(t_v), np.log10(p_v), color=col, s=40, zorder=3, label='Data')
    ax.plot(np.log10(np.exp(log_t_fit)),
            np.log10(np.exp(alpha*log_t_fit + C)),
            'k-', lw=2, label=f'alpha = {alpha:.4f} ± {se:.4f}')
    ax.set_xlabel('log10(t)', fontsize=10)
    ax.set_ylabel('log10(P)', fontsize=10)
    ax.set_title(f'theta > {int(th)} deg   R² = {r2:.4f}   (n={len(t_v)})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].axis('off')
plt.suptitle('Phase 3 — Individual Power Law Fits per Angle Threshold',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('phase3_loglog_individual.png', dpi=150)
print('Saved: phase3_loglog_individual.png')

# ═══════════════════════════════════════════════
# PLOT 3 — Alpha summary
# ═══════════════════════════════════════════════
alphas = [results[th][0] for th in thresholds]
errs   = [results[th][1] for th in thresholds]

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between([8, 22], 1.0, 2.0, alpha=0.10, color='gold',
                label='Moliere transition band  (1 < alpha < 2)')
ax.errorbar(thresholds, alphas, yerr=errs, fmt='o', color='steelblue',
            ms=10, lw=2, elinewidth=2, capsize=5, label='Measured alpha', zorder=5)
ax.axhline(1.0, color='red',    lw=2,   ls='--', label='alpha = 1  (pure single scattering)')
ax.axhline(2.0, color='purple', lw=1.5, ls=':',  label='alpha = 2  (Gaussian multiple scattering limit)')
for th, a in zip(thresholds, alphas):
    ax.annotate(f'alpha = {a:.3f}', xy=(th, a), xytext=(0, 14),
                textcoords='offset points', ha='center', fontsize=9, color='steelblue')
ax.set_xlabel('Angle threshold  theta0  (degrees)', fontsize=12)
ax.set_ylabel('Power law exponent  alpha', fontsize=12)
ax.set_title('Phase 3 & 4 — Alpha vs Threshold Angle\n'
             'All values in Moliere transition regime  (alpha approx 1.65)\n'
             'Consistent result across all thresholds confirms robust measurement',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(8, 22); ax.set_ylim(0, 2.5)
plt.tight_layout()
plt.savefig('phase3_alpha_summary.png', dpi=150)
print('Saved: phase3_alpha_summary.png')