import numpy as np
import matplotlib.pyplot as plt

def gauss_pdf(x, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma)**2)

data = {}
data[0.05] = np.array([
    [0.15,339733],[0.45,443767],[0.75,152929],[1.05,33163],[1.35,11043],
    [1.65,5483],[1.95,3264],[2.25,2134],[2.55,1414],[2.85,1057],[3.15,830],
    [3.45,583],[3.75,555],[4.05,404],[4.35,326],[4.65,324],[4.95,216],
    [5.25,208],[5.55,206],[5.85,158],[6.15,138],[6.45,146],[6.75,126],
    [7.05,109],[7.35,93],[7.65,85],[7.95,69],[8.25,71],[8.55,73],[8.85,66],
    [9.15,50],[9.45,59],[9.75,37],[10.05,42],[10.35,40],[10.65,47],
    [10.95,29],[11.25,29],[11.55,40],[11.85,21],[12.15,20],[12.45,24],
    [12.75,32],[13.05,20],[13.35,32],[13.65,19],[13.95,17],[14.25,23],
    [14.55,6],[14.85,14],[15.15,8],[15.45,21],[15.75,22],[16.05,13],
    [16.35,7],[16.65,19],[16.95,10],[17.25,14],[17.55,9],[17.85,16],
    [18.15,9],[18.45,10],[18.75,17],[19.05,12],[19.35,8],[19.65,7],
    [19.95,10]
])
data[0.20] = np.array([
    [0.15,79749],[0.45,196375],[0.75,223516],[1.05,181145],[1.35,117775],
    [1.65,66423],[1.95,36813],[2.25,21428],[2.55,13489],[2.85,9308],
    [3.15,6947],[3.45,5474],[3.75,4400],[4.05,3498],[4.35,2922],[4.65,2461],
    [4.95,2082],[5.25,1753],[5.55,1566],[5.85,1407],[6.15,1186],[6.45,1085],
    [6.75,984],[7.05,956],[7.35,827],[7.65,710],[7.95,695],[8.25,583],
    [8.55,573],[8.85,543],[9.15,482],[9.45,435],[9.75,448],[10.05,422],
    [10.35,373],[10.65,369],[10.95,355],[11.25,307],[11.55,321],[11.85,282],
    [12.15,262],[12.45,256],[12.75,241],[13.05,238],[13.35,206],[13.65,232],
    [13.95,213],[14.25,194],[14.55,159],[14.85,181],[15.15,158],[15.45,145],
    [15.75,143],[16.05,137],[16.35,144],[16.65,125],[16.95,124],[17.25,99],
    [17.55,122],[17.85,111],[18.15,121],[18.45,114],[18.75,103],[19.05,105],
    [19.35,86],[19.65,98],[19.95,85]
])
data[0.40] = np.array([
    [0.15,31176],[0.45,86096],[0.75,121411],[1.05,133800],[1.35,125368],
    [1.65,106021],[1.95,81918],[2.25,60551],[2.55,43778],[2.85,31742],
    [3.15,23403],[3.45,17742],[3.75,13977],[4.05,11168],[4.35,9182],
    [4.65,7876],[4.95,6661],[5.25,5741],[5.55,5025],[5.85,4362],[6.15,3847],
    [6.45,3510],[6.75,3064],[7.05,2848],[7.35,2609],[7.65,2399],[7.95,2134],
    [8.25,1898],[8.55,1864],[8.85,1686],[9.15,1508],[9.45,1423],[9.75,1379],
    [10.05,1244],[10.35,1249],[10.65,1173],[10.95,1051],[11.25,1070],
    [11.55,916],[11.85,907],[12.15,843],[12.45,852],[12.75,780],[13.05,734],
    [13.35,670],[13.65,656],[13.95,634],[14.25,616],[14.55,556],[14.85,561],
    [15.15,522],[15.45,545],[15.75,454],[16.05,502],[16.35,459],[16.65,418],
    [16.95,417],[17.25,404],[17.55,419],[17.85,386],[18.15,365],[18.45,346],
    [18.75,322],[19.05,324],[19.35,349],[19.65,323],[19.95,307]
])

sigma_cores = {0.05: 0.50092, 0.20: 1.27496, 0.40: 2.88771}
bin_width   = 0.30
thicknesses = [0.05, 0.20, 0.40]
colors      = ['royalblue', 'darkorange', 'green']

def compute_ratio(t):
    arr       = data[t]
    theta     = arr[:, 0]
    counts    = arr[:, 1]
    total     = np.sum(counts) * 2 * bin_width
    prob_dens = counts / total
    sc        = sigma_cores[t]
    g         = gauss_pdf(theta, sc)
    # Only valid where Gaussian is above 0.01% of its peak
    peak_g    = gauss_pdf(0, sc)
    valid     = g > peak_g * 0.0001
    R         = np.where(valid, prob_dens / g, np.nan)
    R_norm    = theta / sc
    return theta, prob_dens, g, R, R_norm, sc



for t, col in zip(thicknesses, colors):
    theta, prob_dens, g, R, R_norm, sc = compute_ratio(t)
    valid = ~np.isnan(R)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta[valid], R[valid], 'o-', color=col, ms=5, lw=1.5,
            label='R(theta) = Data / Gaussian')
    ax.axhline(1.0,  color='black', lw=1.5, ls='--', label='R = 1  (pure Gaussian)')
    ax.axvline(sc,   color='grey',  lw=1.0, ls=':',  label='1 sigma = ' + str(round(sc,3)) + ' deg')
    ax.axvline(2*sc, color='grey',  lw=1.0, ls='--', label='2 sigma')
    ax.axvline(3*sc, color='grey',  lw=1.0, ls='-',  label='3 sigma', alpha=0.5)
    ax.set_yscale('log')   # FIX
    ax.set_xlabel('Scattering angle  theta  (degrees)', fontsize=12)
    ax.set_ylabel('R(theta) = Data / Gaussian  (log scale)', fontsize=12)
    ax.set_title('Block 2 — Tail Ratio R(theta)  |  t = ' + str(t), fontsize=13, fontweight='bold')
    ax.set_xlim(0, None)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    tname = str(t).replace('.', '')
    plt.savefig('phase2_block2_Rtheta_fixed_t' + tname + '.png', dpi=150)
    plt.show()
    print('Saved: phase2_block2_Rtheta_fixed_t' + tname + '.png')



fig, ax = plt.subplots(figsize=(9, 6))
for t, col in zip(thicknesses, colors):
    theta, prob_dens, g, R, R_norm, sc = compute_ratio(t)
    valid = ~np.isnan(R)
    ax.plot(theta[valid], R[valid], 'o-', color=col, ms=5, lw=1.5, label='t = ' + str(t))

ax.axhline(1.0, color='black', lw=2, ls='--', label='R = 1')
ax.set_yscale('log')   # FIX
ax.set_xlabel('Scattering angle  theta  (degrees)', fontsize=12)
ax.set_ylabel('R(theta) = Data / Gaussian  (log scale)', fontsize=12)
ax.set_title('Block 2 — Tail Ratio R(theta)  |  All 3 thicknesses', fontsize=13, fontweight='bold')
ax.set_xlim(0, 20)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('phase2_block2_Rtheta_fixed_combined.png', dpi=150)
plt.show()
print('Saved: phase2_block2_Rtheta_fixed_combined.png')



for t, col in zip(thicknesses, colors):
    theta, prob_dens, g, R, R_norm, sc = compute_ratio(t)
    valid = ~np.isnan(R)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(R_norm[valid], R[valid], 'o-', color=col, ms=5, lw=1.5,
            label='R vs theta/sigma_core')
    ax.axhline(1.0, color='black', lw=1.5, ls='--', label='R = 1')
    ax.axvline(1.0, color='grey',  lw=1.0, ls=':')
    ax.axvline(2.0, color='grey',  lw=1.0, ls='--')
    ax.axvline(3.0, color='grey',  lw=1.0, ls='-', alpha=0.5)
    ax.text(1.05, 1.5, '1s', fontsize=9, color='grey')
    ax.text(2.05, 1.5, '2s', fontsize=9, color='grey')
    ax.text(3.05, 1.5, '3s', fontsize=9, color='grey')
    ax.set_yscale('log')   # FIX
    ax.set_xlabel('Normalised angle  theta / sigma_core', fontsize=12)
    ax.set_ylabel('R = Data / Gaussian  (log scale)', fontsize=12)
    ax.set_title('Block 2 — Normalised Tail Ratio  |  t = ' + str(t), fontsize=13, fontweight='bold')
    ax.set_xlim(0, None)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    tname = str(t).replace('.', '')
    plt.savefig('phase2_block2_Rnorm_fixed_t' + tname + '.png', dpi=150)
    plt.show()
    print('Saved: phase2_block2_Rnorm_fixed_t' + tname + '.png')



fig, ax = plt.subplots(figsize=(9, 6))
for t, col in zip(thicknesses, colors):
    theta, prob_dens, g, R, R_norm, sc = compute_ratio(t)
    valid = ~np.isnan(R)
    ax.plot(R_norm[valid], R[valid], 'o-', color=col, ms=5, lw=1.5, label='t = ' + str(t))

ax.axhline(1.0, color='black', lw=2,   ls='--', label='R = 1')
ax.axvline(1.0, color='grey',  lw=1.0, ls=':')
ax.axvline(2.0, color='grey',  lw=1.0, ls='--')
ax.axvline(3.0, color='grey',  lw=1.0, ls='-', alpha=0.5)
ax.text(1.05, 1.5, '1 sigma', fontsize=9, color='grey')
ax.text(2.05, 1.5, '2 sigma', fontsize=9, color='grey')
ax.text(3.05, 1.5, '3 sigma', fontsize=9, color='grey')
ax.set_yscale('log')   # FIX
ax.set_xlabel('Normalised angle  theta / sigma_core', fontsize=12)
ax.set_ylabel('R = Data / Gaussian  (log scale)', fontsize=12)
ax.set_title('Block 2 — Normalised Tail Ratio  |  All thicknesses\n(Moliere signature)',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 12)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('phase2_block2_Rnorm_fixed_combined.png', dpi=150)
plt.show()
print('Saved: phase2_block2_Rnorm_fixed_combined.png')

print('\nBlock 2 fixed complete!')
print('Total plots: 8')
print('  R vs theta     : Rtheta_fixed_t005, t02, t04, combined')
print('  R vs theta/sig : Rnorm_fixed_t005,  t02, t04, combined')