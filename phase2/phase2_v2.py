import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# Normalised Gaussian (probability density)
# ─────────────────────────────────────────
def gauss_pdf(x, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma)**2)

# ─────────────────────────────────────────
# FULL DATA
# ─────────────────────────────────────────
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
sigma_hls   = {0.05: 0.30881, 0.20: 0.65433, 0.40: 0.95133}
bin_width   = 0.30

def get_xlim(theta, counts):
    peak = counts.max()
    meaningful = theta[counts >= 0.001 * peak]
    return min(meaningful[-1] * 1.3, 20) if len(meaningful) > 0 else 10

def get_cut(t):
    if t <= 0.10: return 1.5
    elif t <= 0.25: return 2.5
    else: return 3.0

def make_panel(ax, t):
    arr    = data[t]
    theta  = arr[:, 0]
    counts = arr[:, 1]

    theta_full  = np.concatenate([-theta[::-1], theta])
    counts_full = np.concatenate([ counts[::-1], counts])

    total     = np.sum(counts_full) * bin_width
    prob_dens = counts_full / total
    mask      = prob_dens > 0

    sc = sigma_cores[t]
    sh = sigma_hls[t]
    x_smooth = np.linspace(-20, 20, 2000)
    g_core   = gauss_pdf(x_smooth, sc)
    g_hl     = gauss_pdf(x_smooth, sh)

    xlim = get_xlim(theta, counts)
    ymin = prob_dens[mask].min() * 0.1
    ymax = prob_dens[mask].max() * 5
    cut  = get_cut(t)

    ax.scatter(theta_full[mask], prob_dens[mask],
               s=10, color='royalblue', zorder=3, label='Data')
    ax.plot(x_smooth, g_core, 'r-',  lw=2.0, label='Gaussian core  s = ' + str(sc) + ' deg')
    ax.plot(x_smooth, g_hl,   'g--', lw=2.0, label='Highland         s = ' + str(sh) + ' deg')
    ax.axvline( cut, color='grey', lw=1, ls=':', alpha=0.7)
    ax.axvline(-cut, color='grey', lw=1, ls=':', alpha=0.7)
    ax.text(cut + 0.1, ymax * 0.3, 'core\ncut', fontsize=8, color='grey')
    ax.set_yscale('log')
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(ymin, ymax)
    ax.set_title('t = ' + str(t) + '  (x/X0)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability density (deg-1)', fontsize=11)
    ax.legend(fontsize=10, loc='lower center')
    ax.grid(True, which='both', alpha=0.3)

# =========================
# 1 - INDIVIDUAL FIGURES
# =========================
for t in [0.05, 0.20, 0.40]:
    fig, ax = plt.subplots(figsize=(8, 5))
    make_panel(ax, t)
    ax.set_xlabel('Scattering angle  (degrees)', fontsize=12)
    plt.suptitle('Phase 2 - Block 1 | Log-Scale Distribution | 500 MeV electrons',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    tname = str(t).replace('.', '')
    plt.savefig('phase2_block1_t' + tname + '.png', dpi=150)
    plt.show()

# =========================
# 2 - STACKED FIGURE
# =========================
fig, axes = plt.subplots(3, 1, figsize=(9, 13))
for ax, t in zip(axes, [0.05, 0.20, 0.40]):
    make_panel(ax, t)
axes[-1].set_xlabel('Scattering angle  (degrees)', fontsize=12)
plt.suptitle('Phase 2 - Block 1\nLog-Scale Angular Distribution vs Gaussian\n500 MeV electrons',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('phase2_block1_stacked.png', dpi=150, bbox_inches='tight')
plt.show()

print("Done! Saved: phase2_block1_t005.png, t020.png, t040.png, and stacked.")