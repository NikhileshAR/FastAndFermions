import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, sigma):
    return np.exp(-(x**2)/(2*sigma**2))

def process_dataset(theta, counts, sigma_core, sigma_hl, thickness):

    prob = counts / np.sum(counts)

    theta_full = np.concatenate((-theta[::-1], theta))
    prob_full  = np.concatenate((prob[::-1], prob))

    g_core = gaussian(theta_full, sigma_core)
    g_hl   = gaussian(theta_full, sigma_hl)

    g_core *= prob_full.max() / g_core.max()
    g_hl   *= prob_full.max() / g_hl.max()

    return theta_full, prob_full, g_core, g_hl


# =========================
# DATA
# =========================

# ---- 0.05 ----
theta_005 = np.array([0.15,0.45,0.75,1.05,1.35,1.65,1.95,2.25,2.55,2.85,
3.15,3.45,3.75,4.05,4.35,4.65,4.95,5.25,5.55,5.85])

counts_005 = np.array([
339733,443767,152929,33163,11043,5483,3264,2134,1414,1057,
830,583,555,404,326,324,216,208,206,158
])

# ---- 0.20 ----
theta_020 = np.array([0.15,0.45,0.75,1.05,1.35,1.65,1.95,2.25,2.55,2.85,
3.15,3.45,3.75,4.05,4.35,4.65,4.95,5.25,5.55,5.85])

counts_020 = np.array([
79749,196375,223516,181145,117775,66423,36813,21428,13489,9308,
6947,5474,4400,3498,2922,2461,2082,1753,1566,1407
])

# ---- 0.40 ----
theta_040 = np.array([0.15,0.45,0.75,1.05,1.35,1.65,1.95,2.25,2.55,2.85,
3.15,3.45,3.75,4.05,4.35,4.65,4.95,5.25,5.55,5.85])

counts_040 = np.array([
31176,86096,121411,133800,125368,106021,81918,60551,43778,31742,
23403,17742,13977,11168,9182,7876,6661,5741,5025,4362
])

# =========================
# SIGMA VALUES
# =========================

sigma_005_core = 0.50092
sigma_005_hl   = 0.30881

sigma_020_core = 1.27496
sigma_020_hl   = 0.65433

sigma_040_core = 2.88771
sigma_040_hl   = 0.95133


# =========================
# PROCESS ALL
# =========================

data_005 = process_dataset(theta_005, counts_005, sigma_005_core, sigma_005_hl, 0.05)
data_020 = process_dataset(theta_020, counts_020, sigma_020_core, sigma_020_hl, 0.20)
data_040 = process_dataset(theta_040, counts_040, sigma_040_core, sigma_040_hl, 0.40)


# =========================
# 1️⃣ SEPARATE FIGURES
# =========================

for data, t in zip([data_005, data_020, data_040], [0.05,0.20,0.40]):
    theta_full, prob_full, g_core, g_hl = data

    plt.figure()
    plt.scatter(theta_full, prob_full, s=8, label="Data")
    plt.plot(theta_full, g_core, label="Gaussian σ_core")
    plt.plot(theta_full, g_hl, linestyle="--", label="Highland")

    plt.yscale("log")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Probability")
    plt.title(f"t = {t}")
    plt.legend()
    plt.grid()
    plt.show()


# =========================
# 2️⃣ STACKED COMPARISON
# =========================

fig, axes = plt.subplots(3, 1, sharex=True)

for ax, data, t in zip(axes, 
                       [data_005, data_020, data_040], 
                       [0.05,0.20,0.40]):
    
    theta_full, prob_full, g_core, g_hl = data
    
    ax.scatter(theta_full, prob_full, s=6)
    ax.plot(theta_full, g_core)
    ax.plot(theta_full, g_hl, linestyle="--")
    
    ax.set_yscale("log")
    ax.set_title(f"t = {t}")
    ax.grid()

axes[-1].set_xlabel("Angle (deg)")
plt.tight_layout()
plt.show()