import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from utils import init

fig, axs = plt.subplots(3, 2, figsize=(10,12))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
blue = colors[0]
coords = {"LunarLanderContinuous-v2": (0.42, 100),
          "Swimmer-v4": (0.3, 250),
          "Hopper-v4": (0.33, 900),
          "Walker2d-v4": (0.33, 2500),
          "HalfCheetah-v4": (0.17, 8000),
          "Ant-v4": (0.0, 2000)}

for num, env in enumerate(["LunarLanderContinuous-v2",
                           "Swimmer-v4",
                           "Hopper-v4",
                           "Walker2d-v4",
                           "HalfCheetah-v4",
                           "Ant-v4"]):

    ax = axs[num//2][num%2]

    DIR = f"./Results/{env}/FI/"
    _, _, _, act_dim, feature_names = init(env)

    means = np.load(f"{DIR}means.npy")
    stds = np.load(f"{DIR}stds.npy")
    ref = np.load(f"{DIR}ref_performance.npy")
    ref_mean = ref[0]
    ref_std = ref[1]
    fi = np.load(f"{DIR}fi.npy")
    print(f"{env}: Performance fully observable: $\overline{{R}} = {ref_mean} \pm {ref_std}$")

    x_min = -0.07*(max(fi)-min(fi))
    x_max = max(fi)+0.07*(max(fi)-min(fi))

    coef = np.polyfit(fi, means, 1)
    poly1d_func = np.poly1d(coef)
    x_lin = np.linspace(x_min, x_max, 100)
    y_lin = poly1d_func(x_lin)
    ax.plot(x_lin, y_lin, "--", c=blue, alpha=0.3)

    r = pearsonr(fi, means)
    print(f"Correlation r={r[0]}")

    # Coefficient of determination
    r2 = r2_score(means, poly1d_func(fi))
    print(f"Coefficient of determination R²={r2}")

    ax.errorbar(fi, means, yerr=stds, fmt="o", c=blue, label="one observable hidden")
    for i in range(len(fi)):
        xytext = (0.3, 0.3)
        # Beautify for special cases
        if env == "LunarLanderContinuous-v2" and i == 1:
            xytext = (-0.9, -0.7)
        elif env == "LunarLanderContinuous-v2" and i == 7:
            xytext = (-2, -1.1)
        elif env == "Swimmer-v4" and i == 1:
            xytext = (0.5, -0.3)
        elif env == "Swimmer-v4" and i == 2:
            xytext = (-1.5, 0.7)
        elif env == "Swimmer-v4" and i == 3:
            xytext = (-1.5, -0.7)
        elif env == "Swimmer-v4" and i == 5:
            xytext = (0.3, -1)
        elif env == "Swimmer-v4" and i == 6:
            xytext = (-0.3, 0.6)
        elif env == "Hopper-v4" and i == 6:
            xytext = (-2, 0.7)
        elif env == "Walker2d-v4" and i == 2:
            xytext = (0.3, -0.7)
        elif env == "Walker2d-v4" and i == 5:
            xytext = (-1.6, -1.3)
        elif env == "Walker2d-v4" and i == 7:
            xytext = (0.4, -0.1)
        elif env == "Walker2d-v4" and i == 10:
            xytext = (-1.7, 0.7)
        elif env == "Walker2d-v4" and i == 14:
            xytext = (-1.3, -1.3)

        ax.annotate(feature_names[i], xy=(fi[i], means[i]), xytext=xytext, textcoords="offset fontsize")

    ax.axhline(y=ref_mean, color="g", linestyle="-.", label="all observables visible")
    ax.fill_between(np.linspace(x_min, x_max), ref_mean-ref_std, ref_mean+ref_std, color="g", alpha=0.1)

    with open('random_baseline.pkl', 'rb') as f:
        random_baseline = pickle.load(f)
    ax.axhline(y=random_baseline[env + "_mean"], color="r", linestyle="--", label="random actions")
    ax.fill_between(np.linspace(x_min, x_max), random_baseline[env + "_mean"]-random_baseline[env + "_std"], random_baseline[env + "_mean"]+random_baseline[env + "_std"], color="r", alpha=0.1)

    ax.annotate(f"$r={{{r[0]:.3f}}}$\n$R^2={{{r2:.3f}}}$", xy=coords[env], xytext=(0,0), textcoords="offset fontsize", color=blue, alpha=0.7)

    if num//2 == 2:
        ax.set_xlabel(r"$\mathrm{FI}_j$")
    if num%2 == 0:
        ax.set_ylabel(r"$\overline{R}_{\setminus j}$")

    if num == 2:
        print(ax.get_legend_handles_labels())
        handles, labels = ax.get_legend_handles_labels()
        order = [0, 2, 1]
        print(handles, labels)
        ax.legend([handles[idx] for idx in order],
                  [labels[idx] for idx in order],
                  loc="lower left", bbox_to_anchor=(0.05, 0.5))

    ax.set_title(f"\\textsc{{{env}}}")
    ax.set_xlim(x_min, x_max)

plt.savefig("Results/FI_all.pdf", bbox_inches="tight")
