import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
rc('text', usetex=True)
rc('font', size=14)
from matplotlib.lines import Line2D
import seaborn as sns

LST_SEEDS = [436924, 927436, 298392, 264023, 652473]

colors = sns.color_palette("colorblind")
c_shap = colors[3]
c_redu = colors[2]
fig, axs = plt.subplots(3, 2, sharex=True, figsize=(10,12))

for num, env in enumerate(["LunarLanderContinuous-v2",
                           "Swimmer-v4",
                           "Hopper-v4",
                           "Walker2d-v4",
                           "HalfCheetah-v4",
                           "Ant-v4"]):

    ax = axs[num//2][num%2]
    DIR = f"./Results/{env}/Robustness/"

    # Get computation time for SHAP computation
    shap_times = np.zeros((len(LST_SEEDS), 7))  # 7 is the number of different background data sizes we test
    for i, seed in enumerate(LST_SEEDS):
        with open(f"{DIR}{seed}_sampling_shap_times.pkl", "rb") as f:
            results = pickle.load(f)
            x = np.array(list(results.keys()))
            shap_times[i, :] = np.array(list(results.values()))

    # Get computation time for clustering
    overhead = pd.read_feather("overhead.feather")
    oh_clustering = overhead[(overhead["Environment"] == env) &
                            (overhead["Mode"] == "clustering")]
    oh_c_mean = oh_clustering.groupby(by="BG_size")["Time"].mean()
    oh_c_std = oh_clustering.groupby(by="BG_size")["Time"].std()

    # Plot SHAP computation time
    y = shap_times / 3600
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    ax.errorbar(x, y_mean, yerr=y_std, fmt="o-", color=c_shap)

    # Plot clustering computation time
    ax2 = ax.twinx()
    ax2.errorbar(x, oh_c_mean, yerr=oh_c_std, fmt="D--", color=c_redu)

    # Cosmetic changes to plots
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    if env in ["LunarLanderContinuous-v2", "Hopper-v4", "HalfCheetah-v4"]:
        ax.set_ylabel("process time KernelSHAP [h]", color=c_shap)
        ax2.set_ylabel("", color=c_redu)
    else:
        ax.set_ylabel("", color=c_shap)
        ax2.set_ylabel("process time clustering [s]", color=c_redu)

    if env in ["HalfCheetah-v4", "Ant-v4"]:
        ax.set_xlabel(r"$N_b$")

    ax.tick_params(axis='y', colors=c_shap, which='both')
    ax2.tick_params(axis='y', colors=c_redu, which='both')

    ax2.spines["right"].set_color(c_redu)
    ax.spines["left"].set_color(c_shap)
    ax.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    for tick_label in ax.get_yticklabels():
        tick_label.set_color(c_shap)
    for tick in ax.get_yticklines():
        tick.set_color(c_shap)
    for tick_label in ax2.get_yticklabels():
        tick_label.set_color(c_redu)
    for tick in ax2.get_yticklines():
        tick.set_color(c_redu)

    if env == "Walker2d-v4":
        labels = ["SHAP computation", "Clustering"]
        custom_lines = [Line2D([0], [0], color=c_shap, marker="o"),
                        Line2D([0], [0], color=c_redu, ls="--", marker="D")]
        ax.legend(handles=custom_lines, labels=labels,
                loc="lower left", bbox_to_anchor=(0.03, 0.75))

    ax.xaxis.set_tick_params(labelbottom=True)
    ax.set_title(f"\\textsc{{{env}}}")

plt.tight_layout()
plt.savefig("Results/compute time.pdf", bbox_inches="tight")
