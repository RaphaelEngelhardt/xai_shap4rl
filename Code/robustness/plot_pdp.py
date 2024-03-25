import argparse
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt

from utils import init

parser = argparse.ArgumentParser(description='Collect samples and SHAP values to test robustness of KernelSHAP')
parser.add_argument("-e", "--env", type=str, required=True,
                    help='The Farama Gymnasium environment ID')
args = parser.parse_args()
ENV_ID = args.env
DIR = f"./Results/{ENV_ID}/Robustness/"

_, _, num_obs, num_act, feature_names = init(ENV_ID)

with open(f"{DIR}shap_values.pkl", "rb") as f:
    shap_values = pickle.load(f)

features = np.load(f"{DIR}eval_samples_1000.npy")

# Find min and max SHAP value
all_shap = np.concatenate(list(shap_values.values()), axis=0)
min_shap = np.min(all_shap)
max_shap = np.max(all_shap)

color = {key: plt.cm.tab10(i) for i, key in enumerate(shap_values.keys())}

fig, ax = plt.subplots(num_act, num_obs, figsize=(12.5*num_act, (15/8)*num_obs), squeeze=False)

for num_bg, values in shap_values.items():

    for j in range(num_act):  # loop over action dimensions
        for i in range(num_obs):  # loop over observation dimensions

            shap.dependence_plot(i, values[j], features,
                                        feature_names=feature_names, interaction_index=None, show=False, ax=ax[j, i], alpha=0.1, color=color[num_bg], dot_size=10)
            if num_bg == 1:  # histogram only once, not for every number of samples
                ax_histx = ax[j, i].inset_axes([0, 0, 1, 0.05], sharex=ax[j, i])
                ax_histx.hist(features[:,i], align='mid', alpha=0.3, color="gray")

                ax[j, i].vlines([np.mean(features[:,i])], min_shap, max_shap, linestyle="--", colors="gray", alpha=0.3)
            ax[j, i].hlines([np.mean(values[j][:,i])], np.min(features[:,i]), np.max(features[:,i]), linestyle="--", colors=color[num_bg], alpha=0.3)
            ax_histx.axis("off")

            if i == 0:
                ax[j, i].set_ylabel(f"SHAP of feature (actor {j})")
            else:
                ax[j, i].set_ylabel("")

            if j == num_act - 1:
                ax[j, i].set_xlabel(feature_names[i])
            else:
                ax[j, i].set_xlabel("")

            ax[j, i].set_ylim([min_shap, max_shap])

legend_handles = [plt.Line2D([0], [0], color=c, lw=4) for c in color.values()]
legend_labels = [f"{i}" for i in color.keys()]

fig.legend(legend_handles, legend_labels, title="number of background samples", ncols=len(color), loc="lower center")

plt.savefig(f"{DIR}pdp_all.png", dpi=300, bbox_inches="tight")
plt.close()
