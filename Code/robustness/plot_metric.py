import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import seaborn as sns
from matplotlib import rc
rc('text', usetex=True)
rc('font', size=12)
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


N_BG = [1, 5, 10, 20, 50, 100, 1000]
LST_SEEDS = [436924, 927436, 298392, 264023, 652473]
MODES = ["sampling", "clustering"]

colors = sns.color_palette("colorblind")

fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(10,12))

for num, env in enumerate(["LunarLanderContinuous-v2",
                           "Swimmer-v4",
                           "Hopper-v4",
                           "Walker2d-v4",
                           "HalfCheetah-v4",
                           "Ant-v4"]):

    ax = axs[num//2][num%2]
    DIR = f"./Results/{env}/Robustness/"

    for m_i, mode in enumerate(MODES):

        mse = np.zeros((len(LST_SEEDS), len(N_BG)-1))
        stds = []

        for i, seed in enumerate(LST_SEEDS):
            with open(f"{DIR}{seed}_{mode}_shap_values.pkl", "rb") as f:
                shap_values = pickle.load(f)

            # Get sigma of actions
            eval_samples = pd.read_feather(f"{DIR}{seed}_{mode}_eval_samples_full.feather")
            act_stds = np.std(np.stack(eval_samples.actions), axis=0)  # vector of len act_dim containing the std of each action dimension

            # Normalize shap values w.r.t. sigma of action dimension
            for n_bg in shap_values.keys():
                for act_dim in range(len(act_stds)):
                    shap_values[n_bg][act_dim] /= act_stds[act_dim]

            # Find maximum number of background samples.
            # This will serve as the reference.
            max_key = max(N_BG)
            ref = shap_values[max_key]

            ref_array = np.array(ref)
            all_stds_ref = np.std(ref_array, axis=1).flatten()
            stds.append(all_stds_ref)
            ref = np.stack(ref, axis=0).reshape(-1)

            # Compute RMSE for each set of eval values using the
            # ones computed with most bg samples as reference
            for j, bg in enumerate(N_BG[:-1]):
                sh = shap_values[bg]
                comp = np.stack(sh, axis=0).reshape(-1)
                mse[i, j] = np.sqrt(mean_squared_error(ref, comp))

        if mode == "clustering":
            all_stds_fl = np.array(stds).flatten()
            sort_idx = np.argsort(all_stds_fl)
            top_ref_idx = 5
            top_ref = all_stds_fl[sort_idx]
            mtr = np.mean(top_ref)

            err = ":" if mode=="sampling" else ":"
            ax.axhline(y=mtr, color="gray", linestyle=err)

        if mode == "sampling":
            fmt = "-."
            sub = "s"
        elif mode == "clustering":
            fmt = "--"
            sub = "c"
        else:
            raise ValueError

        x = N_BG[:-1]

        y_mse_mean = np.mean(mse, axis=0)
        y_mse_std = np.std(mse, axis=0)

        # Fit log-log values
        log_x = np.log(x)
        log_y_mse = np.log(y_mse_mean)
        coef_mse = np.polyfit(log_x, log_y_mse, 1)
        poly1d_func_mse = np.poly1d(coef_mse)
        k_mse = coef_mse[0]
        a_mse = np.exp(coef_mse[1])

        ax.plot(x, np.exp(poly1d_func_mse(log_x)), fmt, color=colors[m_i], alpha=0.6, lw=1)
        x_text = np.exp((log_x[-1]+log_x[0])/2)
        y_text = np.exp((log_y_mse[-1]+log_y_mse[0])/2)
        if mode=="clustering":
            if env == "Walker2d-v4":
                x_text = 2
                y_text = 0.02
            elif env == "HalfCheetah-v4":
                x_text = 1
                y_text = 0.015
            else:
                x_text = 1
                y_text *= 0.4
        else:
            x_text, y_text = (1.5, 0.5)
            if env in ["HalfCheetah-v4", "Ant-v4"]:
                x_text, y_text = (1.3, 0.25)
        ax.text(x_text, y_text,
                f"$\mathrm{{RMSE_{sub}}} \\approx {{{a_mse:.2}}}\cdot N_{{b}}^{{{k_mse:.2}}}$",
                color=colors[m_i], alpha=1)

        fmt2 = "X" if mode =="sampling" else "D"
        ax.errorbar(x, y_mse_mean, yerr=y_mse_std, fmt=fmt2, c=colors[m_i])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelbottom=True)
    ax.set_title(f"\\textsc{{{env}}}")
    if num//2 == 2:
        ax.set_xlabel(r"$N_b$")
    if num%2 == 0:
        ax.set_ylabel(f"RMSE (using $N_b = {{{max_key}}}$ as reference)")

    if env == "Walker2d-v4":
        labels = ["Sampling", "Clustering"]
        custom_lines =  [Line2D([0], [0], color=colors[0], ls="-.", marker="X"),
                         Line2D([0], [0], color=colors[1], ls="--", marker="D"),]
        ax.legend(handles=custom_lines, labels=labels, ncol=1,
                loc="lower left", bbox_to_anchor=(0, 0.005))

plt.subplots_adjust(wspace=0.12)
plt.savefig("Results/mse_all_error.pdf", bbox_inches="tight")
