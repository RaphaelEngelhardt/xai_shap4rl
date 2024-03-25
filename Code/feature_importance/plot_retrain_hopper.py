import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from utils import init

fig, axs = plt.subplots(1, 1, figsize=(10/2,12/3), squeeze=False)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
blue = colors[0]
coord = {"Hopper-v4": (0.36, 700)}
coord_re = {"Hopper-v4": (0.05, 2500)}

for num, env in enumerate(["Hopper-v4"]):

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

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.plot(x_lin, y_lin, "--", alpha=0.3, color=blue)

    r = pearsonr(fi, means)
    print(f"Correlation r={r[0]}")

    # Coefficient of determination
    r2 = r2_score(means, poly1d_func(fi))
    print(f"Coefficient of determination R²={r2}")

    ax.errorbar(fi, means, yerr=stds, fmt="o", c=blue, label="partially blinded agent")
    for i in range(len(fi)):
        xytext = (0.3, 0.3)
        # Beautify for special cases
        if i == 6 and env == "Hopper-v4":
            xytext=(-3, 0.3)
        elif i == 7 and env == "Hopper-v4":
            xytext=(-3, -0.3)
        elif i == 2 and env == "Hopper-v4":
            xytext=(-1, 0.5)
        elif i == 10 and env == "Hopper-v4":
            xytext=(-0.8, -0.8)
        elif i == 5 and env == "Hopper-v4":
            xytext=(0.15, 0.45)
        elif i == 0 and env == "Hopper-v4":
            xytext = (-0.8, -0.8)
        ax.annotate(feature_names[i], xy=(fi[i], means[i]), xytext=xytext, textcoords="offset fontsize")

    ax.axhline(y=ref_mean, color="g", linestyle="-.", label="unblinded agent")
    ax.fill_between(np.linspace(x_min, x_max), ref_mean-ref_std, ref_mean+ref_std, color="g", alpha=0.1)

    with open('random_baseline.pkl', 'rb') as f:
        random_baseline = pickle.load(f)
    ax.axhline(y=random_baseline[env + "_mean"], color="r", linestyle="--", label="random agent")
    ax.fill_between(np.linspace(x_min, x_max),
                    random_baseline[env + "_mean"]-random_baseline[env + "_std"],
                    random_baseline[env + "_mean"]+random_baseline[env + "_std"],
                    color="r", alpha=0.1)


    ax.annotate(f"$r={{{r[0]:.3f}}}$\n$R^2={{{r2:.3f}}}$", xy=coord[env], xytext=(0,0), textcoords="offset fontsize", alpha=0.7, color=blue)

    ax.set_xlabel(r"$\mathrm{FI}_j$")

    if num%2 == 0:
        ybox1 = TextArea(r"$\overline{R}_{\setminus j}^{\mathrm{(retrain)}}$ ", textprops=dict(color=colors[1], size=10, rotation=90, ha='left', va='bottom'))
        ybox2 = TextArea("and ", textprops=dict(color="k", size=10, rotation=90, ha='left', va='bottom'))
        ybox3 = TextArea(r"$\overline{R}_{\setminus j}$", textprops=dict(color=blue, size=10, rotation=90, ha='left', va='bottom'))

        ybox = VPacker(children=[ybox1, ybox2, ybox3], align="bottom", pad=0, sep=5)

        anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.13, 0.3), bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)

    # Plot retraining
    m_retrain = np.load(f"train missing features/masked_m.npy")
    s_retrain = np.load(f"train missing features/masked_s.npy")
    ax.errorbar(fi, m_retrain, yerr=s_retrain, fmt="^", color=colors[1])
    for i in range(len(fi)):
        xytext = (0.3, 0.3)
        if i == 6 and env == "Hopper-v4":
            xytext=(-3, 0.3)
        ax.annotate(feature_names[i], xy=(fi[i], m_retrain[i]), xytext=xytext, textcoords="offset fontsize")

    coef_retrain = np.polyfit(fi, m_retrain, 1)
    poly1d_func_retrain = np.poly1d(coef_retrain)

    r_re = pearsonr(fi, m_retrain)
    print(f"Correlation r={r_re[0]}")
    # Coefficient of determination
    r2_re = r2_score(m_retrain, poly1d_func_retrain(fi))
    print(f"Coefficient of determination R²={r2_re}")

    y_lin_r = poly1d_func_retrain(x_lin)
    ax.plot(x_lin, y_lin_r, "--", alpha=0.3, c=colors[1])
    ax.annotate(f"$r={{{r_re[0]:.3f}}}$\n$R^2={{{r2_re:.3f}}}$", xy=coord_re[env], xytext=(0,0), textcoords="offset fontsize", alpha=0.7, color=colors[1])

    ax.set_title(f"\\textsc{{{env}}}")
    ax.set_xlim(x_min, x_max)

plt.savefig("Results/FI_retrain_hopper.pdf", bbox_inches="tight")
