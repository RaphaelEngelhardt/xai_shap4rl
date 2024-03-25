import argparse
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from utils import init

parser = argparse.ArgumentParser(description='Collect samples and SHAP values to test robustness of KernelSHAP')
parser.add_argument("-e", "--env", type=str, required=True,
                    help='The Farama Gymnasium environment ID')
args = parser.parse_args()
ENV_ID = args.env
DIR = f"./Results/{ENV_ID}/FI/"
_, _, _, act_dim, feature_names = init(ENV_ID)

with open(f"{DIR}samples.pickle", 'rb') as f:
    samples = pickle.load(f)

data_violin = []
for i in range(len(samples)):
    returns = samples[i].groupby("episode_index")["return"].mean().to_numpy()
    data_violin.append(returns)

means = np.load(f"{DIR}means.npy")
stds = np.load(f"{DIR}stds.npy")
ref = np.load(f"{DIR}ref_performance.npy")
ref_mean = ref[0]
ref_std = ref[1]
fi = np.load(f"{DIR}fi.npy")
print(f"{args.env}: Performance fully observable: $\overline{{R}} = {ref_mean} \pm {ref_std}$")

c = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

x_min = -0.07*(max(fi)-min(fi))
x_max = max(fi)+0.07*(max(fi)-min(fi))

coef = np.polyfit(fi, means, 1)
poly1d_func = np.poly1d(coef)
x_lin = np.linspace(x_min, x_max, 100)
y_lin = poly1d_func(x_lin)

fig, ax = plt.subplots(1, 1, figsize=(10/2,12/3))
ax.plot(x_lin, y_lin, "--", color=c, alpha=0.3)

r = pearsonr(fi, means)
print(f"Correlation r={r[0]}")

# Coefficient of determination
r2 = r2_score(means, poly1d_func(fi))
print(f"Coefficient of determination R²={r2}")

ax.errorbar(fi, means, yerr=stds, fmt="o")

violin_parts = plt.violinplot(data_violin, positions=fi, showextrema=False, widths=0.05)
for pc in violin_parts['bodies']:
    pc.set_facecolor(c)
    pc.set_edgecolor(c)
x_violin = fi[0] + np.random.uniform(low=-0.019, high=0.019, size=len(data_violin[0]))
ax.scatter(x = x_violin, y=data_violin[0], s=0.5, c=c)

for i in range(len(fi)):
    xytext = (0.3, 0.3)
    # Beautify for special cases
    if args.env == "Swimmer-v4" and i == 2:
        xytext = (-2.4, 0.6)
    elif args.env == "Swimmer-v4" and i == 5:
        xytext = (0.3, -1)
    elif args.env == "Swimmer-v4" and i == 6:
        xytext = (-0.3, 0.7)
    ax.annotate(feature_names[i], xy=(fi[i], means[i]),
                xytext=xytext, textcoords="offset fontsize")

ax.axhline(y=ref_mean, color="g", linestyle="-.")
ax.fill_between(np.linspace(x_min, x_max),
                ref_mean-ref_std,
                ref_mean+ref_std,
                color="g", alpha=0.1)

with open('random_baseline.pkl', 'rb') as f:
    random_baseline = pickle.load(f)
ax.axhline(y=random_baseline[ENV_ID + "_mean"], color="r", linestyle="--")
ax.fill_between(np.linspace(x_min, x_max),
                random_baseline[ENV_ID + "_mean"]-random_baseline[ENV_ID + "_std"], random_baseline[ENV_ID + "_mean"]+random_baseline[ENV_ID + "_std"],
                color="r", alpha=0.1)

ax.annotate(f"$r={{{r[0]:.3f}}}$\n$R^2={{{r2:.3f}}}$", xy=(0.3, 200),
            xytext=(1,1), textcoords="offset fontsize", color=c, alpha=0.7)

ax.set_xlabel(r"$\mathrm{FI}_j$")
ax.set_ylabel(r"$\overline{R}_{\setminus j}$")

ax.set_title(f"\\textsc{{{args.env}}}")
ax.set_xlim(x_min, x_max)

plt.savefig(f"{DIR}FI_{args.env}_violin.pdf", bbox_inches="tight")
