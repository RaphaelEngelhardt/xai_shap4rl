import argparse
import pickle
import numpy as np
from utils import evaluate_model, init, MaskObservation

SEED = 1000

parser = argparse.ArgumentParser(description='Collect samples and SHAP values to test robustness of KernelSHAP')
parser.add_argument("-e", "--env", type=str, required=True,
                    help='The Farama Gymnasium environment ID')
args = parser.parse_args()
ENV_ID = args.env
DIR = f"./Results/{ENV_ID}/FI/"

env, model, _, num_act, _ = init(ENV_ID)

# Compute feature importance
with open(f'{DIR}eval_shap.pickle', 'rb') as f:
    eval_shap = pickle.load(f)

#new: normalize w.r.t. average action
_, _, samples = evaluate_model(env, model, seed=988, eps=10, render=False)
act = np.stack(samples.actions)
act = np.std(act, axis=0)

fi = []
for act_dim in range(num_act):
    mean_abs_shap = np.mean(np.abs(eval_shap[act_dim]/act[act_dim]), axis=0)
    fi.append(mean_abs_shap)
fi = np.array(fi)
fi = np.mean(fi, axis=0)
np.save(f"{DIR}fi.npy", fi)

# Get values to fill observation vector for masked observations
with open(f"{DIR}eval_samples.pickle", 'rb') as f:
    eval_samples = pickle.load(f)

observations = np.stack(eval_samples.observations)
fill_values = np.mean(observations, axis=0)

# Perform evaluations
mask_matrix = np.identity(len(fill_values), dtype=bool)

m_ref, s_ref, _ = evaluate_model(env, model, eps=100, seed=SEED)
save = np.array([m_ref, s_ref])
np.save(f"{DIR}ref_performance.npy", save)

means = np.zeros(len(fill_values))
stds = np.zeros(len(fill_values))
samples = []
for i in range(len(fill_values)):
    env, _, _, _, _ = init(ENV_ID)
    env = MaskObservation(env, mask_matrix[i], fill_values)
    means[i], stds[i], s = evaluate_model(env, model, eps=100, seed=SEED)
    samples.append(s)
    del env

np.save(f"{DIR}means.npy", means)
np.save(f"{DIR}stds.npy", stds)
with open(f"{DIR}samples.pickle", 'wb') as f:
    pickle.dump(samples, f)
