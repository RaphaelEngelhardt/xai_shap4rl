"""Check robustness of SHAP values as a function of used samples. The evaluation is done using a fixed number of samples from episodes
"""
import os
import argparse
import shap
import pickle
import joblib  # for saving explainer objects (see shap bug https://github.com/shap/shap/issues/2122#issuecomment-1688258990)
# Loading of saved explainers does not work anyway unfortunately. When loading it throws
#File "/usr/lib/python3.11/pickle.py", line 331, in _getattribute
#    raise AttributeError("Can't get attribute {!r} on {!r}"
#AttributeError: Can't get attribute 'pred' on <module '__main__' (built-in)>
import os
import numpy as np
from tqdm import tqdm
from utils import evaluate_model, init
from time import process_time

SEED = 987
BG_EPS = 10
EVAL_EPS = 10
N_EVAL = 1000
N_BG = [1, 5, 10, 20, 50, 100, 1000]


def pred(X):
    a, _ = model.predict(X, deterministic=True)
    return a

parser = argparse.ArgumentParser(description='Collect samples and SHAP values to test robustness of KernelSHAP')
parser.add_argument("-e", "--env", type=str, required=True,
                    help='The Farama Gymnasium environment ID')
parser.add_argument("-s", "--seed", type=int,
                    help='Seed used for experiment')
parser.add_argument("-m", "--mode", choices=['sampling', 'clustering'],
                    required=True,
                    help='Mode of sample reduction (sampling or clustering)')
args = parser.parse_args()
ENV_ID = args.env
DIR = f"./Results/{ENV_ID}/Robustness/"
SEED = args.seed
os.makedirs(DIR, exist_ok=True)

env, model, _, _, _ = init(ENV_ID)

# Get evaluation samples
print(f"Playing {EVAL_EPS} episodes to generate evaluation samples")
_, _, samples = evaluate_model(env, model, seed=SEED+1, eps=EVAL_EPS)
samples.to_feather(f"{DIR}{SEED}_{args.mode}_eval_samples_full.feather")
samples = samples.sample(N_EVAL, random_state=SEED)
eval_data = np.stack(samples.observations)
np.save(f"{DIR}{SEED}_{args.mode}_eval_samples_{N_EVAL}.npy", eval_data)

# Get background samples
print(f"Playing {BG_EPS} episodes to generate background samples")
_, _, samples = evaluate_model(env, model, seed=SEED, eps=BG_EPS)
samples.to_feather(f"{DIR}{SEED}_{args.mode}_bg_samples_full.feather")
bg_data = np.stack(samples.observations)

vals = {}
shap_times = {}

for i in tqdm(N_BG):

    if args.mode == "sampling":
        bg_data_subsamples = shap.utils.sample(bg_data, i, random_state=SEED)
    elif args.mode == "clustering":
        bg_data_subsamples = shap.kmeans(bg_data, i, round_values=False)  # random_state is hardcoded to 0, stochasticity is givne by the samples
    else:
        raise ValueError

    np.save(f"{DIR}{SEED}_{args.mode}_bg_samples_{i:04}.npy", bg_data_subsamples)


    explainer = shap.KernelExplainer(model=pred,
                                     data=bg_data_subsamples,
                                     seed=SEED)

    with open(f"{DIR}{SEED}_{args.mode}_explainer_{i:04}.sav", "wb") as f:
        joblib.dump(explainer, f)

    t0 = process_time()
    shap_values = explainer.shap_values(X=eval_data)
    t1 = process_time()
    shap_times[i] = t1 - t0
    print(f"Getting {eval_data.shape[0]} SHAP values using {i} background samples took {t1-t0} s")
    with open(f'{DIR}{SEED}_{args.mode}_shap_times.pkl', 'wb') as f:
        pickle.dump(shap_times, f)
    vals[i] = shap_values

    with open(f'{DIR}{SEED}_{args.mode}_shap_values.pkl', 'wb') as f:
        pickle.dump(vals, f)

    del shap_values
    del explainer
