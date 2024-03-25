import os
import argparse
import pickle
import numpy as np
import shap

from utils import evaluate_model, init

def pred(X):
    a, _ = model.predict(X, deterministic=True)
    return a


SEED = 987
BG_EPS = 10
EVAL_EPS = 10

parser = argparse.ArgumentParser(description='Collect samples and SHAP values to test robustness of KernelSHAP')
parser.add_argument("-e", "--env", type=str, required=True,
                    help='The Farama Gymnasium environment ID')
args = parser.parse_args()
ENV_ID = args.env
DIR = f"./Results/{ENV_ID}/FI/"

os.makedirs(DIR, exist_ok=True)

env, model, _, _, feature_names = init(ENV_ID)

m_bg, s_bg, samples = evaluate_model(env, model, seed=SEED, eps=BG_EPS, render=False)
samples.to_pickle(f"{DIR}bg_samples.pickle")
bg_data = np.stack(samples.observations)
bg_data = shap.utils.sample(bg_data, nsamples=1000, random_state=SEED)
np.save(f"{DIR}bg_samples_1000.npy", bg_data)

m_eval, s_eval, samples = evaluate_model(env, model, seed=SEED+1, eps=EVAL_EPS, render=False)
samples.to_pickle(f"{DIR}eval_samples.pickle")
eval_data = np.stack(samples.observations)
eval_data = shap.utils.sample(eval_data, nsamples=1000, random_state=SEED)
np.save(f"{DIR}eval_samples_1000.npy", eval_data)

explainer = shap.KernelExplainer(model=pred,
                                 data=bg_data,
                                 feature_names=feature_names,
                                 seed=SEED)

eval_shap = explainer.shap_values(X=eval_data)
with open(f'{DIR}eval_shap.pickle', 'wb') as f:
    pickle.dump(eval_shap, f)

bg_shap = explainer.shap_values(X=bg_data)
with open(f'{DIR}bg_shap.pickle', 'wb') as f:
    pickle.dump(bg_shap, f)

save = np.array([m_bg, s_bg, m_eval, s_eval])
np.save(f"{DIR}shap_perf.npy", save)
