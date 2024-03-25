import numpy as np
import pandas as pd
import gymnasium as gym
from tqdm import tqdm
from stable_baselines3 import TD3
from sb3_contrib import TQC

def init(env_id, **env_kwargs):  # gets env, model, num_obs, num_abs, feature names

    if env_id == "Swimmer-v4":
        env = gym.make(env_id, **env_kwargs)
        model = TD3.load("./models/swimmer_td3")
        feature_names = [r"$\theta_{\mathrm{tip}}$",
                         r"$\theta_{\mathrm{1 rot.}}$",
                         r"$\theta_{\mathrm{2 rot.}}$",
                         r"$v_{x_{\mathrm{tip}}}$",
                         r"$v_{y_{\mathrm{tip}}}$",
                         r"$\omega_{\mathrm{tip}}$",
                         r"$\omega_{\mathrm{1 rot.}}$",
                         r"$\omega_{\mathrm{2 rot.}}$"]

    elif env_id == "LunarLanderContinuous-v2":
        env = gym.make(env_id, continuous=True, **env_kwargs)
        model = TQC.load("./models/LLcont_TQC")
        feature_names = [r"$x$",
                         r"$y$",
                         r"$v_x$",
                         r"$v_y$",
                         r"$\theta$",
                         r"$\omega$",
                         r"$leg_1$",
                         r"$leg_2$"]

    elif env_id == "Hopper-v4":
        env = gym.make(env_id, **env_kwargs)
        model = TQC.load("./models/Hopper-v4_TQC")
        feature_names = [r"$z$",
                         r"$\theta_{\mathrm{torso}}$",
                         r"$\theta_{\mathrm{thigh}}$",
                         r"$\theta_{\mathrm{leg}}$",
                         r"$\theta_{\mathrm{foot}}$",
                         r"$v_{x_{\mathrm{torso}}}$",
                         r"$v_{z_{\mathrm{torso}}}$",
                         r"$\omega_{\mathrm{torso}}$",
                         r"$\omega_{\mathrm{thigh}}$",
                         r"$\omega_{\mathrm{leg}}$",
                         r"$\omega_{\mathrm{foot}}$"]

    elif env_id == "Walker2d-v4":
        env = gym.make(env_id, **env_kwargs)
        model = TD3.load("./models/Walker2d-v4_TD3")
        feature_names = [r"$z_{\mathrm{torso}}$",
                         r"$\theta_{\mathrm{torso}}$",
                         r"$\theta_{\mathrm{r. thigh}}$",
                         r"$\theta_{\mathrm{r. leg}}$",
                         r"$\theta_{\mathrm{r. foot}}$",
                         r"$\theta_{\mathrm{l. thigh}}$",
                         r"$\theta_{\mathrm{l. leg}}$",
                         r"$\theta_{\mathrm{l. foot}}$",
                         r"$v_{x_{\mathrm{torso}}}$",
                         r"$v_{z_{\mathrm{torso}}}$",
                         r"$\omega_{\mathrm{torso}}$",
                         r"$\omega_{\mathrm{r. thigh}}$",
                         r"$\omega_{\mathrm{r. leg}}$",
                         r"$\omega_{\mathrm{r. foot}}$",
                         r"$\omega_{\mathrm{l. thigh}}$",
                         r"$\omega_{\mathrm{l. leg}}$",
                         r"$\omega_{\mathrm{l. foot}}$"]

    elif env_id == "HalfCheetah-v4":
        env = gym.make(env_id, **env_kwargs)
        model = TQC.load("./models/HalfCheetah-v4_TQC")
        feature_names = [f"${{{i}}}$" for i in range(17)]  # too many variables for labelling
    elif env_id == "Ant-v4":
        env = gym.make(env_id, **env_kwargs)
        model = TD3.load("./models/Ant-v4_TD3")
        feature_names = [f"${{{i}}}$" for i in range(27)]  # too many variables for labelling
    else:
        raise ValueError("No model found for this environment.")

    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]

    return env, model, num_obs, num_act, feature_names

def evaluate_model(env, model, eps=100, seed=None, render=False):

    # Check seed argument sanity
    if (seed is None) or isinstance(seed, int):
        rng = np.random.default_rng(seed=seed)
        seeds = rng.choice(np.iinfo(np.int64).max , size=eps, replace=False)
    elif isinstance(seed, (list, np.ndarray)): # or isinstance(seed, np.ndarray):
        seeds = seed
    else:
        raise ValueError

    returns = np.zeros(eps)

    eps_o_list = []
    eps_a_list = []
    eps_seed = []
    eps_len = []
    eps_ret = []
    eps_idx = []
    frames = []

    for i in tqdm(range(eps)):
        observation_list = []
        action_list = []
        curr_eps_seed = int(seeds[i])
        s, _ = env.reset(seed=curr_eps_seed)
        terminated = truncated = False
        ret = 0

        while not (terminated or truncated):
            a, _ = model.predict(s, deterministic=True)
            observation_list.append(s)
            action_list.append(a)
            if render: frames.append(env.render())
            s, r, terminated, truncated, _ = env.step(a)

            ret += r
        returns[i] = ret

        n = len(action_list)
        eps_seed.append(np.full(shape=n, fill_value=curr_eps_seed))
        eps_len.append(np.full(shape=n, fill_value=n))
        eps_ret.append(np.full(shape=n, fill_value=ret))

        eps_o_list.extend(observation_list)
        eps_a_list.extend(action_list)
        eps_idx.append(np.full(shape=n, fill_value=i))

    eps_seed = np.concatenate(eps_seed)
    eps_len = np.concatenate(eps_len)
    eps_ret = np.concatenate(eps_ret)
    eps_idx = np.concatenate(eps_idx)

    if not render:
        frames = [None] * len(eps_seed)

    print(f"Average return in {eps} episodes: <R> = {np.mean(returns)} ± {np.std(returns)}")
    o_m = np.mean(returns)
    o_s = np.std(returns)
    samples = pd.DataFrame({"episode_index": eps_idx,
                            "seed": eps_seed,
                            "len": eps_len,
                            "return": eps_ret,
                            "observations": eps_o_list,
                            "actions": eps_a_list,
                            "frames": frames})
    return o_m, o_s, samples


class MaskObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env, mask, subst_values):

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        self.mask = mask
        self.subst_values = subst_values  # numpy array of values to substitute masked original ones with

    def observation(self, observation):

        new_obs = observation.copy()
        new_obs[self.mask] = self.subst_values[self.mask]

        return new_obs
