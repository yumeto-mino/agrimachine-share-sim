import json
import os
import pickle
import random
import numpy as np
import pyspiel
import torch
import yaml
from open_spiel.python.algorithms.psro_v2 import utils as psro_utils
from open_spiel.python import policy as policy_lib
from agrimachine_share_sim import checkpoint, results
from agrimachine_share_sim import train_ppo, train_psro

# Load <run_dir>/model/, simulate one episode per seed (seed0, ..., seed0 + num_episodes - 1),
# and write <run_dir>/eval_{episodes,members,cells}.csv. Each seed drives both chance events and policy sampling.
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Load
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def load_run(run_dir: str) -> tuple:
    """(algo, env_cfg, algo_cfg)"""
    with open(os.path.join(run_dir, "model", "info.json"), encoding="utf-8") as f:
        algo = json.load(f)["algo"]
    cfgs = []
    for name in ("env.yaml", "algo.yaml"):
        with open(os.path.join(run_dir, name), encoding="utf-8") as f:
            cfgs.append(yaml.safe_load(f))
    return algo, *cfgs

def load_ppo(run_dir: str, game, algo_cfg: dict):
    agent = train_ppo.make_agent(game, algo_cfg)
    agent.network.load_state_dict(torch.load(os.path.join(run_dir, "model", "ppo.pt")))
    agent.eval()
    return agent

def load_psro(run_dir: str, env, algo_cfg: dict) -> tuple:
    """(policies[player][k], data); policies are rebuilt with the training config, then get their weights"""
    with open(os.path.join(run_dir, "model", "psro.pkl"), "rb") as f:
        data = pickle.load(f)
    oracle, _ = train_psro.RESPONDERS[data["oracle_type"]](env, algo_cfg)
    policies = []
    for player, weights in enumerate(data["weights"]):
        policies.append([])
        for w in weights:
            if data["oracle_type"] == "BR":
                pol = policy_lib.TabularPolicy(env.game)
            else:
                pol = oracle._best_response_class(env, player, **oracle._best_response_kwargs)  # pylint: disable=protected-access
                pol.freeze()
            checkpoint.set_weights(pol, w)
            policies[-1].append(pol)
    return policies, data
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Rollouts
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def rollout_ppo(env, agent, seed: int):
    torch.manual_seed(seed)
    time_step = env.reset()
    while not time_step.last():
        output = agent.step([time_step], is_evaluation=True)[0]
        time_step = env.step([output.action])
    return env.get_state

def sample_profile(policies: list, data: dict, rng: np.random.Generator, meta_sampling: str) -> list:
    """One policy per member. joint: from the joint meta-strategy (alpharank); marginal: independently per member"""
    joint = data["joint_meta_strategy"]
    if meta_sampling == "joint" and joint is not None:
        joint = np.asarray(joint, dtype=float).ravel()
        profile_id = rng.choice(len(joint), p=joint / joint.sum())
        return psro_utils.get_joint_policies_from_id_list(data["meta_game"], policies, [profile_id])[0]
    profile = []
    for pols, probs in zip(policies, data["meta_strategies"]):
        probs = np.asarray(probs, dtype=float)
        profile.append(pols[rng.choice(len(pols), p=probs / probs.sum())])
    return profile

def rollout_psro(env, profile: list, rng: np.random.Generator):
    time_step = env.reset()
    while not time_step.last():
        player = time_step.observations["current_player"]
        probs = profile[player].action_probabilities(env.get_state, player)
        actions = list(probs)
        p = np.asarray([probs[a] for a in actions], dtype=float)
        time_step = env.step([int(actions[rng.choice(len(actions), p=p / p.sum())])])
    return env.get_state
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Main
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def main(run_dir: str, num_episodes: int = 100, seed0: int = 0, meta_sampling: str = "joint") -> dict:
    algo, env_cfg, algo_cfg = load_run(run_dir)
    random.seed(seed0)
    np.random.seed(seed0)

    if algo == "ppo":
        game = pyspiel.load_game(train_ppo.GAME_NAME, env_cfg)
        agent = load_ppo(run_dir, game, algo_cfg)
    elif algo == "psro":
        game = pyspiel.load_game(train_psro.GAME_NAME, env_cfg)   # Raw POSG for the recorder
        env = train_psro.make_env(env_cfg, seed0)
        policies, data = load_psro(run_dir, env, algo_cfg)
    else:
        raise ValueError(f"Unknown algo {algo} in {run_dir}")

    recorder = results.EpisodeRecorder(game, run_dir, "eval", results.scenario(env_cfg, algo))
    for seed in range(seed0, seed0 + num_episodes):
        if algo == "ppo":
            state = rollout_ppo(train_ppo.make_env(game, seed), agent, seed)
        else:
            rng = np.random.default_rng(seed)
            profile = sample_profile(policies, data, rng, meta_sampling)
            state = rollout_psro(train_psro.make_env(env_cfg, seed), profile, rng)
        recorder.add(state, seed=seed)
    recorder.save()

    summary = recorder.summary()
    print(f"{run_dir}: {summary}")
    return summary
