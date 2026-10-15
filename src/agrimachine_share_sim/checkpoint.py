import json
import os
import pickle
import numpy as np
import torch
import flax.nnx as nn
import jax
from open_spiel.python import policy as policy_lib
from open_spiel.python.jax import dqn
from open_spiel.python.jax import policy_gradient

# Saved files (under <run_dir>/model/):
#   info.json  : {"algo": ...}
#   ppo.pt     : state_dict of PPO.network (rebuild the agent with train_ppo.make_agent, then load it)
#   psro.pkl   : {"oracle_type", "weights": [[w per policy] per player], "meta_strategies", "joint_meta_strategy", "meta_game"}
#                weights are numpy pytrees (DQN: Q-network params / PG: haiku params / BR: tabular action probabilities)
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Save
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def model_dir(run_dir: str) -> str:
    path = os.path.join(run_dir, "model")
    os.makedirs(path, exist_ok=True)
    return path

def save_info(run_dir: str, algo: str) -> None:
    with open(os.path.join(model_dir(run_dir), "info.json"), "w", encoding="utf-8") as f:
        json.dump({"algo": algo}, f)

def save_ppo(agent, run_dir: str) -> str:
    path = os.path.join(model_dir(run_dir), "ppo.pt")
    torch.save(agent.network.state_dict(), path)
    save_info(run_dir, "ppo")
    return path

def save_psro(solver, run_dir: str, oracle_type: str, game=None) -> str:
    """Save PSRO's policies and meta-strategies; BR policies need game (turn-based) to be tabulated"""
    weights = [[get_weights(p, game) for p in policies] for policies in solver.get_policies()]
    joint = solver.get_and_update_non_marginalized_meta_strategies(update=False)
    data = {
        "oracle_type": oracle_type,
        "weights": weights,
        "meta_strategies": [np.asarray(p) for p in solver.get_meta_strategies()],
        "joint_meta_strategy": None if joint is None or np.size(joint) == 0 else np.asarray(joint),
        "meta_game": [np.asarray(m) for m in solver.get_meta_game()],
    }
    path = os.path.join(model_dir(run_dir), "psro.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    save_info(run_dir, "psro")
    return path
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Weights of one PSRO policy (RLPolicy wrapping DQN / PolicyGradient, or a tabular-convertible policy)
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def get_weights(pol, game=None):
    agent = getattr(pol, "_policy", None)
    if isinstance(agent, dqn.DQN):
        return _numpy(nn.state(agent._q_network, nn.Param).to_pure_dict())  # pylint: disable=protected-access
    if isinstance(agent, policy_gradient.PolicyGradient):
        return _numpy(agent.hk_net_params)
    if game is None:
        raise ValueError(f"Cannot save {type(pol).__name__} without game (needed to tabulate it)")
    return policy_lib.tabular_policy_from_callable(game, pol).action_probability_array.copy()

def set_weights(pol, weights) -> None:
    agent = getattr(pol, "_policy", None)
    if isinstance(agent, dqn.DQN):
        state = nn.state(agent._q_network, nn.Param)  # pylint: disable=protected-access
        nn.replace_by_pure_dict(state, weights)
        nn.update(agent._q_network, state)  # pylint: disable=protected-access
    elif isinstance(agent, policy_gradient.PolicyGradient):
        agent.hk_net_params = jax.tree.map(jax.numpy.asarray, weights)
    elif isinstance(pol, policy_lib.TabularPolicy):
        pol.action_probability_array[:] = weights
    else:
        raise ValueError(f"Cannot restore weights into {type(pol).__name__}")

def _numpy(tree):
    return jax.tree.map(np.asarray, tree)
