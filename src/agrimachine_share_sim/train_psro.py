import os
import random
import time
import numpy as np
import pyspiel

# Must -- --- --- --- --- --- --- --- --- --- --- --- --- ---
from open_spiel.python import policy
from open_spiel.python import rl_agent
from open_spiel.python.algorithms.psro_v2 import best_response_oracle
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import rl_policy
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
from open_spiel.python.jax import dqn
from open_spiel.python.jax import policy_gradient
from open_spiel.python.rl_environment import Environment
# Optional -- --- --- --- --- --- --- --- --- --- --- --- ---
# To set the random seed for reproducibility
from open_spiel.python.rl_environment import ChanceEventSampler
# Exploitability of the meta-strategy mixture (full game tree: tiny games only)
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_aggregator
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# Register the game with OpenSpiel
from agrimachine_share_sim import posg # pylint: disable=unused-import
from agrimachine_share_sim import checkpoint, results
GAME_NAME = "python_agrimachine_share_posg"

# Follows open_spiel/python/examples/psro_v2_example.py

def _numpy_probs(agent_class):
    # RLPolicy.action_probabilities reads probs element by element ({a: p[a] for a in legal_actions}),
    # which is slow on a JAX array with many actions (~0.3 s for 15,625). Return probs as a numpy array.
    class Agent(agent_class):
        def step(self, time_step, is_evaluation=False):
            output = super().step(time_step, is_evaluation)
            if output is None:
                return output
            return rl_agent.StepOutput(action=output.action, probs=np.asarray(output.probs))
    return Agent

# rl_policy.PGPolicy / DQNPolicy with the agents above
PGPolicy = rl_policy.rl_policy_factory(_numpy_probs(policy_gradient.PolicyGradient))
DQNPolicy = rl_policy.rl_policy_factory(_numpy_probs(dqn.DQN))

def make_env(env_cfg, seed):
    # PSRO runs on sequential games: members report θ̃ one by one without seeing the others' reports
    game = pyspiel.load_game_as_turn_based(GAME_NAME, env_cfg)
    # Agents observe information_state_tensor, which is o_i^t (see posg.POSGObserver)
    return Environment(game, chance_event_sampler=ChanceEventSampler(seed=seed))

def init_pg_responder(env, cfg):
    """Initializes the Policy Gradient-based responder and agents."""
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    agent_class = PGPolicy
    agent_kwargs = {
        "info_state_size": info_state_size,
        "num_actions": num_actions,
        "loss_str": cfg["loss_str"],
        "loss_class": False,
        "hidden_layers_sizes": [cfg["hidden_layer_size"]] * cfg["n_hidden_layers"],
        "entropy_cost": cfg["entropy_cost"],
        "critic_learning_rate": cfg["critic_learning_rate"],
        "pi_learning_rate": cfg["pi_learning_rate"],
        "num_critic_before_pi": cfg["num_q_before_pi"],
        "optimizer_str": cfg["optimizer_str"],
    }
    oracle = rl_oracle.RLOracle(
        env,
        agent_class,
        agent_kwargs,
        number_training_episodes=cfg["number_training_episodes"],
        self_play_proportion=cfg["self_play_proportion"],
        sigma=cfg["sigma"])
    agents = [agent_class(env, player_id, **agent_kwargs) for player_id in range(env.num_players)]
    for agent in agents:
        agent.freeze()
    return oracle, agents

def init_br_responder(env, cfg):
    """Initializes the tabular best-response based responder and agents.

    Caution: max_steps <= 2 only. BR walks the game tree by information state, and o_i^t can repeat
    within an episode, so the walk never ends for max_steps >= 3 (crash).
    """
    del cfg
    random_policy = policy.TabularPolicy(env.game)
    oracle = best_response_oracle.BestResponseOracle(game=env.game, policy=random_policy)
    agents = [random_policy.__copy__() for _ in range(env.num_players)]
    return oracle, agents

def init_dqn_responder(env, cfg):
    """Initializes the DQN-based responder and agents."""
    state_representation_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    agent_class = DQNPolicy
    agent_kwargs = {
        "state_representation_size": state_representation_size,
        "num_actions": num_actions,
        "hidden_layers_sizes": [cfg["hidden_layer_size"]] * cfg["n_hidden_layers"],
        "batch_size": cfg["batch_size"],
        "learning_rate": cfg["dqn_learning_rate"],
        "update_target_network_every": cfg["update_target_network_every"],
        "learn_every": cfg["learn_every"],
        "optimizer_str": cfg["optimizer_str"],
    }
    oracle = rl_oracle.RLOracle(
        env,
        agent_class,
        agent_kwargs,
        number_training_episodes=cfg["number_training_episodes"],
        self_play_proportion=cfg["self_play_proportion"],
        sigma=cfg["sigma"])
    agents = [agent_class(env, player_id, **agent_kwargs) for player_id in range(env.num_players)]
    for agent in agents:
        agent.freeze()
    return oracle, agents

RESPONDERS = {"PG": init_pg_responder, "BR": init_br_responder, "DQN": init_dqn_responder}

def gpsro_looper(env, oracle, agents, cfg, logger=None):
    """Initializes and executes the GPSRO training loop."""
    sample_from_marginals = True  # As in the example (TODO there: set False for alpharank)
    training_strategy_selector = (cfg["training_strategy_selector"] or strategy_selectors.probabilistic)

    g_psro_solver = psro_v2.PSROSolver(
        env.game,
        oracle,
        initial_policies=agents,
        training_strategy_selector=training_strategy_selector,
        rectifier=cfg["rectifier"],
        sims_per_entry=cfg["sims_per_entry"],
        number_policies_selected=cfg["number_policies_selected"],
        meta_strategy_method=cfg["meta_strategy_method"],
        prd_iterations=50000,
        prd_gamma=1e-10,
        sample_from_marginals=sample_from_marginals,
        symmetric_game=cfg["symmetric_game"])

    start_time = time.time()
    for gpsro_iteration in range(cfg["gpsro_iterations"]):
        if cfg["verbose"]:
            print(f"Iteration : {gpsro_iteration}")
            print(f"Time so far: {time.time() - start_time}")
        g_psro_solver.iteration()
        meta_game = g_psro_solver.get_meta_game()
        meta_probabilities = g_psro_solver.get_meta_strategies()
        policies = g_psro_solver.get_policies()

        if cfg["verbose"]:
            print(f"Meta game : {meta_game}")
            print(f"Probabilities : {meta_probabilities}")
        if logger is not None:
            logger.log(iteration=gpsro_iteration, time=time.time() - start_time,
                       num_policies=[len(p) for p in policies], meta_strategies=meta_probabilities)

        # NashConv walks the full game tree with BR: tiny games with max_steps <= 2 only (see init_br_responder)
        if cfg["eval_nash_conv"]:
            aggregator = policy_aggregator.PolicyAggregator(env.game)
            aggr_policies = aggregator.aggregate(range(env.num_players), policies, meta_probabilities)
            exploitabilities, expl_per_player = exploitability.nash_conv(
                env.game, aggr_policies, return_only_nash_conv=False)
            print(f"Exploitabilities : {exploitabilities}")
            print(f"Exploitabilities per player : {expl_per_player}")
        # TODO: Add an evaluation metric that works on the default setting, where NashConv cannot be used

    return g_psro_solver

def main(env_config, algo_config):

    # From .yaml
    algo_cfg = algo_config.copy()
    env_cfg = env_config.copy()

    random.seed(algo_cfg["seed"])
    np.random.seed(algo_cfg["seed"])

    # Create environment, oracle and initial policies
    env = make_env(env_cfg, algo_cfg["seed"])
    oracle, agents = RESPONDERS[algo_cfg["oracle_type"]](env, algo_cfg)

    run_dir = results.make_run_dir(env_cfg, "psro", algo_cfg)
    logger = results.CSVLogger(os.path.join(run_dir, "train.csv"))

    # Run PSRO
    solver = gpsro_looper(env, oracle, agents, algo_cfg, logger)
    checkpoint.save_psro(solver, run_dir, algo_cfg["oracle_type"], env.game)
    print(f"Saved to {run_dir}")
    return solver

if __name__ == "__main__":
    main()
