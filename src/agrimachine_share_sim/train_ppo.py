import os
import random
import numpy as np
import pyspiel
import torch

# Must -- --- --- --- --- --- --- --- --- --- --- --- --- ---
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.rl_environment import Environment
from open_spiel.python.vector_env import SyncVectorEnv
# Optional -- --- --- --- --- --- --- --- --- --- --- --- ---
# MLP version of the agent (Atari CNN version is also available)
from open_spiel.python.pytorch.ppo import PPOAgent
# To set the random seed for reproducibility
from open_spiel.python.rl_environment import ChanceEventSampler
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

# Register the game with OpenSpiel
from agrimachine_share_sim import mdp # pylint: disable=unused-import
from agrimachine_share_sim import checkpoint, results
GAME_NAME = "python_agrimachine_share_mdp"

def make_env(game, seed):
    return Environment(game, chance_event_sampler=ChanceEventSampler(seed=seed))

def make_agent(game, cfg):
    return PPO(
        input_shape=tuple(game.observation_tensor_shape()),
        num_actions=game.num_distinct_actions(),
        num_players=game.num_players(),
        player_id=0,
        num_envs=cfg["num_envs"],
        steps_per_batch=cfg["num_steps"],
        num_minibatches=cfg["num_minibatches"],
        update_epochs=cfg["update_epochs"],
        learning_rate=cfg["learning_rate"],
        gae=cfg["gae"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        normalize_advantages=cfg["norm_adv"],
        clip_coef=cfg["clip_coef"],
        clip_vloss=cfg["clip_vloss"],
        entropy_coef=cfg["ent_coef"],
        value_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        target_kl=cfg["target_kl"],
        device="cpu",
        agent_fn=PPOAgent,
    )

def evaluate(env, agent, num_episodes):
    total = 0.0
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            # is_evaluation=True -> Not for training, just for evaluation
            output = agent.step([time_step], is_evaluation=True)[0]   
      
            # Deterministically choose the action with the highest probability
            # time_step = env.step(int(torch.argmax(output.probs)))  
            # Probabilistically sample an action from the policy distribution
            time_step = env.step([output.action]) 
      
            total += time_step.rewards[0]

    return total / num_episodes

def main(env_config, algo_config):
    
    # From .yaml
    algo_cfg = algo_config.copy()
    env_cfg = env_config.copy()
    
    random.seed(algo_cfg["seed"])
    np.random.seed(algo_cfg["seed"])
    torch.manual_seed(algo_cfg["seed"])

    # Create environment and agent
    game = pyspiel.load_game(GAME_NAME, env_cfg)
    envs = SyncVectorEnv(
        [make_env(game, algo_cfg["seed"] + i) for i in range(algo_cfg["num_envs"])])
    eval_env = make_env(game, algo_cfg["seed"] + 10_000)
    agent = make_agent(game, algo_cfg)
    run_dir = results.make_run_dir(env_cfg, "ppo", algo_cfg)
    logger = results.CSVLogger(os.path.join(run_dir, "train.csv"))

    # Train the agent
    batch_size = algo_cfg["num_envs"] * algo_cfg["num_steps"]
    num_updates = algo_cfg["total_timesteps"] // batch_size
    time_step = envs.reset()
    for update in range(num_updates):
        for _ in range(algo_cfg["num_steps"]):
            
            # Select actions and store obs, log-probs and values in the buffer
            agent_output = agent.step(time_step)
            
            # Advance all envs; finished episodes are reset automatically
            time_step, reward, done, _ = envs.step(agent_output, reset_if_done=True)
            
            # Store rewards and done flags for advantage estimation (GAE)
            agent.post_step(reward, done)

        # Linearly decay the learning rate
        if algo_cfg["anneal_lr"]:
            agent.anneal_learning_rate(update, num_updates)
        
        # PPO update
        agent.learn(time_step)

        # PPO update using the collected batch
        if update % algo_cfg["eval_every"] == 0 or update == num_updates - 1:
            avg_return = evaluate(eval_env, agent, algo_cfg["eval_episodes"])
            print(f"step={agent.total_steps_done:>8}  avg_return={avg_return:.4f}")
            logger.log(update=update, step=agent.total_steps_done, avg_return=avg_return)

    checkpoint.save_ppo(agent, run_dir)
    print(f"Saved to {run_dir}")
    return agent

if __name__ == "__main__":
    main()
