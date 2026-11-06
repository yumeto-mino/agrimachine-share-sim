import argparse
import yaml
import os
from importlib import import_module

ALGORITHMS = {
    "ppo":"agrimachine_share_sim.train_ppo",
    "psro":"agrimachine_share_sim.train_psro",
    # "dp": "agrimachine_share_sim.calculate_dp",
}

def load_yaml(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main() -> None:
    
    parser = argparse.ArgumentParser(prog="agrimachine-share-sim")
    parser.add_argument(
        "--env", type=str, 
        default="default", 
        help="File path to the environment configuration",
    )
    parser.add_argument(
        "--algo", type=str, 
        default="ppo", 
        choices=ALGORITHMS.keys(),
        help="File path to the algorithm configuration",
    )
    parser.add_argument(
        "--algo_config", type=str, default=None,
    )    
    parser.add_argument(
        "--set", type=str, action="append", default=[], metavar="KEY=VALUE",
        help="Override an environment parameter (repeatable), e.g. --set theta_dist_params='0.25;'",
    )
    args = parser.parse_args()
      
    env_path = os.path.join("configs", "envs", f"{args.env}.yaml")
    algo_file = args.algo_config if args.algo_config else f"{args.algo}.yaml"
    algo_path = os.path.join("configs", "algorithms", algo_file)
    env_config = load_yaml(env_path)
    algo_config = load_yaml(algo_path)
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or key not in env_config:
            parser.error(f"--set needs KEY=VALUE with KEY in {env_path}; got {item}")
        env_config[key] = yaml.safe_load(value)

    import_module(ALGORITHMS[args.algo]).main(env_config, algo_config)
