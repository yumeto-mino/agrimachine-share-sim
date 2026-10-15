import argparse
import os
from agrimachine_share_sim import evaluate

# uv run agrimachine-share-sim-evaluate [PATH ...] [--episodes 100] [--seed0 0] [--meta_sampling joint]
#   PATH: a run directory (with model/info.json), or a directory containing runs (all are evaluated)

def find_runs(path: str) -> list:
    runs = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        if os.path.isfile(os.path.join(root, "model", "info.json")):
            runs.append(root)
            dirs.clear()
    return runs

def main() -> None:
    parser = argparse.ArgumentParser(prog="agrimachine-share-sim-evaluate")
    parser.add_argument(
        "paths", nargs="*",
        default=["results"],
        help="Run directories, or directories containing them",
    )
    parser.add_argument(
        "--episodes", type=int,
        default=100,
        help="Number of episodes (one fixed seed each)",
    )
    parser.add_argument(
        "--seed0", type=int,
        default=0,
        help="Seeds are seed0, ..., seed0 + episodes - 1",
    )
    parser.add_argument(
        "--meta_sampling", type=str,
        default="joint",
        choices=("joint", "marginal"),
        help="PSRO: sample the policy profile from the joint meta-strategy or from each member's marginal",
    )
    parser.add_argument(
        "--skip_done", action="store_true",
        help="Skip runs that already have eval_episodes.csv",
    )
    args = parser.parse_args()

    runs = [run for path in args.paths for run in find_runs(path)]
    if not runs:
        parser.error(f"No trained model found under {args.paths}")
    for run in runs:
        if args.skip_done and os.path.isfile(os.path.join(run, "eval_episodes.csv")):
            print(f"Skip {run}")
            continue
        evaluate.main(run, args.episodes, args.seed0, args.meta_sampling)

if __name__ == "__main__":
    main()
