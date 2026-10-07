import argparse
from importlib import import_module

ALGORITHMS = {
    "single": {
        # Default
        "ppo": "agrimachine_share_sim.train_ppo",
        # Option
        # "dp": "agrimachine_share_sim.exact_dp",
    },
    # "multi": {
    #     # Default
    #     "psro": "agrimachine_share_sim.train_psro",
    #     # Option
    # },
}

def main() -> None:
    
    parser = argparse.ArgumentParser(prog="agrimachine-share-sim")
    parser.add_argument("mode", nargs="?", choices=ALGORITHMS, help="OPTIONAL")
    parser.add_argument("algo", nargs="?", help="OPTIONAL")
    args = parser.parse_args()

    if args.mode is None:
        # TODO: Define "uv run agrimachine-share-sim"
        return

    algos = ALGORITHMS[args.mode]
    algo = args.algo or next(iter(algos))
    if algo not in algos:
        parser.error(f"ALGORITHMS that can be used with mode '{args.mode}': {', '.join(algos)}")

    # TODO: Uncomment
    # import_module(algos[algo]).main()
