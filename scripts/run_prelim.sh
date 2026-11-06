#!/bin/bash
# Preliminary experiment (実験計画.md): train one or more scenarios on this PC.
#   Parameters other than the scenario variables come from configs/envs/default.yaml and configs/algorithms/<algo>.yaml
#
# Usage:
#   ./scripts/run_prelim.sh list              # show scenario numbers
#   ./scripts/run_prelim.sh 5                 # run scenario 5
#   ./scripts/run_prelim.sh 4 7 10            # run several, one after another
#   ./scripts/run_prelim.sh all               # run all 15
#   ./scripts/run_prelim.sh --eval 5          # also run the indicator experiment (100 fixed seeds) after training
#
# Outputs: results/<theta>/<label>/<algo>/<run_id>/ (models, train.csv, eval_*.csv), results/logs/<scenario>_<time>.log
set -euo pipefail
cd "$(dirname "$0")/.."

# 1-3: PPO (MDP, overall optimum) per theta_dist / 4-15: PSRO (POSG) per theta_dist × strategy_space × aggregation
THETAS="0.25 0.50 0.75"            # 余裕 / 緊張 / 枯渇
STRATEGIES="withholder deceiver"
AGGREGATIONS="bid_to_win scale_to_win"

SCENARIOS=()
for theta in $THETAS; do
  SCENARIOS+=("ppo $theta - -")
done
for theta in $THETAS; do
  for strategy in $STRATEGIES; do
    for aggregation in $AGGREGATIONS; do
      SCENARIOS+=("psro $theta $strategy $aggregation")
    done
  done
done
N=${#SCENARIOS[@]}

list() {
  printf "%3s  %-5s %-6s %-11s %s\n" "No." "algo" "theta" "strategy" "aggregation"
  for i in $(seq 1 "$N"); do
    printf "%3d  %-5s %-6s %-11s %s\n" "$i" ${SCENARIOS[$((i - 1))]}
  done
}

run() {
  local no=$1 algo theta strategy aggregation
  read -r algo theta strategy aggregation <<< "${SCENARIOS[$((no - 1))]}"
  local name="${no}_${algo}_theta${theta}"
  local args=(--algo "$algo" --set "theta_dist_params=${theta};")
  if [ "$algo" = "psro" ]; then
    name="${name}_${strategy}_${aggregation}"
    args+=(--set "strategy_space=$strategy" --set "aggregation=$aggregation")
  fi
  local log="results/logs/${name}_$(date +%Y%m%d-%H%M%S).log"
  mkdir -p results/logs
  echo "=== [$no] $name  (log: $log)"
  # caffeinate -i: keep the Mac awake while training
  caffeinate -i uv run agrimachine-share-sim "${args[@]}" 2>&1 | tee "$log" || return 1

  if [ "$EVAL" = 1 ]; then
    local label="optimal"
    [ "$algo" = "psro" ] && label="${strategy}_${aggregation}"
    caffeinate -i uv run agrimachine-share-sim-evaluate "results/theta${theta}/${label}/${algo}" --skip_done 2>&1 | tee -a "$log" || return 1
  fi
}

EVAL=0
TARGETS=()
for arg in "$@"; do
  case "$arg" in
    list) list; exit 0 ;;
    --eval) EVAL=1 ;;
    all) for i in $(seq 1 "$N"); do TARGETS+=("$i"); done ;;
    *)
      if ! [[ "$arg" =~ ^[0-9]+$ ]] || [ "$arg" -lt 1 ] || [ "$arg" -gt "$N" ]; then
        echo "Unknown scenario: $arg (1-$N, all, list, --eval)" >&2
        exit 1
      fi
      TARGETS+=("$arg") ;;
  esac
done
if [ ${#TARGETS[@]} -eq 0 ]; then
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  echo
  list
  exit 1
fi

command -v uv > /dev/null || { echo "uv not found: install it first (https://docs.astral.sh/uv/)" >&2; exit 1; }
uv sync --frozen    # Builds .venv for this PC from uv.lock (do not copy .venv between PCs)

# A failed scenario does not stop the others
FAILED=()
for no in "${TARGETS[@]}"; do
  run "$no" || FAILED+=("$no")
done
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== Failed: ${FAILED[*]} (see results/logs/)" >&2
  exit 1
fi
echo "=== Done: ${TARGETS[*]}"
