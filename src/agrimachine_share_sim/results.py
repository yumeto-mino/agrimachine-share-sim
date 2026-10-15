import csv
import datetime
import json
import os
import numpy as np
import pyspiel
import yaml
import agrimachine_share_sim.core as core

# Output layout (one directory per run):
#   results/<theta_dist>/<label>/<algo>/<run_id>/
#     env.yaml, algo.yaml   : configs used for the run
#     train.csv             : training progress (CSVLogger)
#     <name>_episodes.csv   : 1 row per episode   (Σw_i, Σv_i, Σcost_i, Π(w_i + v_i), per-member values)
#     <name>_members.csv    : 1 row per (episode, t, i)      (w_i, v_i, cost_i, penalty_i)
#     <name>_cells.csv      : 1 row per (episode, t, e, f)   (φ, θ, θ̃, assigned)
#   label = "optimal" for the MDP (PPO), "<strategy_space>_<aggregation>" for the POSG (PSRO)
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Run directory
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def scenario(env_cfg: dict, algo: str) -> dict:
    """Experiment variables and label ("optimal" for the MDP, which has no reports)"""
    theta = [x for x in str(env_cfg["theta_dist_params"]).replace(",", ";").split(";") if x.strip()]
    is_mdp = algo == "ppo"
    return {
        "algo": algo,
        "theta_dist": f"{float(theta[0]):.2f}" if theta else "",
        "strategy_space": "" if is_mdp else env_cfg["strategy_space"],
        "aggregation": "" if is_mdp else env_cfg["aggregation"],
        "label": "optimal" if is_mdp else f"{env_cfg['strategy_space']}_{env_cfg['aggregation']}",
    }

def make_run_dir(env_cfg: dict, algo: str, algo_cfg: dict, root: str = "results", run_id: str | None = None) -> str:
    """Create the run directory, save the configs, and return its path"""
    sc = scenario(env_cfg, algo)
    run_id = run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_seed{algo_cfg.get('seed', '')}"
    path = os.path.join(root, f"theta{sc['theta_dist']}", sc["label"], algo, run_id)
    os.makedirs(path, exist_ok=True)
    for name, cfg in (("env.yaml", env_cfg), ("algo.yaml", algo_cfg)):
        with open(os.path.join(path, name), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# CSV
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class CSVLogger:
    """Append-only CSV; columns come from the first row (list / dict / ndarray are saved as JSON)"""

    def __init__(self, path: str):
        self.path = path
        self._fields = None

    def log(self, **row):
        row = {k: _cell(v) for k, v in row.items()}
        if self._fields is None:
            self._fields = list(row)
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, self._fields).writeheader()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, self._fields, extrasaction="raise").writerow(row)

def write_csv(path: str, rows: list) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows({k: _cell(v) for k, v in row.items()} for row in rows)

def _cell(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, default=lambda x: np.asarray(x).tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Episodes: replay the history on the raw game (MDP or POSG) and decompose the rewards
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class EpisodeRecorder:
    """Split the rewards of finished episodes into w_i, v_i, report cost and truncation penalty.

    game: the raw mdp.MDP / posg.POSG. add() takes its states or turn-based ones (history() is replayed on game).
    MDP: θ̃ = NaN and report cost = 0.
    """

    def __init__(self, game, run_dir: str, name: str, info: dict | None = None):
        self.game = game
        self.model: core.Model = game.model
        self.is_posg = game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS
        self.run_dir, self.name = run_dir, name
        self.info = info or {}      # Columns added to every row, e.g. scenario(...)
        self.episodes, self.members, self.cells = [], [], []

    def add(self, state, seed: int | None = None) -> dict:
        episode = len(self.episodes)
        periods, truncated, penalty = self._replay(state.history())
        model, n = self.model, self.model.n
        head = {**self.info, "episode": episode, "seed": seed}

        totals = {key: np.zeros(n) for key in ("w", "v", "cost")}
        for t, (s, til_theta, a, w, v, cost) in enumerate(periods):
            for key, x in (("w", w), ("v", v), ("cost", cost)):
                totals[key] += x
            for i in range(n):
                self.members.append({**head, "t": t, "member": i,
                                     "w": w[i], "v": v[i], "cost": cost[i]})
            assigned = model.actions[a].any(axis=0)
            for e in range(model.m):
                for f in range(model.k):
                    self.cells.append({**head, "t": t, "phi": int(s.phi[e, f]),
                                       "member": int(np.flatnonzero(model.own[:, f])[0]), "e": e, "f": f,
                                       "theta": float(s.theta[e, f]), "til_theta": float(til_theta[e, f]),
                                       "assigned": int(assigned[e, f])})

        wv = totals["w"] + totals["v"]
        row = {**head, "length": len(periods), "truncated": int(truncated),
               "sum_wv": float(wv.sum()), "sum_w": float(totals["w"].sum()), "sum_v": float(totals["v"].sum()),
               "sum_cost": float(totals["cost"].sum()), "sum_penalty": float(penalty.sum()),
               "prod_wv": float(np.prod(wv))}
        for i in range(n):
            row |= {f"w_{i}": totals["w"][i], f"v_{i}": totals["v"][i], f"cost_{i}": totals["cost"][i],
                    f"penalty_{i}": penalty[i], f"return_{i}": wv[i] - totals["cost"][i] - penalty[i]}
        self.episodes.append(row)
        return row

    def _replay(self, history: list) -> tuple:
        """Replay history on game; return (s^t, θ̃^t, a^t, w^t, v^t, cost^t) per period"""
        model, n = self.model, self.model.n
        state = self.game.new_initial_state()
        periods = []
        history = list(history)
        while history:
            if state.is_chance_node():
                state.apply_action(history.pop(0))
                continue
            s = state._s  # pylint: disable=protected-access
            if self.is_posg:
                # Turn-based wrapper: θ̃_0, ..., θ̃_{n-1} one by one; then chance draws a from x(·|φ,θ̃)
                comms = [history.pop(0) for _ in range(n)]
                state.apply_actions(comms)
                a = history.pop(0)
                outcome = state._pending[a][0]  # pylint: disable=protected-access
                state.apply_action(a)
                a = outcome
                til_theta = model.observation(comms, a, s, 0)[0]
                cost = np.array([model.communication_cost(i, c) for i, c in enumerate(comms)])
            else:
                a = history.pop(0)
                state.apply_action(a)
                til_theta = np.full((model.m, model.k), np.nan)
                cost = np.zeros(n)
            action = model.actions[a]
            w = np.array([model._w(i, s.theta, action) for i in range(n)])  # pylint: disable=protected-access
            v = np.array([model._v(i, s.theta) for i in range(n)])          # pylint: disable=protected-access
            periods.append((s, til_theta, a, w, v, cost))
        truncated = state.is_terminal() and not model.is_terminal(state._s)  # pylint: disable=protected-access
        # Truncation penalty m·|F_i| per member (MDP: its total m·k is split the same way)
        penalty = model.m * model.own.sum(axis=1) * float(truncated)
        return periods, truncated, penalty

    def save(self) -> None:
        for kind, rows in (("episodes", self.episodes), ("members", self.members), ("cells", self.cells)):
            write_csv(os.path.join(self.run_dir, f"{self.name}_{kind}.csv"), rows)

    def summary(self) -> dict:
        """Means over episodes (estimates of expected values)"""
        keys = ("sum_wv", "sum_w", "sum_v", "sum_cost", "sum_penalty", "prod_wv", "length", "truncated")
        return {k: float(np.mean([row[k] for row in self.episodes])) for k in keys}
