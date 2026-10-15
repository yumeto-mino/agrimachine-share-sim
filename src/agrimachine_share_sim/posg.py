import functools
import numpy as np
import pyspiel
import agrimachine_share_sim.core as core
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Parameters
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
_DEFAULT_PARAMS = {
  "num_member": 3,
  "num_machine": 2,
  "num_field": 6,
  "field_owner": "0,0,0,1,1,2",
  "theta_space": "0.0,0.25,0.5,0.75,1.0",
  "phi_init_params": "0.0,1.0",
  "phi_dist_params": "0.7,0.3,0.4,0.6",
  "theta_init_params": "0.25",
  "theta_dist_params": "0.5",
  "mu": 0.05,
  "c": 0.05,
  "aggregation": "bid_to_win",
  "strategy_space": "deceiver",
  # Guarantees that an episode will always end.
  "max_steps": 100,
}
_MAX_NUM_PLAYERS = 10
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# GameType: Declare the "nature" of the game
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
_GAME_TYPE = pyspiel.GameType(
    short_name="python_agrimachine_share_posg",
    long_name="Python Agrimachine Share POSG",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_MAX_NUM_PLAYERS,
    min_num_players=1,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification=_DEFAULT_PARAMS,
)

# For .load_game
@functools.cache
def _load_model(params_items):
  return core.Model(dict(params_items))
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Game: Holds parameters, creates initial state and Observer
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class POSG(pyspiel.Game):
  def __init__(self, params=None):
    params = {**_DEFAULT_PARAMS, **(params or {})}
    self.model = _load_model(tuple(sorted(params.items())))
    self.max_steps = params["max_steps"]
    # Penalty for truncation: member i loses the number of its tasks m·|F_i|
    self.penalty = self.model.m * self.model.own.sum(axis=1)

    super().__init__(
        _GAME_TYPE,
        pyspiel.GameInfo(
            num_distinct_actions= self.model.num_communications,
            # Chance draws either s from T(s,a,·) / Pr(s^0), or a from x(·|φ,θ̃)
            max_chance_outcomes= max(self.model.max_chance_outcomes, self.model.num_actions),
            num_players=self.model.n,
            min_utility= - self.max_steps * (self.model.max_cost + self.model.c * self.model.m * self.model.k)
                         - float(self.penalty.max()),
            max_utility= self.max_steps * self.model.max_reward,
            utility_sum=None,
            # Initial draw + (report, aggregation, transition) per period
            max_game_length= 3 * self.max_steps + 1,
        ),
        params,
    )

  def new_initial_state(self):
    return POSGState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    perfect_recall = iig_obs_type is not None and iig_obs_type.perfect_recall
    return POSGObserver(self, perfect_recall, params)
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# State: Holds the current state, time step, and whether it's a chance turn. Implements the OpenSpiel API.
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class POSGState(pyspiel.State):
  def __init__(self, game):
    super().__init__(game)
    self._model = game.model
    self._s = None                                        # core.State; None = before drawing from Pr(s^0)
    self._t = 0                                           # Elapsed periods
    self._history = []                                    # ĥ^t = [s^0, θ̃^0, a^0, s^1, ..., θ̃^{t-1}, a^{t-1}, s^t]
    self._pending = self._model.initial_distribution()    # Chance outcomes [(State or action, p), ...]; None = members' turn
    self._rewards = np.zeros(self._model.n)               # Rewards of the last period (do NOT reset them at chance turns)
    self._returns = np.zeros(self._model.n)

  def current_player(self):
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
    return pyspiel.PlayerId.CHANCE if self._pending is not None else pyspiel.PlayerId.SIMULTANEOUS

  def is_terminal(self):
    return self._pending is None and (self._model.is_terminal(self._s) or self._t >= self.get_game().max_steps)

  def _legal_actions(self, player):
    # θ̃_{i,e,f} = 0.0 is forced where θ_{e,f} = 0.0, so the legal reports depend on θ_i (observed by i)
    return self._model.legal_communications(self._s, player)

  def chance_outcomes(self):
    return [(i, p) for i, (_, p) in enumerate(self._pending)]

  def _apply_actions(self, actions):
    # Members: report θ̃ and pay its cost, then x(·|φ,θ̃)
    comms = list(actions)
    self._history.append(comms)
    self._rewards = - np.array([self._model.communication_cost(i, c) for i, c in enumerate(comms)])
    self._returns += self._rewards
    self._pending = self._model.aggregation(self._s, comms)

  def _apply_action(self, action):
    outcome = self._pending[action][0]
    self._pending = None
    self._history.append(outcome)
    if isinstance(outcome, int):
      # Chance: drew a from x(·|φ,θ̃), then R_i(s, a) and T(s, a, ·)
      rewards = np.array([self._model.reward(self._s, outcome, i) for i in range(self._model.n)])
      self._rewards += rewards
      self._returns += rewards
      self._pending = self._model.transitions(self._s, outcome)
      return
    # Chance: drew s^0 or s^{t+1}
    is_first = self._s is None
    self._s = outcome
    if is_first:
      return
    self._t += 1
    if self._t >= self.get_game().max_steps and not self._model.is_terminal(self._s):
      # Penalty for truncation
      self._rewards -= self.get_game().penalty
      self._returns -= self.get_game().penalty

  def _action_to_string(self, player, action):
    if player == pyspiel.PlayerId.CHANCE:
      return f"chance:{action}"
    til = self._model.communications[player][action]
    cells = [f"e{e}-f{f}={til[e, f]:g}" for e, f in np.argwhere(til != core.NO_REPORT)]
    return f"p{player}:" + (",".join(cells) if cells else "none")

  def rewards(self):
    return self._rewards.tolist()

  def returns(self):
    return self._returns.tolist()

  def __str__(self):
    if self._s is None:
      return f"t={self._t} s=None"
    phi = self._model.phi_index(self._s.phi)
    return f"t={self._t} phi={phi} theta={self._s.theta.tolist()} last={self._history[-1] if self._t else None}"
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Observer: Implements the OpenSpiel API for observations
# [ θ̃^{t-1} (m·k, ∅ = NO_REPORT) | a^{t-1} (n·m·k) | φ^t (m·k) | θ_i^t (m·k, 0 for f ∉ F_i) ]: o_i^t as it is
#   t = 0: o_i^0 = O(,,s^0), so θ̃ = ∅ and a = 0
# Information state = o_i^t as well: the policy is π_i(θ̃_i^t | o_i^t), not conditioned on h_i^t or t
#   (imperfect recall: the same o_i^t may recur on one path)
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class POSGObserver:

  def __init__(self, game, perfect_recall, params=None):
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    del perfect_recall  # Both observation and information state are o_i^t
    n, m, k = game.model.n, game.model.m, game.model.k
    sizes = {"til_theta": (m, k), "a": (n, m, k), "phi": (m, k), "theta": (m, k)}
    self.tensor = np.zeros(sum(np.prod(shape) for shape in sizes.values()), np.float32)
    self.dict, offset = {}, 0
    for name, shape in sizes.items():
      self.dict[name] = self.tensor[offset:offset + np.prod(shape)].reshape(shape)
      offset += np.prod(shape)

  def set_from(self, state, player):
    self.tensor.fill(0.0)
    o = _observation(state, player)
    if o is None:
      return
    til_theta, a, phi, theta_i = o
    self.dict["til_theta"][:] = til_theta
    self.dict["a"][:] = a
    self.dict["phi"][:] = phi
    self.dict["theta"][:, state._model.own[player]] = theta_i  # pylint: disable=protected-access

  def string_from(self, state, player):
    o = _observation(state, player)
    if o is None:
      return ""
    til_theta, a, phi, theta_i = o
    return f"o=({til_theta.tolist()},{a.astype(int).tolist()},{state._model.phi_index(phi)},{theta_i.tolist()})"  # pylint: disable=protected-access

def _observation(state, player):
  # o_i^t = (θ̃^{t-1}, a^{t-1}, φ^t, θ_i^t); None before drawing s^0
  history, model = state._history, state._model  # pylint: disable=protected-access
  if not history:
    return None
  t = (len(history) - 1) // 3
  if t == 0:
    s0 = history[0]
    return (np.full((model.m, model.k), core.NO_REPORT), np.zeros((model.n, model.m, model.k)),
            s0.phi, s0.theta[:, model.own[player]])
  return model.observation(*history[3 * t - 2:3 * t + 1], player)
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Register the game with OpenSpiel
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
pyspiel.register_game(_GAME_TYPE, POSG)
