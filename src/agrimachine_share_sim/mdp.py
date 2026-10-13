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
  "theta_space": "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
  "phi_params": "0.7,0.3,0.4,0.6",
  "theta_params": "0.1",
  "mu": 0.05,
  "c": 0.0,
  # Guarantees that an episode will always end.
  "max_steps": 1_000,
}
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# GameType: Declare the "nature" of the game
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
_GAME_TYPE = pyspiel.GameType(
    short_name="python_agrimachine_share_mdp",
    long_name="Python Agrimachine Share MDP",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=1,
    min_num_players=1,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
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
class MDP(pyspiel.Game):
  def __init__(self, params=None):
    params = {**_DEFAULT_PARAMS, **(params or {})}
    self.model = _load_model(tuple(sorted(params.items())))
    self.max_steps = params["max_steps"]
    self.penalty = params["num_machine"] * params["num_field"]

    super().__init__(
        _GAME_TYPE,
        pyspiel.GameInfo(
            num_distinct_actions= self.model.num_actions,
            max_chance_outcomes= self.model.max_chance_outcomes,
            num_players=1,
            min_utility= - self.max_steps * self.model.max_cost - self.penalty,
            max_utility= self.max_steps * self.model.max_reward,
            utility_sum=None,
            # Chance turns are also counted: initial draw + (action, chance) per period
            max_game_length= 2 * self.max_steps + 1,
        ),
        params,
    )

  def new_initial_state(self):
    return MDPState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    return MDPObserver(self, params)
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# State: Holds the current state, time step, and whether it's a chance turn. Implements the OpenSpiel API.
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class MDPState(pyspiel.State):
  def __init__(self, game):
    super().__init__(game)
    self._model = game.model
    self._s = None                                        # core.State; None = before drawing from Pr(s^0)
    self._t = 0                                           # Elapsed periods
    self._pending = self._model.initial_distribution()    # Chance outcomes [(State, p), ...]; None = player's turn
    self._reward = 0.0                                    # Reward of the last action (do NOT reset it at chance turns)
    self._return = 0.0

  def current_player(self):
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
    return pyspiel.PlayerId.CHANCE if self._pending is not None else 0

  def is_terminal(self):
    return self._pending is None and (self._model.is_terminal(self._s) or self._t >= self.get_game().max_steps)

  def _legal_actions(self, player):
    assert player == 0
    return self._model.legal_actions(self._s)

  def chance_outcomes(self):
    return [(i, p) for i, (_, p) in enumerate(self._pending)]

  def _apply_action(self, action):
    if self._pending is None:
      # Player: R(s, a) and T(s, a, ·)
      self._reward = self._model.reward(self._s, action)
      self._return += self._reward
      self._pending = self._model.transitions(self._s, action)
      return
    # Chance: draw s^0 or s^{t+1}
    is_first = self._s is None
    self._s = self._pending[action][0]
    self._pending = None
    if is_first:
      return
    self._t += 1
    if self._t >= self.get_game().max_steps and not self._model.is_terminal(self._s):
      # Penalty for truncation
      self._reward -= self.get_game().penalty
      self._return -= self.get_game().penalty

  def _action_to_string(self, player, action):
    if player == pyspiel.PlayerId.CHANCE:
      return f"chance:{action}"
    tasks = np.argwhere(self._model.actions[action])
    if len(tasks) == 0:
      return "idle"
    return ",".join(f"i{i}-e{e}-f{f}" for i, e, f in tasks)

  def rewards(self):
    return [self._reward]

  def returns(self):
    return [self._return]

  def __str__(self):
    if self._s is None:
      return f"t={self._t} s=None"
    phi = self._model.phi_index(self._s.phi)
    return f"t={self._t} phi={phi} theta={self._s.theta.tolist()}"
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Observer: Implements the OpenSpiel API for observations
# [ φ (m·k) | θ (m·k) ]: the state s = (φ, θ) as it is
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class MDPObserver:

  def __init__(self, game, params=None):
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    m, k = game.model.m, game.model.k
    self.tensor = np.zeros(2 * m * k, np.float32)
    self.dict = {"phi": self.tensor[:m * k].reshape(m, k),
                 "theta": self.tensor[m * k:].reshape(m, k)}

  def set_from(self, state, player):
    del player
    s = state._s  # pylint: disable=protected-access
    if s is None:
      self.tensor.fill(0.0)
      return
    self.dict["phi"][:] = s.phi
    self.dict["theta"][:] = s.theta

  def string_from(self, state, player):
    del player
    s, model = state._s, state._model  # pylint: disable=protected-access
    if s is None:
      return ""
    return f"t={state._t} s={model.encode(s)}"  # pylint: disable=protected-access
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Register the game with OpenSpiel
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
pyspiel.register_game(_GAME_TYPE, MDP)
