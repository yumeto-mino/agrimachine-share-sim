import itertools
import re
from typing import NamedTuple
import numpy as np

class State(NamedTuple):
    phi: np.ndarray     # (m, k)
    theta: np.ndarray   # (m, k)

class Model:
    def __init__(self, params):
        # FOR CALCULATION (NOT INTERFACE)
        self._legal_cache: dict = {}    # φ.tobytes() -> A(φ)'s index 
        self._index: dict = {}          # (φ, θ).tobytes() -> state index
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # DEFINE PARAMS
        self.n: int = params["num_member"]
        self.m: int = params["num_machine"]
        self.k: int = params["num_field"]
        self.field_owner_params: tuple = tuple([int(x) for x in _split(params["field_owner"])])
        
        self.phi_params: tuple = tuple([float(x) for x in _split(params["phi_params"])])
        self.phi_space: tuple = tuple(_frozen(p) for p in self._phi_space())
        self.theta_params: tuple = tuple([float(x) for x in _split(params["theta_params"])])
        self.theta_space: np.ndarray = np.asarray([float(x) for x in _split(params["theta_space"])])
        if not np.all(np.diff(self.theta_space) > 0):
            raise ValueError(f"theta_space must be strictly ascending; got {params['theta_space']}")
        
        self.mu: float = params["mu"]
        self.c: float = params["c"]
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # CALCULATE
        self.own: np.ndarray = self._field_owner()
        
        self._initial: list = [(State(_frozen(s.phi), _frozen(s.theta)), p)
                               for s, p in self._initial_dist()]

        self.actions: np.ndarray = _frozen(self._create_actions())
        self.num_actions: int = len(self.actions)
        # self.communications: tuple = tuple(self._create_communications())
        # self.num_communications: int = len(self.communications)
        
        self.states: tuple
        self.max_chance_outcomes: int   # max(|supp Pr(s^0)|, max_{s,a} |supp T(s,a,·)|)
        # Pr(s^0), A(φ), T(s,a,s') -> Feasible S
        self.states, self.max_chance_outcomes = self._create_states()
        self.num_states: int = len(self.states)
        
        self.max_reward: float
        self.max_cost: float
        self.max_reward, self.max_cost = self._reward_bounds()

    # 👻: YOU CAN REWRITE ===================================================
    def _field_owner(self) -> np.ndarray:
        if self.field_owner_params:
            owner = np.asarray(self.field_owner_params)
            if len(owner) != self.k or owner.min() < 0 or owner.max() >= self.n:
                raise ValueError(f"field_owner needs {self.k} values in 0..{self.n - 1}; got {self.field_owner_params}")
        else:
            owner = np.arange(self.k) * self.n // self.k
        return owner[None, :] == np.arange(self.n)[:, None]

    def _psi(self, a: np.ndarray) -> bool:
        return bool(
            np.all(a.sum(axis=(1, 2)) <= 1)                 # ∀i: Σ_{e,f} a_{i,e,f} ≤ 1
            and np.all(a.sum(axis=(0, 2)) <= 1)             # ∀e: Σ_{i,f} a_{i,e,f} ≤ 1
            and not np.any(a & ~self.own[:, None, :])       # ∀i: Σ_{e,f∉F_i} a_{i,e,f} = 0
        )

    def _create_actions(self) -> np.ndarray:
        choices = [None] + list(itertools.product(range(self.m), range(self.k)))
        actions = []
        for profile in itertools.product(choices, repeat=self.n):
            a = np.zeros((self.n, self.m, self.k), dtype=bool)
            for i, task in enumerate(profile):
                if task is not None:
                    a[i, task[0], task[1]] = True
            if self._psi(a):
                actions.append(a)
        return np.stack(actions)

    def _phi_space(self) -> list:
        return [np.zeros((self.m, self.k), dtype=bool),
                np.ones((self.m, self.k), dtype=bool)]

    def _w(self, i: int, theta: np.ndarray, a: np.ndarray) -> float:
        rewarded = a[i] & (theta != 0.0) & (theta != 1.0)    # (m, k)
        return float(rewarded[:, self.own[i]].sum())

    def _v(self, i: int, theta: np.ndarray) -> float:
        costed = (theta == 1.0)                              # (m, k)
        return - self.mu * float(costed[:, self.own[i]].sum())

    def _phi_dist(self, phi: np.ndarray, a: np.ndarray) -> list:
        p = self.phi_params
        now = self.phi_index(phi)
        return [(self.phi_space[0], p[2 * now]), (self.phi_space[1], p[2 * now + 1])]

    def _theta_dist(self, theta: np.ndarray, a: np.ndarray) -> list:
        level = self.level(theta)
        next_theta = self.theta_space[np.minimum(level + 1, len(self.theta_space) - 1)]
        next_theta[(theta == 0.0) | a.any(axis=0)] = 0.0
        return [(next_theta, 1.0)]

    def _initial_dist(self) -> list:
        values = list(self.theta_params)
        if len(values) == 1:
            values *= self.m * self.k
        if len(values) != self.m * self.k:
            raise ValueError(f"theta_params needs 1 or {self.m * self.k} values; got {len(values)}")
        if not set(values) <= set(self.theta_space.tolist()):
            raise ValueError(f"theta_params must be in theta_space; got {values}")
        theta = np.asarray(values).reshape(self.m, self.k)
        return [(State(self.phi_space[1], theta), 1.0)]

    def _is_terminal(self, s: State) -> bool:
        return not s.theta.any()

    def _reward_bounds(self) -> tuple:
        return float(self.actions.sum(axis=(1, 2, 3)).max()), self.mu * self.m * self.k
    # =======================================================================

    # 👻: FOR OPEN SPIEL GAME  ==============================================
    def initial_distribution(self) -> list:
        return self._initial

    def legal_actions(self, s: State) -> list:
        key = s.phi.tobytes()
        if key not in self._legal_cache:
            masked = self.actions & s.phi
            match = np.all(masked[:, None] == self.actions[None], axis=(2, 3, 4))
            if not match.any(axis=1).all():
                raise ValueError("A(φ) ⊄ A: some φ∘a is not in A, so it has no action index")
            self._legal_cache[key] = np.flatnonzero(match.any(axis=0)).tolist()
        return self._legal_cache[key]

    def reward(self, s: State, a: int, i: int | None = None) -> float:
        """i を指定すれば w_i + v_i，None なら R(s, a) = Σ_i {w_i + v_i}"""
        action = self.actions[a]
        members = range(self.n) if i is None else (i,)
        return sum(self._w(j, s.theta, action) + self._v(j, s.theta) for j in members)

    def transitions(self, s: State, a: int) -> list:
        """T(s, a, ·) の確率が正の次状態だけを [(State, 確率), ...] で返す"""
        action = self.actions[a]
        thetas = [(_frozen(next_theta), q) for next_theta, q in self._theta_dist(s.theta, action) if q > 0]
        return [(State(next_phi, next_theta), p * q)
                for next_phi, p in self._phi_dist(s.phi, action) if p > 0
                for next_theta, q in thetas]

    def transition(self, s: State, a: int, next_s: State) -> float:
        """T(s, a, s') の点評価（DP 用）"""
        return sum(p for t, p in self.transitions(s, a)
                   if np.array_equal(t.phi, next_s.phi) and np.array_equal(t.theta, next_s.theta))

    def is_terminal(self, s: State) -> bool:
        return self._is_terminal(s)
    # =======================================================================

    # 👻: ENCODING ==========================================================
    def level(self, theta: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.theta_space, theta)

    def phi_index(self, phi: np.ndarray) -> int:
        for idx, p in enumerate(self.phi_space):
            if np.array_equal(p, phi):
                return idx
        raise ValueError("phi is not in phi_space")

    def encode(self, s: State) -> int:
        return self._index[_key(s)]

    def decode(self, x: int) -> State:
        return self.states[x]

    def _create_states(self) -> tuple:
        states, index = [], self._index

        def visit(s: State):
            key = _key(s)
            if key not in index:
                index[key] = len(states)
                states.append(s)

        for s, _ in self._initial:
            visit(s)
        max_outcomes = len(self._initial)
        head = 0
        while head < len(states):
            s = states[head]
            head += 1
            if self._is_terminal(s): continue
            for a in self.legal_actions(s):
                outcomes = self.transitions(s, a)
                max_outcomes = max(max_outcomes, len(outcomes))     # |supp T(s,a,·)|
                for next_s, _ in outcomes:
                    visit(next_s)
        return tuple(states), max_outcomes
    # =======================================================================

    def __deepcopy__(self, memo):
        return self

    # def aggregation(self, state: int, c_action: int):
    #     pass

    # def _create_communications(self):
    #     pass


def _split(text: str) -> list:
    return [x for x in re.split(r"[,;]", text) if x.strip()]


def _key(s: State) -> tuple:
    return s.phi.tobytes(), s.theta.tobytes()


def _frozen(x: np.ndarray) -> np.ndarray:
    """状態は共有されるので書き換え不可にしておく"""
    x = np.array(x)
    x.setflags(write=False)
    return x
