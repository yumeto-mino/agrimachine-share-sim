import itertools
import re
from typing import NamedTuple
import numpy as np

NO_REPORT: float = -1.0     # θ̃_{e,f} = ∅ (no report); outside theta_space, so distinct from reporting 0.0

class State(NamedTuple):
    phi: np.ndarray     # (m, k)
    theta: np.ndarray   # (m, k)

class Model:
    def __init__(self, params):
        # FOR CALCULATION (NOT INTERFACE)
        self._legal_cache: dict = {}    # φ.tobytes() -> A(φ)'s index
        self._comm_cache: dict = {}     # (i, θ.tobytes()) -> legal θ̃_i's index
        self._index: dict = {}          # (φ, θ).tobytes() -> state index
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # DEFINE PARAMS
        self.n: int = params["num_member"]
        self.m: int = params["num_machine"]
        self.k: int = params["num_field"]
        self.field_owner_params: tuple = tuple([int(x) for x in _split(params["field_owner"])])
        
        self.phi_init_params: tuple = tuple([float(x) for x in _split(params["phi_init_params"])])
        self.phi_dist_params: tuple = tuple([float(x) for x in _split(params["phi_dist_params"])])
        self.phi_space: tuple = tuple(_frozen(p) for p in self._phi_space())
        self.theta_init_params: tuple = tuple([float(x) for x in _split(params["theta_init_params"])])
        self.theta_dist_params: tuple = tuple([float(x) for x in _split(params["theta_dist_params"])])
        self.theta_space: np.ndarray = np.asarray([float(x) for x in _split(params["theta_space"])])
        if not np.all(np.diff(self.theta_space) > 0):
            raise ValueError(f"theta_space must be strictly ascending; got {params['theta_space']}")
        if NO_REPORT in self.theta_space:
            raise ValueError(f"theta_space must not contain {NO_REPORT} (reserved for ∅); got {params['theta_space']}")
        self.aggregation_rule: str = params["aggregation"]
        self.strategy_space: str = params["strategy_space"]
        if self.strategy_space not in ("withholder", "deceiver"):
            raise ValueError(f"strategy_space must be withholder or deceiver; got {self.strategy_space}")
        
        self.mu: float = params["mu"]
        self.c: float = params["c"]
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # CALCULATE
        self.own: np.ndarray = self._field_owner()
        
        self._initial: list = [(State(_frozen(s.phi), _frozen(s.theta)), p)
                               for s, p in self._initial_dist()]

        self.actions: np.ndarray = _frozen(self._create_actions())
        self.num_actions: int = len(self.actions)
        self.communications: tuple = tuple(_frozen(c) for c in self._create_communications())
        self.num_communications: int = max(len(c) for c in self.communications)
        
        # max(|supp Pr(s^0)|, max_{s,a} |supp T(s,a,·)|), without enumerating S
        self.max_chance_outcomes: int = max(len(self._initial), self._max_transition_outcomes())
        # Feasible S (Pr(s^0), A(φ), T(s,a,s')): built on first use (DP only), since S can be huge
        self._states: tuple | None = None
        
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

    def _create_communications(self) -> list:
        # Per member i: every θ̃_i ∈ ({∅} ∪ Θ)^{E×F_i}, as (m, k) with f ∉ F_i = ∅
        # θ̃_{i,e,f} = NO_REPORT means "not reported" (∅); 0.0 is a report of θ = 0.0
        result = []
        choices = [NO_REPORT] + self.theta_space.tolist()
        for i in range(self.n):
            cells = [(e, f) for e in range(self.m) for f in np.flatnonzero(self.own[i])]
            comms = []
            for values in itertools.product(choices, repeat=len(cells)):
                til = np.full((self.m, self.k), NO_REPORT)
                for (e, f), v in zip(cells, values):
                    til[e, f] = v
                comms.append(til)
            result.append(np.stack(comms))
        return result
    
    def _psi(self, a: np.ndarray) -> bool:
        return bool(
            np.all(a.sum(axis=(1, 2)) <= 1)                 # ∀i: Σ_{e,f} a_{i,e,f} ≤ 1
            and np.all(a.sum(axis=(0, 2)) <= 1)             # ∀e: Σ_{i,f} a_{i,e,f} ≤ 1
            and not np.any(a & ~self.own[:, None, :])       # ∀i: Σ_{e,f∉F_i} a_{i,e,f} = 0
        )

    def _phi_space(self) -> list:
        return [np.zeros((self.m, self.k), dtype=bool),
                np.ones((self.m, self.k), dtype=bool)]

    def _phi_dist(self, phi: np.ndarray, a: np.ndarray) -> list:
        p = self.phi_dist_params
        now = self.phi_index(phi)
        return [(self.phi_space[0], p[2 * now]), (self.phi_space[1], p[2 * now + 1])]

    def _theta_dist(self, theta: np.ndarray, a: np.ndarray) -> list:
        # TODO: Lists up to 2^{m·k} outcomes per period. Split into one chance node per cell (2 outcomes each)
        #       to speed up simulation; the history layout in posg._observation must change accordingly.
        # each cell: level + 1 w.p. p, keep level w.p. 1 - p (independently); θ = 0 or harvested → 0
        p = self.theta_dist_params[0]
        level = self.level(theta)
        reset = (theta == 0.0) | a.any(axis=0)
        base = theta.copy()
        base[reset] = 0.0
        cells = list(zip(*np.nonzero(~reset & (level < len(self.theta_space) - 1))))
        result = []
        for ups in itertools.product((False, True), repeat=len(cells)):
            next_theta = base.copy()
            prob = 1.0
            for (e, f), up in zip(cells, ups):
                if up:
                    next_theta[e, f] = self.theta_space[level[e, f] + 1]
                prob *= p if up else 1.0 - p
            if prob > 0:
                result.append((next_theta, prob))
        return result

    def _max_transition_outcomes(self) -> int:
        # TODO: Update this bound whenever _phi_dist / _theta_dist change (it is checked only when S is enumerated)
        # Upper bound of |supp T(s,a,·)| = |supp Pr(φ'|φ)|·|supp Pr(θ'|θ,a)|; θ: each cell rises or stays
        return len(self.phi_space) * 2 ** (self.m * self.k)

    def _initial_dist(self) -> list:
        # Pr(s^0): φ^0 ~ phi_init_params (probability of each φ ∈ phi_space), θ^0 = theta_init_params
        if len(self.phi_init_params) != len(self.phi_space):
            raise ValueError(f"phi_init_params needs {len(self.phi_space)} values; got {self.phi_init_params}")
        values = list(self.theta_init_params)
        if len(values) == 1:
            values *= self.m * self.k
        if len(values) != self.m * self.k:
            raise ValueError(f"theta_init_params needs 1 or {self.m * self.k} values; got {len(values)}")
        if not set(values) <= set(self.theta_space.tolist()):
            raise ValueError(f"theta_init_params must be in theta_space; got {values}")
        theta = np.asarray(values).reshape(self.m, self.k)
        return [(State(phi, theta), p) for phi, p in zip(self.phi_space, self.phi_init_params) if p > 0]

    def _aggregation_dist(self, phi: np.ndarray, til_theta: np.ndarray) -> list:
        # x(a | φ, θ̃): tasks with φ_{e,f} = 0 are not assignable
        til_theta = np.where(phi, til_theta, 0.0)
        if self.aggregation_rule == "bid_to_win":
            profiles = _bid_to_win(til_theta)
        elif self.aggregation_rule == "scale_to_win":
            profiles = [_scale_to_win(til_theta)]
        else:
            raise ValueError(f"aggregation must be bid_to_win or scale_to_win; got {self.aggregation_rule}")
        actions = []
        for profile in profiles:
            a = np.zeros((self.n, self.m, self.k), dtype=bool)
            for task in profile:
                a[task] = True
            if self._psi(a):
                actions.append(a)
        return [(a, 1.0 / len(actions)) for a in actions]
    
    def _is_terminal(self, s: State) -> bool:
        return not s.theta.any()

    def _w(self, i: int, theta: np.ndarray, a: np.ndarray) -> float:
        rewarded = a[i] & (theta != 0.0) & (theta != 1.0)    # (m, k)
        return float(rewarded[:, self.own[i]].sum())

    def _v(self, i: int, theta: np.ndarray) -> float:
        costed = (theta == 1.0)                              # (m, k)
        return - self.mu * float(costed[:, self.own[i]].sum())
    
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

    def legal_communications(self, s: State, i: int) -> list:
        """Indices of the θ̃_i member i can report in s (θ_{e,f} = 0.0 forces θ̃_{e,f} = 0.0).
        withholder: θ̃_{e,f} ∈ {θ_{e,f}, ∅}, deceiver: θ̃_{e,f} ∈ Θ ∪ {∅}"""
        key = (i, s.theta.tobytes())
        if key not in self._comm_cache:
            comms = self.communications[i]                          # (C, m, k), f ∉ F_i = ∅
            if self.strategy_space == "withholder":
                ok = (comms == s.theta) | (comms == NO_REPORT)
            else:
                ok = np.ones(comms.shape, dtype=bool)
            ok &= (comms == 0.0) | (s.theta != 0.0)
            self._comm_cache[key] = np.flatnonzero(np.all(ok | ~self.own[i], axis=(1, 2))).tolist()
        return self._comm_cache[key]

    def reward(self, s: State, a: int, i: int | None = None) -> float:
        """w_i + v_i of member i, or R(s, a) = Σ_i (w_i + v_i) if i is None"""
        action = self.actions[a]
        members = range(self.n) if i is None else (i,)
        return sum(self._w(j, s.theta, action) + self._v(j, s.theta) for j in members)

    def transitions(self, s: State, a: int) -> list:
        """[(next state, prob), ...] with prob > 0 under T(s, a, ·)"""
        action = self.actions[a]
        thetas = [(_frozen(next_theta), q) for next_theta, q in self._theta_dist(s.theta, action) if q > 0]
        return [(State(next_phi, next_theta), p * q)
                for next_phi, p in self._phi_dist(s.phi, action) if p > 0
                for next_theta, q in thetas]

    def transition(self, s: State, a: int, next_s: State) -> float:
        """T(s, a, s') (for DP)"""
        return sum(p for t, p in self.transitions(s, a)
                   if np.array_equal(t.phi, next_s.phi) and np.array_equal(t.theta, next_s.theta))

    def is_terminal(self, s: State) -> bool:
        return self._is_terminal(s)

    def observation(self, comms: list, a: int, s: State, i: int) -> tuple:
        """o_i^t = (θ̃^{t-1}, a^{t-1}, φ^t, θ_i^t) for t ≥ 1. Shapes: θ̃ (m, k), a (n, m, k), θ_i (m, |F_i|)"""
        til_theta = np.full((self.m, self.k), NO_REPORT)
        for j, c in enumerate(comms):
            til_theta[:, self.own[j]] = self.communications[j][c][:, self.own[j]]
        return til_theta, self.actions[a], s.phi, s.theta[:, self.own[i]]

    def aggregation(self, s: State, comms: list) -> list:
        """[(action index, prob), ...] with prob > 0 under x(· | φ, θ̃); comms[i] is the index of θ̃_i"""
        til_theta = np.stack([self.communications[i][c] for i, c in enumerate(comms)])
        legal = self.legal_actions(s)
        result = []
        for a, p in self._aggregation_dist(s.phi, til_theta):
            if p <= 0: continue
            index = np.flatnonzero(np.all(self.actions == a, axis=(1, 2, 3)))
            if len(index) == 0 or index[0] not in legal:
                raise ValueError("x(·|φ, θ̃) puts probability on an action not in A(φ)")
            result.append((int(index[0]), p))
        return result

    def communication_cost(self, i: int, c: int) -> float:
        """c · #{(e, f): θ̃_{i,e,f} ∉ {∅, 0.0}}; reporting 0.0 is free"""
        til = self.communications[i][c]
        return self.c * float(np.count_nonzero((til != NO_REPORT) & (til != 0.0)))
    # =======================================================================

    # 👻: ENCODING ==========================================================
    def level(self, theta: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.theta_space, theta)

    def phi_index(self, phi: np.ndarray) -> int:
        for idx, p in enumerate(self.phi_space):
            if np.array_equal(p, phi):
                return idx
        raise ValueError("phi is not in phi_space")

    @property
    def states(self) -> tuple:
        """Feasible states (for DP), enumerated on first access. Never finishes when S is large: small settings only"""
        if self._states is None:
            self._states = self._create_states()
        return self._states

    @property
    def num_states(self) -> int:
        return len(self.states)

    def encode(self, s: State) -> int:
        _ = self.states     # Builds self._index
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
        head = 0
        while head < len(states):
            s = states[head]
            head += 1
            if self._is_terminal(s): continue
            for a in self.legal_actions(s):
                outcomes = self.transitions(s, a)
                if len(outcomes) > self.max_chance_outcomes:
                    raise ValueError(f"|supp T(s,a,·)| = {len(outcomes)} exceeds _max_transition_outcomes()")
                for next_s, _ in outcomes:
                    visit(next_s)
        return tuple(states)
    # =======================================================================

    def __deepcopy__(self, memo):
        return self


def _bid_to_win(til_theta: np.ndarray) -> set:
    """Profiles filled greedily from the highest θ̃_{i,e,f} (branching on ties)"""
    cand = sorted(((til_theta[i, e, f], (i, e, f)) for i, e, f in np.argwhere(til_theta > 0)),
                  key=lambda x: -x[0])
    profiles, visited, stack = set(), set(), [frozenset()]
    while stack:
        chosen = stack.pop()
        if chosen in visited: continue
        visited.add(chosen)
        used_i = {t[0] for t in chosen}
        used_e = {t[1] for t in chosen}
        top, nexts = None, []
        for v, (i, e, f) in cand:
            if top is not None and v < top: break
            if i in used_i or e in used_e: continue
            top = v
            nexts.append((i, e, f))
        if not nexts:
            profiles.add(chosen)
            continue
        for task in nexts:
            stack.append(chosen | {task})
    return profiles


def _scale_to_win(til_theta: np.ndarray) -> frozenset:
    """Members with more reported tasks go first; each gets its highest-θ̃ task"""
    n, m, k = til_theta.shape
    task_count = [int((til_theta[i] > 0).sum()) for i in range(n)]
    profile, used_e = set(), set()
    for i in sorted(range(n), key=lambda i: (-task_count[i], i)):
        if task_count[i] == 0: break
        best_val, best_task = 0.0, None
        for e in range(m):
            if e in used_e: continue
            for f in range(k):
                if til_theta[i, e, f] > best_val:
                    best_val, best_task = til_theta[i, e, f], (i, e, f)
        if best_task is None: continue
        profile.add(best_task)
        used_e.add(best_task[1])
    return frozenset(profile)


def _split(text: str) -> list:
    return [x for x in re.split(r"[,;]", text) if x.strip()]


def _key(s: State) -> tuple:
    return s.phi.tobytes(), s.theta.tobytes()


def _frozen(x: np.ndarray) -> np.ndarray:
    """Read-only, since states are shared"""
    x = np.array(x)
    x.setflags(write=False)
    return x
