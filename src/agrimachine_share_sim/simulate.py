import numpy as np

class Member:
    def __init__(self, id):
        self.id = id
        self.holdings : set[int] = self._map_holdings()
        
    def _map_holdings(self):
        return set()
    
    def observe(self):
        pass
    
    def declare(self):
        pass
    
def simulate(policies, aggregate_policy):
    rewards_trend = []
    revenues = np.zeros(n)
    
    state = _initialize_state()
    observations = _initialize_observations(state)
    while True:
        if state.is_terminal():
            break
        feasibility, grade = state
        
        grade_estimations = [member[i].estimate(observations[i]) for i in range(n)]
        declarations = _summarize_declarations(feasibility, grade_estimations, policies)
        action = _select_action(feasibility, declarations, aggregate_policy)
        
        rewards = _generate_rewards(feasibility, grade, declarations, action)
        rewards_trend.append(rewards)
        revenues += rewards
        
        next_feasibility = _transit_feasibility(feasibility, action)
        next_grade = _transit_grade(grade, next_feasibility, action)
        
        state = (next_feasibility, next_grade)
        observations = _observe(declarations, action, state)
        
    return revenues, rewards_trend

