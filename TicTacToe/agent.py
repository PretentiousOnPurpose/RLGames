import numpy as np
import itertools

class Agent():
    def __init__(self, epsilon, step_size, gamma, sym):
        self.q_values = {}
        self.epsilon = epsilon
        self.step_size = step_size
        self.gamma = gamma
        self.sym = sym
    
    def update(self, current_state, current_action, next_state, reward, done):
        h_cs = str(np.array(current_state, dtype=int).reshape(-1,) * self.sym)
        h_ns = str(np.array(next_state, dtype=int).reshape(-1,) * self.sym)
        h_a = 3 * current_action[0] + current_action[1]

        if h_cs not in self.q_values.keys():
            self.q_values[h_cs] = np.zeros((9,))
        
        if h_ns not in self.q_values.keys():
            self.q_values[h_ns] = np.zeros((9,))
        
        if not done:
            self.q_values[h_cs][h_a] += self.step_size * (reward + np.max(self.q_values[h_ns]) - self.q_values[h_cs][h_a])
        else:
            self.q_values[h_cs][h_a] += self.step_size * (reward - self.q_values[h_cs][h_a])

    def greedy_action(self, current_state, action_space):
        pr = np.random.rand(1)[0]
        h_cs = str(np.array(current_state, dtype=int).reshape(-1,) * self.sym)
        h_a = 3 * action_space[:, 0] + action_space[:, 1]

        if h_cs not in self.q_values.keys():
            self.q_values[h_cs] = np.zeros((9,))

        q_ps = self.q_values[h_cs]

        if pr <= 1 - self.epsilon:
            next_action_idx = np.argmax(q_ps[h_a])
            next_action = h_a[next_action_idx]    
        else:
            next_action_idx = np.random.choice(np.linspace(0, len(h_a) - 1, len(h_a)))
            next_action = h_a[int(next_action_idx)]    
        
        return [int(next_action / 3), int(next_action % 3)]