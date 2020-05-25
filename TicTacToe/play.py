import pickle
import numpy as np
from board import Board
from agent import Agent

a1 = pickle.load(open('a1.pkl', 'rb'))
a2 = pickle.load(open('a2.pkl', 'rb'))
env = Board()

a1.epsilon = 0
a2.epsilon = 0

done = False
env.reset()

while not done:
    current_state1 = np.copy(env.current_state)
    action_space = env.getValidActionSpace()
    if len(action_space) == 0:
        break
    current_action1 = a1.greedy_action(current_state1, action_space)
    [next_state1, reward1, done] = env.step(current_action1, 1)
    env.render()
    if not done:
        current_state2 = np.copy(env.current_state)
        action_space = env.getValidActionSpace()
        if len(action_space) == 0:
            break
        current_action2 = a2.greedy_action(current_state2, action_space)
        [next_state2, reward2, done] = env.step(current_action2, -1)
        env.render()