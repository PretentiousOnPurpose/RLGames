import pickle
import numpy as np
from board import Board 
from agent import Agent

a1 = Agent(0.7, 0.4, 1, 1)
a2 = Agent(0.7, 0.4, 1, -1)
env = Board()

numEp = 20000

for i in range(numEp):
    done = False
    env.reset()

    while not done:
        current_state1 = np.copy(env.current_state)
        action_space = env.getValidActionSpace()
        if len(action_space) == 0:
            break
        current_action1 = a1.greedy_action(current_state1, action_space)
        [next_state1, reward1, done] = env.step(current_action1, 1)
        a1.update(current_state1, current_action1, next_state1, reward1, done)

        if done and reward1 == 20:
            a2.update(current_state2, current_action2, next_state2, -reward1, done)
        elif done and reward1 == 5:
            a2.update(current_state2, current_action2, next_state2, reward1, done)
        else:
            current_state2 = np.copy(env.current_state)
            action_space = env.getValidActionSpace()
            if len(action_space) == 0:
                break
            current_action2 = a2.greedy_action(current_state2, action_space)
            [next_state2, reward2, done] = env.step(current_action2, -1)
            a2.update(current_state2, current_action1, next_state2, reward2, done)
            
            if done and reward2 == 20:
                a1.update(current_state1, current_action1, next_state1, -reward2, done)
            elif done and reward2 == 5:
                a1.update(current_state1, current_action1, next_state1, reward2, done)

    print("Epoch ", str(i + 1), " Done")

f1 = open('a1.pkl', 'wb')
f2 = open('a2.pkl', 'wb')

pickle.dump(a1, f1)
pickle.dump(a2, f2)
