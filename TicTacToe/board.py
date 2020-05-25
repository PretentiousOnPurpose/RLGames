import numpy as np 
import itertools

class Board():
    def __init__(self):
        self.current_state = np.zeros((3, 3))
        self.action_space = np.array([list(action) for action in itertools.product(range(3), range(3))])

    def getValidActionSpace(self):
        tmpState = np.copy(self.current_state)
        return np.array(np.where(tmpState == 0)).T

    def step(self, action, user):
        self.current_state[tuple(action)] = user
        [done, reward] = self.checkStatus(self.current_state, user)
        return [self.current_state, reward, done]

    def reset(self):
        self.current_state = np.zeros((3, 3))

    def checkStatus(self, current_state, user):
        tmpState = np.copy(current_state)
        
        if (tmpState[:, 0] == user).all() or (tmpState[:, 1] == user).all() or (tmpState[:, 2] == user).all():
            return [1, 20]
        elif (tmpState[0, :] == user).all() or (tmpState[1, :] == user).all() or (tmpState[2, :] == user).all():
            return [1, 20]
        elif tmpState[0, 0] == user and tmpState[1, 1] == user and tmpState[2, 2] == user:
            return [1, 20]
        elif tmpState[0, 2] == user and tmpState[1, 1] == user and tmpState[2, 0] == user:
            return [1, 20]
        elif (tmpState[:, 0] == -user).all() or (tmpState[:, 1] == -user).all() or (tmpState[:, 2] == -user).all():
            return [1, -20]
        elif (tmpState[0, :] == -user).all() or (tmpState[1, :] == -user).all() or (tmpState[2, :] == -user).all():
            return [1, -20]
        elif tmpState[0, 0] == -user and tmpState[1, 1] == -user and tmpState[2, 2] == -user:
            return [1, -20]
        elif tmpState[0, 2] == -user and tmpState[1, 1] == -user and tmpState[2, 0] == -user:
            return [1, -20]
        elif (tmpState != 0).all():
            return [1, 5]
        else:
            return [0, 1]

    def render(self):
        tmpState = np.copy(self.current_state)
        tmpState = np.array(np.array(tmpState, dtype=int), dtype=str)
        
        tmpState[tmpState == '0'] = '-'
        tmpState[tmpState == '1'] = 'X'
        tmpState[tmpState == '-1'] = 'O'

        print(tmpState)