from abc import ABC, abstractmethod
import numpy as np


class Policy(ABC):

    @abstractmethod
    def take_action(self):
        pass


class E_Greedy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def take_action(self, current_location, q_values):
        if np.random.random() < self.epsilon:  # Act greedly
            idx_action = np.argmax(q_values[current_location[0], current_location[1]])
            return idx_action
        else:
            return np.random.randint(0, 3)