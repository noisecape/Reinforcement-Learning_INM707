from Policy import Policy
import numpy as np

class E_Greedy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def take_action(self, current_location, q_values):
        if np.random.random() < self.epsilon:  # Act greedly
            return np.argmax(q_values[current_location[0], current_location[1]])
        else:
            return np.random.randint(0, 3)
