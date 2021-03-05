from collections import namedtuple


class Agent:

    def __init__(self, start_location):
        self.current_location = start_location

    def jump(self, action):
        """
        This function implements the behaviour of the agent within the environment.
        :param action: the data structure that holds all the values for the pair (s,a).
        :return current_location: the new updated location of the agent
        """
        self.current_location = self.current_location[0] + action.idx_i, self.current_location[1] + action.idx_j
        return self.current_location

