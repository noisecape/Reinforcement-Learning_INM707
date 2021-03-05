from collections import namedtuple


class Agent:

    Action = namedtuple('Action', ['id', 'name', 'idx_i', 'idx_j'])
    up = Action(0, 'up', -1, 0)
    left = Action(1, 'left', 0, -1)
    right = Action(2, 'right', 0, 1)
    idx_to_action = {}
    for action in [up, left, right]:
        idx_to_action[action.id] = action

    def __init__(self, start_location, policy):
        self.current_location = start_location
        self.policy = policy

    def jump(self, q_values):
        """
        This function implements the behaviour of the agent within the environment.
        :param q_values: the data structure that holds all the values for the pair (s,a).
        :param board: the representation of the environment.
        :return current_location: the new updated location of the agent
        """
        action_idx = self.policy.take_action(self.current_location, q_values)
        current_action = Agent.idx_to_action[action_idx]
        action = current_action.idx_i, current_action.idx_j
        self.current_location = self.current_location[0] + action[0], self.current_location[1] + action[1]
        return self.current_location, current_action

