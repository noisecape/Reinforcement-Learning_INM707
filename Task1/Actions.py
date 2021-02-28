from collections import namedtuple


class Actions:

    def __init__(self):

        Action = namedtuple('Action', 'name index delta_i delta_j')

        up = Action('up', 0, -1, 0)
        down = Action('down', 1, 1, 0)
        left = Action('left', 2, 0, -1)
        right = Action('right', 3, 0, 1)

        self.index_to_action = {}
        for action in [up, down, left, right]:
            self.index_to_action[action.index] = action
