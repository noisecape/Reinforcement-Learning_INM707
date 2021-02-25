import numpy as np
import enum
from collections import namedtuple


class EnemyUtils(enum.IntEnum):
    ARROW_DAMAGE = -5
    FIGHT_DAMAGE = -10


Action = namedtuple('Action', 'name index delta_i delta_j')

up = Action('up', 0, -1, 0)
down = Action('down', 1, 1, 0)
left = Action('left', 2, 0, -1)
right = Action('right', 3, 0, 1)

index_to_action = {}
for action in [up, down, left, right]:
    index_to_action[action.index] = action


class Enemy:

    def __init__(self, start_location):
        self.current_location = start_location

    def move(self):
        """
        This function defines how the enemy moves in the board. At each time step each enemy has
        0.1 probability to move randomly in one of its adjacent locations, while the remaining value (0.6)
        expresses the probability not to take any action.
        :return x,y: the new possible coordinates if the enemy moves. Otherwise x,y are the previous coordinates.
        """
        x, y = self.current_location
        actions_probability = [0.1, 0.1, 0.1, 0.1, 0.6]
        index_action = np.argmax(np.random.multinomial(1, actions_probability))

        if index_action == 4: # don't move
            return x, y
        else:
            current_action = index_to_action[index_action]
            x, y = (x+current_action.delta_i, y+current_action.delta_j)
            return x, y





