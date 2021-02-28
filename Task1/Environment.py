import numpy as np
from Agent import Agent
import enum


class EnvironmentUtils(enum.IntEnum):
    """
    This class represents a series of utils used to display
    the board on the console.
    """
    FREE_LOCATION = 0
    WALL = 1
    AGENT = 2
    CAR = 3
    TRUCK = 4
    SAFE = 5


class Environment:
    """
    This class represents the environment of the game.
    It is responsible for the creation of the starting environment of the game. The class also
    handles the actions taken by the agent, updating the board accordingly.
    """

    def __init__(self, N=20, difficulty=1):
        """
        Inits the environment of the board.
        :param N : The dimension of the board. It should be a square board of (N,N). Default = 20
        :param difficulty: The diffuculty of the game. It can have a value betwen 1 to 3. The higher the difficulty,
        the more the number of roads to cross and cars to avoid on the grid world.
        """
        self.dimension = N

        self.board = self.init_board()
        self.agent = self.init_agent()
        self.difficulty = difficulty
        self.display_board()

    def init_agent(self):
        """
        This function first search for a valid location for the agent, then instantiates the agent
        with the valid location. Note that the agent at the start of the game can only spawn in
        one of the possible columns of the the lower row of the board (excluding the walls).
        :return agent: The agent of the environment:
        """
        agent_location = (self.dimension-2, np.random.randint(1, self.dimension-1))
        agent = Agent(agent_location)
        self.board[agent_location[0], agent_location[1]] = EnvironmentUtils.AGENT
        return agent

    def init_board(self):
        """
        This function set up the walls for the board.
        :return board: The board of dimension (N,N):

        """
        board = np.zeros((self.dimension, self.dimension))
        board[:1, :] = EnvironmentUtils.WALL
        board[:, :1] = EnvironmentUtils.WALL
        board[-1, :] = EnvironmentUtils.WALL
        board[:, -1] = EnvironmentUtils.WALL
        return board

    def generate_roads(self):
        """
        Generates the roads on the grid and updates the board accordingly.
        """
        pass

    def generate_envir_resource(self):
        """
        This function generates a tuple of valid coordinates in the board.
        :return x_coord, y_coord: The pair of valid coordinates.
        """
        while True:
            x_coord = np.random.randint(2, self.dimension - 2)
            y_coord = np.random.randint(1, self.dimension - 1)
            if self.board[x_coord, y_coord] == EnvironmentUtils.FREE_LOCATION:
                break
        return x_coord, y_coord

    def display_board(self):
        """
        This function displays the board.
        """
        display_helper = {0: '.',
                          1: 'X',
                          2: 'A',
                          3: 'E',
                          4: 'B',
                          5: 'F',
                          6: 'G'}
        print('Number of roads to cross: {}'.format(self.n_bombs))
        print('Lives: {}'.format(self.agent.lives))
        for row in range(self.dimension):
            for column in range(self.dimension):
                print(display_helper[self.board[row, column]], end=' ')
            print()

    def reset(self):
        self.board = self.init_board()
        self.agent = self.init_agent()


    def step(self):
        """
        This function defines the loop of the game and it's called at every time steps.
        :return:
        """
        for enemy in self.enemies:
            new_x, new_y = enemy.move()
            # TO DO


env = Environment()