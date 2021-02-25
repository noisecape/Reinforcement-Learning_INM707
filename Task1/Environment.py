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
    ENEMY = 3
    BOMB = 4
    FLAG = 5
    GOLD = 6


class Environment:
    """
    This class represents the environment of the game.
    It is responsible for the creation of the starting environment of the game. The class also
    handles the actions taken by the agent, updating the board accordingly.
    """

    def __init__(self, N=20, n_enemies=10, n_gold=15, n_bombs=10):
        """
        Inits the environment of the board.
        :param N : The dimension of the board. It should be a square board of (N,N). Default = 20
        :param n_enemies : The number of enemies in the game. Default = 10
        :param n_gold : The number of gold resources in the game. Default = 15
        :param n_bombs : The number of bombs in the game. Default = 10
        """
        self.dimension = N
        self.board = self.init_board()
        self.agent = self.init_agent()
        self.flag_location = self.generate_flag()
        self.enemies = self.generate_enemies(n_enemies)
        self.gold = self.generate_gold(n_gold)
        self.bombs = self.generate_bombs(n_bombs)
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

    def generate_flag(self):
        """
        This function generate a valid location for the flag. It then updates the board accordingly.
        The flag at the start of the game can only spawn in one of the possible columns of the the upper
        row of the board (excluding the walls).
        :return flag_location : The x,y coordinates of the flag.:
        """
        flag_location = (1, np.random.randint(1, self.dimension-1))
        self.board[flag_location[0], flag_location[1]] = EnvironmentUtils.FLAG
        return flag_location

    def generate_enemies(self, n_enemies):
        """
        Generates the enemies locations and updates the board accordingly.
        :param n_enemies : The number of enemies to generate.
        :return enemies_locations : The list of all the enemies locations
        """
        enemies_locations = []
        for _ in range(n_enemies):
            x, y = self.generate_envir_resource()
            enemies_locations.append((x, y))
            self.board[x, y] = EnvironmentUtils.ENEMY
        return enemies_locations

    def generate_gold(self, n_gold):
        """
        Generates the gold locations and updates the board accordingly.
        :param n_gold: The number of gold resources to generate.
        :return gold_locations: The list of all the gold resources locations.
        """
        gold_locations = []
        for _ in range(n_gold):
            x, y = self.generate_envir_resource()
            gold_locations.append((x, y))
            self.board[x, y] = EnvironmentUtils.GOLD
        return gold_locations

    def generate_bombs(self, n_bombs):
        """
        Generates the bombs locations and updates the board accordingly.
        :param n_bombs: The number of bombs to generate.
        :return n_bombs: The list of all the bombs locations.
        """
        bombs_locations = []
        for _ in range(n_bombs):
            x, y = self.generate_envir_resource()
            bombs_locations.append((x, y))
            self.board[x, y] = EnvironmentUtils.BOMB
        return bombs_locations

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

        for row in range(self.dimension):
            for column in range(self.dimension):
                print(display_helper[self.board[row, column]], end=' ')
            print()


env = Environment()