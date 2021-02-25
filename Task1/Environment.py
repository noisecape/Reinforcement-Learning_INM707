import numpy as np
from Agent import Agent
import enum


class EnvironmentUtils(enum.IntEnum):
    FREE_LOCATION = 0
    WALL = 1
    AGENT = 2
    ENEMY = 3
    BOMB = 4
    FLAG = 5
    GOLD = 6


class Environment:

    def __init__(self, N=20, n_enemies=10, n_gold=15, n_bombs=10):
        self.dimension = N
        self.board = self.init_board()
        self.agent = self.init_agent()
        self.flag_location = self.generate_flag()
        self.enemies = self.generate_enemies(n_enemies)
        self.gold = self.generate_gold(n_gold)
        self.bombs = self.generate_bombs(n_bombs)
        self.display_board()

    def init_agent(self):
        agent_location = (self.dimension-2, np.random.randint(1, self.dimension-1))
        agent = Agent(agent_location)
        self.board[agent_location[0], agent_location[1]] = EnvironmentUtils.AGENT
        return agent

    def init_board(self):
        board = np.zeros((self.dimension, self.dimension))
        board[:1, :] = EnvironmentUtils.WALL
        board[:, :1] = EnvironmentUtils.WALL
        board[-1, :] = EnvironmentUtils.WALL
        board[:, -1] = EnvironmentUtils.WALL
        return board

    def generate_flag(self):
        flag_location = (1, np.random.randint(1, self.dimension-1))
        self.board[flag_location[0], flag_location[1]] = EnvironmentUtils.FLAG
        return flag_location

    def generate_enemies(self, n_enemies):
        enemies_locations = []
        for _ in range(n_enemies):
            x, y = self.generate_envir_resource()
            enemies_locations.append((x, y))
            self.board[x, y] = EnvironmentUtils.ENEMY
        return enemies_locations

    def generate_gold(self, n_gold):
        gold_locations = []
        for _ in range(n_gold):
            x, y = self.generate_envir_resource()
            gold_locations.append((x, y))
            self.board[x, y] = EnvironmentUtils.GOLD
        return gold_locations

    def generate_bombs(self, n_bombs):
        bombs_locations = []
        for _ in range(n_bombs):
            x, y = self.generate_envir_resource()
            bombs_locations.append((x, y))
            self.board[x, y] = EnvironmentUtils.BOMB
        return bombs_locations

    def generate_envir_resource(self):
        while True:
            x_coord = np.random.randint(2, self.dimension - 2)
            y_coord = np.random.randint(1, self.dimension - 1)
            if self.board[x_coord, y_coord] == EnvironmentUtils.FREE_LOCATION:
                break
        return x_coord, y_coord

    def display_board(self):
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