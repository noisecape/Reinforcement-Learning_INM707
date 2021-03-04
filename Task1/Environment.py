import numpy as np
from Agent import Agent
from Car import Car
import enum
from collections import namedtuple
from E_Greedy import E_Greedy


class Rewards(enum.IntEnum):
    """
    This class defines the amount of reward the agent can receive
    depending on the type of location.
    """
    FREE = -1
    WALL = -5
    CAR = -100
    SAFE = 100


class GameDifficulty(enum.Enum):
    """
    This class specifies some parameters of the game.
    """
    EASY = 0
    MEDIUM = 1
    HARD = 2
    EXTREME = 3


class Colors:
    AGENT_BLUE = '\033[94m'
    SAFE_GREEN = '\033[92m'
    CAR_RED = '\033[91m'
    ENDC = '\033[0m'


class EnvironmentUtils(enum.IntEnum):
    """
    This class represents a series of utils used to display
    the board on the console.
    """
    FREE_LOCATION = 0
    WALL = 1
    AGENT = 2
    CAR = 3
    ROAD = 4
    SAFE = 5


class Environment:
    """
    This class represents the environment of the game.
    It is responsible for the creation of the starting environment of the game. The class also
    handles the actions taken by the agent, updating the board accordingly.
    """

    # These variables are used to randomly generate road sections.
    # The first element of the dictionary is a list
    # which represents the probability to generate or not a road section.
    # At index '0' there is the probability of generating a road section. At index 1 there is the
    # probability of not creating it.
    # The second element represents how wide the road section is. The harder the wider.
    ROADS_EASY = {'prob_road': [0.5, 0.5], 'width': 2, 'traffic': [0.85, 0.15]}
    ROADS_MEDIUM = {'prob_road': [0.5, 0.5], 'width': 2, 'traffic': [0.85, 0.15]}
    ROADS_HARD = {'prob_road': [0.5, 0.5], 'width': 3, 'traffic': [0.8, 0.2]}
    ROADS_EXTREME = {'prob_road': [0.5, 0.5], 'width': 4, 'traffic': [0.8, 0.2]}

    # Defines the possible actions
    Action = namedtuple('Action', ['id', 'name', 'idx_i', 'idx_j'])
    up = Action(0, 'up', -1, 0)
    left = Action(1, 'left', 0, -1)
    right = Action(2, 'right', 0, 1)
    idx_to_action = {}
    for action in [up, left, right]:
        idx_to_action[action.name] = action

    def __init__(self, policy, N=20, difficulty=GameDifficulty.EASY):
        """
        Inits the environment of the board.
        :param N : The dimension of the board. It should be a square board of (N,N). Default = 20
        :param difficulty: The diffuculty of the game. It can have a value betwen 1 to 3. The higher the difficulty,
        the more the number of roads to cross and cars to avoid on the grid world.
        """
        self.dimension = N
        self.difficulty = difficulty
        # generate board with walls
        self.board = self.init_board()
        # generate road sections
        self.generate_road_sections()
        # generate safe section
        self.generate_safe_section()
        # generate the agent
        self.agent = self.init_agent(policy)
        # generate cars
        self.cars = self.generate_cars()

        # defines the variables for the Q-Learning algorithm. The each state in this case will be
        # mapped by using the coordinates (row, column) in the board.
        self.q_values = np.zeros((self.dimension, self.dimension, 3))  # the agent cannot move backward
        self.reward_matrix = self.init_reward_matrix()

        self.display_board()

    def init_reward_matrix(self):
        reward_matrix = np.ones((self.dimension, self.dimension)) * -1
        for i in range(self.dimension):
            for j in range(self.dimension):
                if self.board[i][j] == EnvironmentUtils.WALL:
                    reward_matrix[i][j] = Rewards.WALL
                elif self.board[i][j] == EnvironmentUtils.CAR:
                    reward_matrix[i][j] = Rewards.CAR
                elif self.board[i][j] == EnvironmentUtils.SAFE:
                    reward_matrix[i][j] = Rewards.SAFE
        return reward_matrix

    def generate_cars(self):
        cars = []  # the list of all the cars in the board
        possible_locations = []  # list of all the possible locations where cars can be generated
        for row in range(self.dimension):
            if self.board[row][1] == EnvironmentUtils.ROAD:
                possible_locations.append((row, 1))
        for spawn_row, spawn_column in possible_locations:
            gen_car_prob = self.get_traffic_probability()
            for column in range(1, self.dimension-1):  # iterate through the columns and randomly generate cars
                prob_index = np.argmax(np.random.multinomial(1, gen_car_prob))
                if prob_index == 1:  # generate car
                    car = Car((spawn_row, column))
                    cars.append(car)
                    self.board[spawn_row][column] = EnvironmentUtils.CAR
        return cars

    def get_traffic_probability(self):
        if self.difficulty == GameDifficulty.EASY:

            return Environment.ROADS_EASY['traffic']

        elif self.difficulty == GameDifficulty.MEDIUM:

            return Environment.ROADS_MEDIUM['traffic']

        elif self.difficulty == GameDifficulty.HARD:

            return Environment.ROADS_HARD['traffic']

        else:

            return Environment.ROADS_EXTREME['traffic']

    def init_agent(self, policy):
        """
        This function first search for a valid location for the agent, then instantiates the agent
        with the valid location. Note that the agent at the start of the game can only spawn in
        one of the possible columns of the the lower row of the board (excluding the walls).
        :return agent: The agent of the environment.
        """
        agent_location = (self.dimension-2, np.random.randint(1, self.dimension-1))
        agent = Agent(agent_location, policy)
        self.board[agent_location[0], agent_location[1]] = EnvironmentUtils.AGENT
        return agent

    def init_board(self):
        """
        This function set up the walls for the board.
        :return board: The board of dimension (N,N) with walls.
        """
        board = np.zeros((self.dimension, self.dimension))
        board[:1, :] = EnvironmentUtils.WALL
        board[:, :1] = EnvironmentUtils.WALL
        board[-1, :] = EnvironmentUtils.WALL
        board[:, -1] = EnvironmentUtils.WALL
        return board

    def generate_safe_section(self):
        self.board[1:2, 1:-1] = EnvironmentUtils.SAFE

    def generate_road_sections(self):
        """
        Generates the roads on the grid and updates the board accordingly.
        Depending on the difficulty
        """
        if self.difficulty == GameDifficulty.EASY:

            self.build_road_section(Environment.ROADS_EASY)

        elif self.difficulty == GameDifficulty.MEDIUM:

            self.build_road_section(Environment.ROADS_MEDIUM)

        elif self.difficulty == GameDifficulty.HARD:

            self.build_road_section(Environment.ROADS_HARD)

        else:

            self.build_road_section(Environment.ROADS_EXTREME)

    def build_road_section(self, difficulty_settings):
        np.random.seed(47)  # to replicate the same experiment across several epochs
        prob_generate_road = difficulty_settings['prob_road']
        road_width = difficulty_settings['width']
        # the start index at which the road can be built.
        # it starts at dimension-3 because the dim-1 row is reserved for the wall ('X'),
        # the dimension-2 row is reserved for the spawning of the frog.
        board_index = self.dimension-3
        while board_index > 2:
            if board_index % 2 == 0 and self.board[board_index+1, 1] != EnvironmentUtils.ROAD:
                prob_index = np.argmax(np.random.multinomial(1, prob_generate_road))
                if prob_index == 0:  # generate road section
                    for _ in range(road_width):
                        self.board[board_index, 1:-1] = EnvironmentUtils.ROAD
                        board_index -= 1
            board_index -= 1

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
        display_helper = {0: ('.', Colors.ENDC),
                          1: ('X', Colors.ENDC),
                          2: ('A', Colors.AGENT_BLUE),
                          3: ('C', Colors.CAR_RED),
                          4: ('R', Colors.ENDC),
                          5: ('S', Colors.SAFE_GREEN)}

        print('Difficulty: {}'.format(self.difficulty.name))
        for row in range(self.dimension):
            for column in range(self.dimension):
                print(display_helper[self.board[row, column]][1] + display_helper[self.board[row, column]][0], end=' ')
            print()

    def reset(self):
        self.board = self.init_board()
        self.agent = self.init_agent()


    def step(self):
        """
        This function defines the loop of the game and it's called at every time steps.
        :return:
        """
        for car in self.cars:
            new_x, new_y = car.drive()
            if self.board[new_x][new_y] == EnvironmentUtils.WALL: # if the car bumbs into the wall, respown it
                resp_x, resp_y = car.respawn_car()
                self.board[resp_x, resp_y] = EnvironmentUtils.CAR
            else:
                self.board[new_x][new_y] = EnvironmentUtils.CAR
            # now that the car took one step, update the reward matrix

            # check if there's a car behind the current one.
            # If that's the case, don't update the reward matrix in the previous location
            if self.reward_matrix[new_x][new_y-1] != Rewards.CAR:
                self.reward_matrix[new_x][new_y-1] = Rewards.FREE
            self.reward_matrix[new_x][new_y] = Rewards.CAR

        # the agent take an action randomly.
        self.agent.jump(self.q_values)
        # check if is a legit move
        # update board and reward matrix


policy = E_Greedy(0.95)
env = Environment(policy, 12, difficulty=GameDifficulty.EASY)
env.step()
print(env.reward_matrix)