import enum
from collections import namedtuple
import numpy as np


class Car:

    def __init__(self, start_location):
        self.current_location = start_location

    def drive(self):
        """
        This function handles the movement of the car. At each time step the car moves
        from left to right by 1 location. Whenever it enters within a location with a wall,
        the car will be respawned to the location (row, 1).
        :return:
        """
        self.current_location = self.current_location[0], self.current_location[1]+1
        return self.current_location

    def respawn_car(self):
        """
        This function is used to respawn the car to the location (row, 1)
        :return:
        """
        self.current_location = self.current_location[0], 1
        return self.current_location


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

class Rewards(enum.IntEnum):
    """
    This class defines the amount of reward the agent can receive
    depending on the type of location.
    """
    FREE = -1
    WALL = -5
    EXIT_ROAD_SECTION = 25
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
    EXIT_ROAD = 6


class Experiment:

    def __init__(self,
                 episodes=10,
                 dimension=10,
                 difficulty=GameDifficulty.EASY,
                 epsilon=.1,
                 gamma=.9,
                 lr=.01):

        self.episodes = episodes
        self.dimension = dimension
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.difficulty = difficulty

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

    ROADS_EASY = {'prob_road': [0.4, 0.6], 'width': 1, 'traffic': [0.95, 0.05]}
    ROADS_MEDIUM = {'prob_road': [0.4, 0.6], 'width': 1, 'traffic': [0.95, 0.05]}
    ROADS_HARD = {'prob_road': [0.4, 0.6], 'width': 2, 'traffic': [0.92, 0.08]}
    ROADS_EXTREME = {'prob_road': [0.5, 0.5], 'width': 2, 'traffic': [0.85, 0.15]}

    # Defines the possible actions
    Action = namedtuple('Action', ['id', 'name', 'idx_i', 'idx_j'])
    up = Action(0, 'up', -1, 0)
    left = Action(1, 'left', 0, -1)
    right = Action(2, 'right', 0, 1)
    idx_to_action = {}
    for action in [up, left, right]:
        idx_to_action[action.id] = action

    def __init__(self, dimension=20, difficulty=GameDifficulty.EASY):
        """
        Inits the environment of the board.
        :param dimension : The dimension of the board. It should be a square board of (N,N). Default = 20
        :param difficulty: The diffuculty of the game. It can have a value betwen 1 to 3. The higher the difficulty,
        the more the number of roads to cross and cars to avoid on the grid world.
        """
        self.dimension = dimension
        self.difficulty = difficulty
        self.is_gameover = False
        # generate board with walls
        self.board = self.init_board()
        # generate road sections
        self.generate_road_sections()
        # generate safe section
        self.generate_safe_section()
        # generate the agent
        self.agent = self.init_agent()
        # generate cars
        self.cars = self.generate_cars()

        # defines the variables for the Q-Learning algorithm. The each state in this case will be
        # mapped by using the coordinates (row, column) in the board.
        self.q_values = np.zeros((self.dimension, self.dimension, 3))  # the agent cannot move backward
        self.reward_matrix = self.init_reward_matrix()
        self.final_reward = 0

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
                elif self.board[i][j] == EnvironmentUtils.FREE_LOCATION:
                    reward_matrix[i][j] = Rewards.FREE
                elif self.board[i][j] == EnvironmentUtils.EXIT_ROAD:
                    reward_matrix[i][j] = Rewards.EXIT_ROAD_SECTION
                elif self.board[i][j] == EnvironmentUtils.AGENT:
                    reward_matrix[i][j] = Rewards.FREE
                elif self.board[i][j] == EnvironmentUtils.ROAD:
                    reward_matrix[i][j] = Rewards.FREE
        return reward_matrix

    def generate_cars(self):
        cars = []  # the list of all the cars in the board
        possible_locations = []  # list of all the possible locations where cars can be generated
        for row in range(self.dimension):
            if self.board[row][1] == EnvironmentUtils.ROAD:
                possible_locations.append((row, 1))
        for spawn_row, spawn_column in possible_locations:
            gen_car_prob = self.get_traffic_probability()
            for column in range(1, self.dimension - 1):  # iterate through the columns and randomly generate cars
                prob_index = np.argmax(np.random.multinomial(1, gen_car_prob))
                if prob_index == 1:  # generate car
                    car = Car((spawn_row, column))
                    cars.append(car)
                    self.board[spawn_row][column] = EnvironmentUtils.CAR
        cars.reverse()
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

    def init_agent(self):
        """
        This function first search for a valid location for the agent, then instantiates the agent
        with the valid location. Note that the agent at the start of the game can only spawn in
        one of the possible columns of the the lower row of the board (excluding the walls).
        :return agent: The agent of the environment.
        """
        agent_location = (self.dimension - 2, np.random.randint(1, self.dimension - 1))
        agent = Agent(agent_location)
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
        # random_column = np.random.randint(1, self.dimension-1)
        self.board[1][1:-1] = EnvironmentUtils.SAFE

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
        #np.random.seed(47)  # to replicate the same experiment across several epochs
        prob_generate_road = difficulty_settings['prob_road']
        road_width = difficulty_settings['width']
        # the start index at which the road can be built.
        # build the road section starting from the bottom till the top of the board.
        # it starts at dimension-3 because the dim-1 row is reserved for the wall ('X'),
        # the dimension-2 row is reserved for the spawning of the frog.
        board_index = self.dimension - 3
        while board_index > 3:
            if board_index % 2 == 0 and self.board[board_index + 1, 1] != EnvironmentUtils.ROAD:
                prob_index = np.argmax(np.random.multinomial(1, prob_generate_road))
                if prob_index == 0:  # generate road section
                    for n_road_lane in range(road_width):
                        self.board[board_index, 1:-1] = EnvironmentUtils.ROAD
                        if n_road_lane == road_width-1:
                            board_index -= 1
                            self.board[board_index, 1:-1] = EnvironmentUtils.EXIT_ROAD
                        board_index -= 1
            else:
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
                          5: ('S', Colors.SAFE_GREEN),
                          6: ('E', Colors.SAFE_GREEN)
                          }

        print('Difficulty: {}'.format(self.difficulty.name))
        for row in range(self.dimension):
            for column in range(self.dimension):
                print(display_helper[self.board[row, column]][1] + display_helper[self.board[row, column]][0], end=' ')
            print()

    def reset(self):
        self.is_gameover = False
        # generate board with walls
        self.board = self.init_board()
        # generate road sections
        self.generate_road_sections()
        # generate safe section
        self.generate_safe_section()
        # generate the agent
        self.agent = self.init_agent()
        # generate cars
        self.cars = self.generate_cars()

        # defines the variables for the Q-Learning algorithm. The each state in this case will be
        # mapped by using the coordinates (row, column) in the board.
        self.reward_matrix = self.init_reward_matrix()
        self.final_reward = 0

    def is_road_section(self, prev_x, prev_y):
        """
        This function checks if the previous location the agent moved from is part of a road section.
        :return: True if the previous location is part of a road section. Otherwise, False
        """
        if self.board[prev_x][prev_y - 1] == EnvironmentUtils.ROAD or \
                self.board[prev_x][prev_y - 1] == EnvironmentUtils.CAR:
            return True
        if self.board[prev_x][prev_y + 1] == EnvironmentUtils.ROAD or \
                self.board[prev_x][prev_y + 1] == EnvironmentUtils.CAR:
            return True
        return False

    def step(self, idx_action):
        """
        This function moves the agent in the next state and updates the board.
        :param action_idx: the action the agent has to take.
        :return reward: the immediate reward of the step.
        """
        action = Environment.idx_to_action[idx_action]
        reward = 0
        # update cars location
        for car in self.cars:
            prev_x, prev_y = car.current_location
            new_x, new_y = car.drive()
            if self.board[new_x][new_y] == EnvironmentUtils.WALL:  # if the car bumbs into the wall, respown it
                resp_x, resp_y = car.respawn_car()
                self.board[resp_x, resp_y] = EnvironmentUtils.CAR
                self.reward_matrix[prev_x][prev_y] = Rewards.FREE
                self.reward_matrix[resp_x][resp_y] = Rewards.CAR
            elif self.board[new_x][new_y] == EnvironmentUtils.AGENT:
                # is gameover
                self.is_gameover = True
                self.board[new_x][new_y] = EnvironmentUtils.CAR
                self.reward_matrix[prev_x][prev_y] = Rewards.FREE
                self.reward_matrix[new_x][new_y] = Rewards.CAR
                reward += self.reward_matrix[new_x][new_y]
                break
            else:
                self.board[new_x][new_y] = EnvironmentUtils.CAR
                self.reward_matrix[prev_x][prev_y] = Rewards.FREE
                self.reward_matrix[new_x][new_y] = Rewards.CAR
            self.board[prev_x][prev_y] = EnvironmentUtils.ROAD
            # now that the car took one step, update the reward matrix
            # check if there's a car behind the current one.
            # If that's the case, don't update the reward matrix in the previous location
        # check if a car crossed over the agent.
        if self.is_gameover:
            self.final_reward += reward
            return reward
        # store previous agent's location
        prev_x, prev_y = self.agent.current_location
        # get the new location
        new_x, new_y = self.agent.jump(action)
        # update reward for taking the action
        reward += self.reward_matrix[new_x][new_y]
        # update the agent location within the board
        if self.board[new_x][new_y] == EnvironmentUtils.FREE_LOCATION:
            # check if the previous location is an 'exit road' section
            if self.board[prev_x][prev_y-1] == EnvironmentUtils.EXIT_ROAD or self.board[prev_x][prev_y+1] ==\
                    EnvironmentUtils.EXIT_ROAD:
                self.board[prev_x][prev_y] = EnvironmentUtils.EXIT_ROAD
            else:
                self.board[prev_x][prev_y] = EnvironmentUtils.FREE_LOCATION
            # update the new location of the agent in the board
            self.board[new_x][new_y] = EnvironmentUtils.AGENT
        elif self.board[new_x][new_y] == EnvironmentUtils.CAR:
            # update only the previous location.
            # The agent is dead so there's no more 'A' in the board. End of the episode.
            self.is_gameover = True
            self.final_reward += reward
            return reward
        elif self.board[new_x][new_y] == EnvironmentUtils.WALL:
            # restore the previous value
            self.agent.current_location = prev_x, prev_y
        elif self.board[new_x][new_y] == EnvironmentUtils.SAFE:
            self.is_gameover = True
            return reward
        elif self.board[new_x][new_y] == EnvironmentUtils.ROAD:
            # update the previous location
            if self.is_road_section(prev_x, prev_y):
                self.board[prev_x][prev_y] = EnvironmentUtils.ROAD
            else:
                self.board[prev_x][prev_y] = EnvironmentUtils.FREE_LOCATION
            self.board[new_x][new_y] = EnvironmentUtils.AGENT
        elif self.board[new_x][new_y] == EnvironmentUtils.EXIT_ROAD:
            # the previous location must be the road.
            self.board[prev_x][prev_y] = EnvironmentUtils.ROAD
            # remove the whole line of reward otherwise the agent will
            # try to move horizontally collecting all the reward (+25) after the road section
            self.board[new_x][1:-1] = EnvironmentUtils.FREE_LOCATION
            self.reward_matrix[new_x][1:-1] = Rewards.FREE
            self.board[new_x][new_y] = EnvironmentUtils.AGENT

        self.final_reward += reward
        # print('Previous agent location: {}, {}'.format(prev_x, prev_y))
        # print('New Agent location: {}, {}'.format(self.agent.current_location[0], self.agent.current_location[1]))
        # print('Current reward: {}'.format(reward))
        # print('Cumulative epoch reward: {}'.format(self.final_reward))
        # self.display_board()
        # print(self.reward_matrix)

        return reward
