import numpy as np
from Agent import Agent
import enum


class GameDifficulty(enum.Enum):
    """
    This class specifies some parameters of the game.
    """
    EASY = 0
    MEDIUM = 1
    HARD = 2
    EXTREME = 3


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
    ROAD = 5
    SAFE = 6


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
    ROADS_EASY = {'prob_road': [0.4, 0.6], 'width': 2, 'traffic': [0.6, 0.4]}
    ROADS_MEDIUM = {'prob_road': [0.5, 0.5], 'width': 2, 'traffic': [0.6, 0.4]}
    ROADS_HARD = {'prob_road': [0.4, 0.6], 'width': 3, 'traffic': [0.65, 0.35]}
    ROADS_EXTREME = {'prob_road': [0.5, 0.5], 'width': 4, 'traffic': [0.65, 0.35]}

    def __init__(self, N=20, difficulty=GameDifficulty.EASY):
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
        self.agent = self.init_agent()

        self.display_board()

    def init_agent(self):
        """
        This function first search for a valid location for the agent, then instantiates the agent
        with the valid location. Note that the agent at the start of the game can only spawn in
        one of the possible columns of the the lower row of the board (excluding the walls).
        :return agent: The agent of the environment.
        """
        agent_location = (self.dimension-2, np.random.randint(1, self.dimension-1))
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
        prob_generate_road = difficulty_settings['prob_road']
        road_width = difficulty_settings['width']
        # the start index at which the road can be built.
        # it start at dimension-3 because the dim-1 row is reserved for the wall ('X'),
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
        display_helper = {0: '.',
                          1: 'X',
                          2: 'A',
                          3: 'C',
                          4: 'T',
                          5: 'R',
                          6: 'S'}

        print('Difficulty: {}'.format(self.difficulty.name))
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


env = Environment(25, difficulty=GameDifficulty.EXTREME)