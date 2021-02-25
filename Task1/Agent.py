import numpy as np
import enum
from Policy import Policy

class AgentUtils(enum.IntEnum):
    MAX_LIFE_POINTS = 20

class Agent:

    def __init__(self, start_location):
        self.life_points = AgentUtils.MAX_LIFE_POINTS
        self.current_location = start_location
        self.policy = Policy()
