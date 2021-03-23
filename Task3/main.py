import gym
from Task3.model import PPOAgent

# define some hyperparameters from the paper of PPO
T = 30
games = 1000
scores = []
avg_score = 0

# create the environment from gym
env = gym.make('MountainCar-v0')
agent = PPOAgent()
score_history = []


for i in range(games):
    # return the initial state of the environment
    # this will be a uniform random value between [-0.6, -0.4] and
    # the velocity of the car will be 0.
    obs = env.reset()
    done = False
    running_reward = 0
    steps = 0
    while not done:
        # sample an action from the ActorModel
        action = agent.choose_action(obs)
        # take a step into the world
        obs_, reward, done, info = env.step(action)
        # accumulare score
        running_reward += reward
        # once after T steps, update the network's parameter
        if steps % T == 0:
            agent.update_policy()
        obs = obs_
        score_history.append(running_reward)

    # print stats of the episode