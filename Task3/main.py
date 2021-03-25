import gym
from Task3.model import PPOAgent
import os
import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# define some hyperparameters from the paper of PPO
T = 20
games = 1000
avg_score = 0
batch_size = 5
fc_size = 64

# create the environment from gym
env = gym.make('CartPole-v1')
agent = PPOAgent(batch_size, env.observation_space.shape[-1], env.action_space.n, fc_size, T=T)
score_history = []


for i in range(games):
    # return the initial state of the environment --> [car_position, velocity]
    state = env.reset()
    done = False
    running_reward = 0
    steps = 0
    while not done:
        # sample an action from the ActorModel
        # returns the index of the taken action
        action, value, log_prob = agent.choose_action(state)
        # take a step into the world
        new_state, reward, done, info = env.step(action)
        # accumulare score
        running_reward += reward
        # increment step counter
        steps += 1
        # store new information into the memory
        agent.store_info(new_state, reward, action, value, log_prob)
        # once after T steps, update the network's parameter
        env.render()
        if steps % T == 0:
            agent.update_policy()
        state = new_state
    score_history.append(running_reward)
    print('Episode {}, Score: {}'.format(i+1, np.mean(score_history[-100:])))
env.close()