import gym
from Task3.model import ActorModel, CriticModel, Memory, PPO
import torch
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

# define some hyperparameters from the paper of PPO
n_epochs = 10
gamma = 0.97
clip_ratio = 0.15
c_1 = 0.5
games = 1000
fc_size = 128
T = 128
step = 0
wins = []
losses = []


# create the environment from gym
env = gym.make('CartPole-v1')
observation_space = env.observation_space.shape[-1]
action_space = env.action_space.n
# create the model
actor = ActorModel(observation_space, action_space, fc_size)
critic = CriticModel(observation_space, fc_size)
memory = Memory()
ppo = PPO(memory, actor, critic)

running_history = []

## main loop ##

for g in range(games):
    # get initial state
    state = env.reset()
    # setup useful variables
    running_score = 0
    train_counter = 0
    done = False
    # iterate through the game util it's finished
    while not done:
        # env.render()
        act_distr = actor(torch.tensor(state, dtype=torch.float))
        action = np.argmax(np.random.multinomial(1, np.array(act_distr.data)))
        new_state, reward, done, info = env.step(action)
        step += 1
        # update running score
        running_score += reward
        # compute log_probs of the prob distribution
        log_prob = np.log(act_distr.data[action]).item()
        # compute the value for this game iteration
        value = critic(torch.tensor(state, dtype=torch.float)).data.item()
        # store the info in the memory
        memory.push(state, action, reward, value, done, log_prob)
        # update policy once every T steps
        if step % T == 0:
            ppo(n_epochs, gamma, clip_ratio, c_1)
            train_counter += 1
        # update current state to new state
        state = new_state
        running_history.append(running_score)
        # by documentation, the problem is solved
        # if the average reward is greater than 195 over the last 100 trials
        # let't check this
        if step % 100 == 0:
            avg_reward = np.mean(running_history[-100:])
            if avg_reward >= 195:
                wins.append(avg_reward)
            else:
                losses.append(avg_reward)
            print('Game completed: [{}]/[{}],'
                  ' Average reward of last 100 trials: {}'.format(g+1,
                                                                  games,
                                                                  avg_reward))
