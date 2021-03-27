import gym
from Task3.model import ActorModel, CriticModel, Memory
import torch
import os
import numpy as np
from torch.distributions.categorical import Categorical

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

### USEFUL METHODS


def init_memory():
    states = []
    actions = []
    rewards = []
    values = []
    is_done = []
    log_probs = []
    memory = Memory(states, actions, rewards, values, is_done, log_probs)
    return memory


def get_action(act_prob):
    result = []
    for idx, probs in enumerate(act_prob):
        action_idx = np.argmax(np.random.multinomial(1, np.array(probs.data)))
        new_prob = probs[action_idx]
        result.append(new_prob.item())
    result = torch.tensor(np.array(result))
    return result


def ppo(n_epochs, gamma, clip_ratio, labda, c_1, T):
    """
    This function perform an update of the policy
    based on the trajectory stored in memory. The algorithm
    applied is the Actor-Critic style described by the paper
    https://arxiv.org/pdf/1707.06347.pdf
    :return:
    """
    # convert properly all the data structures
    states, actions, rewards, values, is_done, log_probs = memory.convert_np_arrays()
    states = torch.tensor(states, dtype=torch.float).to(device)
    values = torch.tensor(values, dtype=torch.float).to(device)
    log_probs = torch.tensor(log_probs, dtype=torch.float).to(device)
    actions = torch.tensor(actions).to(device)

    for e in range(n_epochs):
        # randomly generate batches
        returns = []
        disc_reward = 0
        # compute discounted reward using MC method.
        for idx, rew in enumerate(reversed(rewards)):
            disc_reward = rewards[idx] + (gamma * disc_reward)
            returns.insert(0, disc_reward)

        returns = torch.tensor(returns)
        advantages = returns - values
        advantages = torch.tensor(np.array(advantages))

        # feedforward pass to the actor net to get the prob distr. of
        # each action per batch
        act_prob = actor(states)
        act_prob = Categorical(act_prob)
        # feedforward pass to the critic net to get the values per batch
        values_ = critic(states).squeeze()
        # compute new prob log.
        new_prob = act_prob.log_prob(actions)
        # in order to get the correct ratio the exp() operator must be applied
        # because of the log value.
        ratio = torch.exp(new_prob - log_probs)
        # represents the first term in the L_CLIP equation
        clip = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        # multiply ratio and advantages to get the 'weighted probs'
        weighted_prob = advantages * ratio
        # compute the L_CLIP function. Because the
        # gradient ascent is applied, we have to change sign
        # take the mean because it's an expectation. This is
        # the actor loss
        l_clip = torch.mean(-torch.min(weighted_prob, clip))
        # compute the L_VF, which is the critic loss
        l_vf = torch.mean((returns - values_)**2)
        # combine the two losses to get the final loss L_CLIP_VF
        total_loss = l_clip + (c_1 * l_vf)

        # perform backprop
        actor.optimizer.zero_grad()
        critic.optimizer.zero_grad()
        total_loss.backward()
        actor.optimizer.step()
        critic.optimizer.step()
    # at the end of the epoch clear the trajectory
    # in the memory
    memory.empty_memory()


# define some hyperparameters from the paper of PPO
n_epochs = 50
gamma = 0.99
clip_ratio = 0.2
ldba = 0.97
c_1 = 0.5
games = 10000
fc_size = 256
T = 10
step = 0


# create the environment from gym
env = gym.make('CartPole-v1')
observation_space = env.observation_space.shape[-1]
action_space = env.action_space.n
actor = ActorModel(observation_space, action_space, fc_size)
critic = CriticModel(observation_space, fc_size)
memory = init_memory()
score_history = []
## main loop ##

for g in range(games):
    # get initial state
    state = env.reset()
    # setup useful variables
    running_score = 0
    running_history = []
    score_history = []
    train_counter = 0
    done = False
    # iterate through the game util it's finished
    while not done:
        # convert state to a tensor.
        # Add a dimension to support the batch for later steps
        env.render()
        state = torch.tensor(state, dtype=torch.float32)
        # return the actions prod. distribution
        act_distr = actor(state).squeeze(0)
        act_distr = Categorical(act_distr)
        # action = np.argmax(np.random.multinomial(1, np.array(act_distr.data)))
        action = act_distr.sample()
        # use the distribribution to sample from a multinomial
        # distribution. This will ensure that the agent always acts
        # in a stochastic manner according to the latest version
        # of the stochastic policy
        # take a step in the environment
        new_state, reward, done, info = env.step(action.item())
        step += 1
        # update running score
        running_score += reward
        # compute log_probs of the prob distribution
        # log_prob = np.log(act_distr.data[action]).item()
        log_prob = torch.squeeze(act_distr.log_prob(action)).item()
        # compute the value for this game iteration
        value = critic(state).data.item()
        # store the info in the memory
        memory.push(new_state, action, reward, value, done, log_prob)
        if step % T == 0:
            ppo(n_epochs, gamma, clip_ratio, ldba, c_1, T)
            train_counter += 1
        # update current state to new state
        state = new_state
        running_history.append(running_score)

    # game finished, train phase
    # perform ppo algorithm
    print('{} game completed, final score: {}, average score {}, training_phase {}'.format(g+1,
                                                                                           running_score,
                                                                                           np.mean(running_history),
                                                                                           train_counter))