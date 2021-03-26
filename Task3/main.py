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


def init_memory(batch_size):
    states = []
    actions = []
    rewards = []
    values = []
    is_done = []
    log_probs = []
    memory = Memory(batch_size, states, actions, rewards, values, is_done, log_probs)
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
    for e in range(n_epochs):
        # randomly generate batches
        batches = memory.create_batch(T)
        states, actions, rewards, values, is_done, log_probs = memory.convert_np_arrays()
        advantages = []
        for t in range(T):
            delta_t = 0
            # default value for Generalized Advantage Estimation
            # GAE is used to 'average' the return which due to
            # stochasticity can result in high variance estimator.
            gae = 1
            # this loop computes all the delta_t
            for i in range(t, T-1):
                delta_t += gae * (rewards[i] + gamma * values[t + 1] * (1-int(is_done[t])) - values[t])
                # update GAE value
                gae *= gamma * labda
            advantages.append(delta_t)
        advantages = torch.tensor(np.array(advantages))

        # for each batch compute the values to update the network
        for batch in batches:
            batch_states = torch.tensor(states[batch], dtype=torch.float).to(device)
            batch_vals = torch.tensor(values[batch], dtype=torch.float).to(device)
            batch_old_probs = torch.tensor(log_probs[batch], dtype=torch.float).to(device)
            batch_actions = torch.tensor(actions[batch]).to(device)
            # feedforward pass to the actor net to get the prob distr. of
            # each action per batch
            act_prob = actor(batch_states)
            act_prob = Categorical(act_prob)
            # feedforward pass to the critic net to get the values per batch
            values_ = critic(batch_states).squeeze()
            # compute new prob log.
            new_prob = act_prob.log_prob(batch_actions)
            # in order to get the correct ratio the exp() operator must be applied
            # because of the log value.
            ratio = new_prob.exp() / batch_old_probs.exp()
            # represents the first term in the L_CLIP equation
            clip = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            # represents the second term in the L_CLIP equation
            clip = clip * advantages[batch]
            # multiply ratio and advantages to get the 'weighted probs'
            weighted_prob = advantages[batch] * ratio
            # compute the L_CLIP function. Because the
            # gradient ascent is applied, we have to change sign
            # take the mean because it's an expectation. This is
            # the actor loss
            l_clip = -torch.min(weighted_prob, clip).mean()
            # compute returns: A_t = returns - values --> returns = A_t + values
            returns = advantages[batch] + batch_vals
            # compute the L_VF, which is the critic loss
            l_vf = (returns - values_)**2
            l_vf = torch.mean(l_vf)
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
n_epochs = 4
gamma = 0.99
clip_ratio = 0.2
ldba = 0.97
c_1 = 0.5
games = 10000
batch_size = 5
fc_size = 64
T = 20
step = 0


# create the environment from gym
env = gym.make('CartPole-v1')
observation_space = env.observation_space.shape[-1]
action_space = env.action_space.n
actor = ActorModel(observation_space, action_space, fc_size)
critic = CriticModel(observation_space, fc_size)
memory = init_memory(batch_size)
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
    print('{} game completed, final score: {}, average score {}, training_phase {}'.format(g+1, running_score, np.mean(running_history), train_counter))

# for i in range(games):
#     # return the initial state of the environment --> [car_position, velocity]
#     state = env.reset()
#     done = False
#     running_reward = 0
#     running_history = []
#     steps = 0
#     while not done:
#         # sample an action from the ActorModel
#         # returns the index of the taken action
#         env.render()
#         action, value, log_prob = choose_action(state)
#         # take a step into the world
#         new_state, reward, done, info = env.step(action)
#         # accumulare score
#         running_reward += reward
#         running_history.append(running_reward)
#         # increment step counter
#         steps += 1
#         # store new information into the memory
#         memory.push(new_state, reward, action, value, done, log_prob)
#         # once after T steps, update the network's parameter
#         if steps % T == 0:
#             update_policy(n_epochs, T, gamma, ldba, eps, c_1)
#         state = new_state
#     print('Episode {}, Average Score: {}'.format(i+1, np.mean(running_history)))
#     score_history.append(running_reward)
# env.close()