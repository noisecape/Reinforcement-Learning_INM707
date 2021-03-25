import gym
from Task3.model import ActorModel, CriticModel, Memory
import Task3.model as tsk3
import torch
import os
import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# define some hyperparameters from the paper of PPO
T = 100
n_epochs = 20
gamma =0.99
eps = 0.2
ldba = 0.95
c_1 = 0.5
games = 1000
avg_score = 0
batch_size = 64
fc_size = 64

# create the environment from gym
env = gym.make('MountainCar-v0')
observation_space = env.observation_space.shape[-1]
action_space = env.action_space.n
actor = ActorModel(observation_space, action_space, fc_size)
critic = CriticModel(observation_space, fc_size)
memory = Memory(batch_size)
score_history = []

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

### USEFUL METHODS

def get_log_prob(distribution):
    return np.log(distribution.detach().numpy())


def choose_action(state):
    # convert ndarray to a tensor
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    # compute the action
    act_prob = actor(state).squeeze(0)
    # sample from a multinomial distribution and get the action's index
    action = np.argmax(np.random.multinomial(1, act_prob.detach().numpy()))
    # compute the value
    value = critic(state).item()
    # compute log probability
    log_prob = get_log_prob(act_prob[action])

    return action, value, log_prob


def get_best_action(act_prob, actions_idx):
    result = []
    for idx, probs in enumerate(act_prob):
        new_prob = probs[actions_idx[idx]].item()
        result.append(new_prob)
    result = torch.tensor(np.array(result))
    return result

def update_policy(n_epochs, T, gamma, lbda, eps, c_1):
    """
    This function perform an update of the policy
    based on the trajectory stored in memory. The algorithm
    applied is the Actor-Critic style described by the paper
    https://arxiv.org/pdf/1707.06347.pdf
    :return:
    """
    states, actions, rewards, values, log_probs = memory.convert_np_arrays()
    for _ in range(n_epochs):
        # randomly generate batches
        batches = memory.create_batch(T)

        advantages = []
        for t in range(T):
            delta_t = 0
            # default value for Generalized Advantage Estimation
            # GAE is used to 'average' the return which due to
            # stochasticity can result in high variance estimator.
            gae = 1
            # this loop computes all the delta_t
            for i in range(t, T-1):
                delta_t += (rewards[i] + \
                            (gamma * values[t + 1]) - values[t]) * gae
                # update GAE value
                gae *= gamma * lbda
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
            act_prob = get_best_action(act_prob, batch_actions)
            # feedforward pass to the critic net to get the values per batch
            values_ = critic(batch_states)
            # compute new prob log.
            act_prob = torch.tensor(get_log_prob(act_prob))
            # in order to get the correct ratio the exp() operator must be applied
            # because of the log value.
            ratio = act_prob.exp() / batch_old_probs.exp()
            # represents the first term in the L_CLIP equation
            ratio *= advantages[batch]
            clip = torch.clamp(ratio, 1-eps, 1+eps)
            # represents the second term in the L_CLIP equation
            clip = clip * advantages[batch]
            # compute the L_CLIP function. Because the
            # gradient ascent is applied, we have to change sign
            # take the mean because it's an expectation. This is
            # the actor loss
            l_clip = -torch.mean(torch.min(ratio, clip))
            # compute returns: A_t = returns - values --> returns = A_t + values
            returns = advantages[batch] + batch_vals
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


for i in range(games):
    # return the initial state of the environment --> [car_position, velocity]
    state = env.reset()
    done = False
    running_reward = 0
    steps = 0
    while not done:
        # sample an action from the ActorModel
        # returns the index of the taken action
        action, value, log_prob = choose_action(state)
        # take a step into the world
        new_state, reward, done, info = env.step(action)
        # accumulare score
        running_reward += reward
        # increment step counter
        steps += 1
        # store new information into the memory
        memory.push(new_state, reward, action, value, log_prob)
        # once after T steps, update the network's parameter
        env.render()
        if steps % T == 0:
            update_policy(n_epochs, T, gamma, ldba, eps, c_1)
        state = new_state
    score_history.append(running_reward)
    print('Episode {}, Score: {}'.format(i+1, np.mean(score_history[-100:])))
env.close()