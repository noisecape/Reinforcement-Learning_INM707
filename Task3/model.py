import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import torch.optim as opt
from torch.distributions.categorical import Categorical

Trajectory = namedtuple('Trajectory', 'states, actions, rewards, values, log_probs')

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)


class Memory:

    def __init__(self, batch_size):
        # init data structures to hold trajectories info
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.trajectory = Trajectory(self.states, self.actions, self.rewards, self.values, self.log_probs)

        self.batch_size = batch_size

    def create_batch(self, t_len):
        """
        This function creates a batch of dimension 'batch_size' that will be
        used during the training of the agent. The batch is created by
        shuffling a list of indices and extracting 'len_trajectory/batch_size' batches.
        :param batch_dim: the dimension of each batch
        :param t_len: the length of a single trajectory.
        :return:
        """
        indices = np.arange(0, t_len)
        np.random.shuffle(indices)
        index_start = np.arange(0, t_len, self.batch_size)
        # creates the batches
        batches = [indices[i:i + self.batch_size] for i in index_start]
        return batches

    def push(self, states, actions, rewards, values, log_probs):
        self.trajectory.states.append(states)
        self.trajectory.actions.append(actions)
        self.trajectory.rewards.append(rewards)
        self.trajectory.values.append(values)
        self.trajectory.log_probs.append(log_probs)

    def empty_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []


class ActorModel(nn.Module):

    def __init__(self, obs_size, action_size, fc_size=64):
        super(ActorModel, self).__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        # implement the neural network that represents the policy.
        # following the paper's implementation: https://arxiv.org/pdf/1707.06347.pdf
        self.model = nn.Sequential(nn.Linear(obs_size, fc_size),
                                   nn.Tanh(),
                                   nn.Linear(fc_size, fc_size),
                                   nn.Tanh(),
                                   nn.Linear(fc_size, action_size),
                                   nn.Softmax(dim=-1)
                                   ).to(device)
        optim_params = {'lr': 10e-5, 'weight_decay': 10e-3}
        self.optimizer = opt.Adam(self.model.parameters(), **optim_params)

    def forward(self, x):
        # compute probabilities for each action
        act_prob = self.model(x)
        return act_prob


class CriticModel(nn.Module):

    def __init__(self, obs_size, fc_size=64):
        super(CriticModel, self).__init__()

        self.model = nn.Sequential(nn.Linear(obs_size, fc_size),
                                   nn.Tanh(),
                                   nn.Linear(fc_size, fc_size),
                                   nn.Tanh(),
                                   nn.Linear(fc_size, 1)
                                   ).to(device)

        optim_params = {'lr': 10e-5, 'weight_decay': 10e-3}
        self.optimizer = opt.Adam(self.model.parameters(), **optim_params)

    def forward(self, x):
        value = self.model(x)
        return value.squeeze(0).item()


class PPOAgent:

    def __init__(self, batch_size, obs_space, act_space, fc_size,
                 lr=10e-4, n_epoch=10, gamma=0.99, lbda=0.95, T=30, eps=0.2):
        # store hyperparameters
        self.lr = lr
        self.n_epoch = n_epoch
        self.gamma = gamma
        self.lbda = lbda
        self.T = T
        self.eps = eps

        # initialize the model classes
        self.memory = Memory(batch_size)
        self.actor = ActorModel(obs_space, act_space, fc_size)
        self.critic = CriticModel(obs_space, fc_size)

    def get_log_prob(self, distribution):
        return np.log(distribution.detach().numpy())

    def choose_action(self, state):
        # convert ndarray to a tensor
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # compute the action
        act_prob = self.actor(state).squeeze(0)
        # sample from a multinomial distribution and get the action's index
        action = np.argmax(np.random.multinomial(1, act_prob.detach().numpy()))
        # compute the value
        value = self.critic(state)
        # compute log probability
        log_prob = self.get_log_prob(act_prob)

        return action, value, log_prob

    def store_info(self, state, reward, action, value, log_prob):
        self.memory.push(state, action, reward, value, log_prob)

    def update_policy(self):
        """
        This function perform an update of the policy
        based on the trajectory stored in memory. The algorithm
        applied is the Actor-Critic style described by the paper
        https://arxiv.org/pdf/1707.06347.pdf
        :return:
        """
        for _ in self.n_epoch:
            # randomly generate batches
            batches = self.memory.create_batch(self.T)

            advantages = []
            for t in range(self.T-1):
                delta_t = 0
                # default value for Generalized Advantage Estimation
                # GAE is used to 'average' the return which due to
                # stochasticity can result in high variance estimator.
                gae = 1
                # this loop computes all the delta_t
                for i in range(t, self.T - 1):
                    delta_t += (self.memory.trajectory.rewards[i] + \
                               (self.gamma * self.memory.trajectory.values[t + 1]) - \
                               self.memory.trajectory.values[t]) * gae
                    # update GAE value
                    gae *= self.gamma * self.lbda
                advantages.append(delta_t)

            # for each batch compute the values to update the network
            for batch in batches:
                batch_states = torch.tensor(self.memory.states[batch], dtype=torch.float32).to(device)
                batch_vals = torch.tensor(self.memory.states[batch], dtype=torch.float32).to(device)
                batch_old_probs = torch.tensor(self.memory.log_probs[batch], data=torch.float32).to(device)
                # feedforward pass to the actor net to get the prob distr. of
                # each action per batch
                act_prob = self.actor(batch_states)
                # feedforward pass to the critic net to get the values per batch
                values = self.critic(batch_states)
                # compute new prob log.
                act_prob = self.get_log_prob(act_prob)
                # in order to get the correct ratio the exp() operator must be applied
                # because of the log value.
                ratio = act_prob.exp() / batch_old_probs.exp()
                # represents the first term in the L_CLIP equation
                ratio *= advantages[batch]
                clip = torch.clamp(ratio, 1-self.eps, 1+self.eps)
                # represents the second term in the L_CLIP equation
                clip = clip * advantages[batch]
                # compute the L_CLIP function. Because the
                # gradient ascent is applied, we have to change sign
                # take the mean because it's an expectation.
                l_clip = -torch.mean(torch.min(ratio, clip))
                # final reward is the sum
                # backprop and update the model


