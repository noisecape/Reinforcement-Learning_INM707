import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import torch.optim as opt
from torch.distributions.categorical import Categorical

Trajectory = namedtuple('Trajectory', 'states, actions, rewards, values, is_done, log_probs')

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

torch.autograd.set_detect_anomaly(True)

class Memory:

    def __init__(self, batch_size, states, actions, rewards, values, is_done, log_probs):
        # init data structures to hold trajectories info
        self.trajectory = Trajectory(states, actions, rewards, values, is_done, log_probs)
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

    def convert_np_arrays(self):
        """
        Useful function to convert the list stored in the memory as
        numpy array so that they can be easily converted to tensors.
        :return:
        """
        return np.array(self.trajectory.states), \
               np.array(self.trajectory.actions, dtype=int), \
               np.array(self.trajectory.rewards), \
               np.array(self.trajectory.values), \
               np.array(self.trajectory.is_done, dtype=int), \
               np.array(self.trajectory.log_probs)

    def push(self, states, actions, rewards, values, is_done, log_probs):
        self.trajectory.states.append(states)
        self.trajectory.actions.append(actions)
        self.trajectory.rewards.append(rewards)
        self.trajectory.values.append(values)
        self.trajectory.is_done.append(is_done)
        self.trajectory.log_probs.append(log_probs)

    def empty_memory(self):
        new_trajectory = Trajectory([], [], [], [], [], [])
        self.trajectory = new_trajectory


class ActorModel(nn.Module):

    def __init__(self, obs_size, action_size, fc_size=64):
        super(ActorModel, self).__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        # implement the neural network that represents the policy.
        # following the paper's implementation: https://arxiv.org/pdf/1707.06347.pdf
        self.model = nn.Sequential(nn.Linear(obs_size, fc_size),
                                   nn.ReLU(),
                                   nn.Linear(fc_size, fc_size),
                                   nn.ReLU(),
                                   nn.Linear(fc_size, action_size),
                                   nn.Softmax(dim=-1)
                                   ).to(device)
        optim_params = {'lr': 0.003}
        self.optimizer = opt.Adam(self.parameters(), **optim_params)

    def forward(self, x):
        act_prob = self.model(x)
        return act_prob


class CriticModel(nn.Module):

    def __init__(self, obs_size, fc_size=64):
        super(CriticModel, self).__init__()

        self.model = nn.Sequential(nn.Linear(obs_size, fc_size),
                                   nn.ReLU(),
                                   nn.Linear(fc_size, fc_size),
                                   nn.ReLU(),
                                   nn.Linear(fc_size, 1)
                                   ).to(device)

        optim_params = {'lr': 0.001}
        self.optimizer = opt.Adam(self.parameters(), **optim_params)

    def forward(self, x):
        value = self.model(x)
        return value
