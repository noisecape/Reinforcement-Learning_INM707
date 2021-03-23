import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple

Trajectory = namedtuple('Trajectory', 'states, actions, rewards, done_mask, values, log_prob')


class Memory:

    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.done_mask = []
        self.values = []
        self.log_prob = []
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
        batches = [indices[i:i+self.batch_size] for i in index_start]
        return batches

    def store_information(self):
        pass

    def empty_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.done_mask = []
        self.values = []
        self.log_prob = []

memory = Memory(10)
memory.create_batch(19)

class ActorModel(nn.Module):

    def __init__(self):
        pass


class CriticModel(nn.Module):

    def __init__(self):
        pass

class PPOAgent:

    def __init__(self):
        pass