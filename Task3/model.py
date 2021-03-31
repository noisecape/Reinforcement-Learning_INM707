import torch
import torch.nn as nn
import torch.optim as opt

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)


class Memory:
    """
    This class implements a mechanism to store required information
    of a trajectory to update the Policy using the PPO algorithm.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.is_done = []
        self.log_probs = []

    def push(self, states, actions, rewards, values, is_done, log_probs):
        """
        This class push the information into memory
        :param states: the states of the environment
        :param actions: the actions taken by the agent
        :param rewards: the rewards gained during each trajectory
        :param values: the values associated with each state
        :param is_done: a boolean vector used to mark games that ended.
        :param log_probs: the log probabilities of each action in a trajectory
        :return:
        """
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.values.append(values)
        self.is_done.append(is_done)
        self.log_probs.append(log_probs)

    def empty_memory(self):
        """
        This function emptify the memory.
        :return:
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.is_done = []
        self.log_probs = []


class ActorModel(nn.Module):
    """
    This class implements the Actor in the PPO algorithm.
    Its main functionality is to implement the Neural Network
    which takes in input the state of the environment and outputs
    the probability distribution associated to each action.
    """
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
        optim_params = {'lr': 0.0025}
        self.optimizer = opt.Adam(self.parameters(), **optim_params)

    def forward(self, x):
        """
        This function takes in input the states of
        the environment and forward pass it through
        the Neural Network to output the action probabilities.
        :param x: state of the environment
        :return: the probability distribution associated to each possible action.
        """
        act_prob = self.model(x)
        return act_prob


class CriticModel(nn.Module):
    """
    This class implements the Critic model of the PPO algorithm
    This model is mainly composed by the Neural Network which takes
    in input the state of the environment and outputs the V values.
    """
    def __init__(self, obs_size, fc_size=64):
        super(CriticModel, self).__init__()
        self.model = nn.Sequential(nn.Linear(obs_size, fc_size),
                                   nn.ReLU(),
                                   nn.Linear(fc_size, fc_size),
                                   nn.ReLU(),
                                   nn.Linear(fc_size, 1)
                                   ).to(device)

        optim_params = {'lr': 0.0025}
        self.optimizer = opt.Adam(self.parameters(), **optim_params)

    def forward(self, x):
        """
        This function forward pass the input through the critic
        Neural Network in order to calculate the V values.
        :param x: the states of the environment.
        :return: the V values
        """
        value = self.model(x)
        return value


class PPO(nn.Module):
    """
    This class acts as a container for the Actor model and the Critic model.
    It wraps their functionalities and merge the two components to implement
    the PPO algorithm and successfully update the agent's policy.
    """
    def __init__(self, memory, actor, critic):
        super(PPO, self).__init__()
        self.memory = memory
        self.actor = actor
        self.critic = critic
        self.criterion = nn.MSELoss()

    def forward(self, n_epochs, gamma, clip_ratio, c_1):
        """
        This function perform an update of the policy
        based on the trajectory stored in memory. The algorithm
        applied is the Actor-Critic style described by the paper
        https://arxiv.org/pdf/1707.06347.pdf
        :return:
        """
        returns = []
        disc_reward = 0

        # use the MC method to estimate the return values.
        for reward, is_terminal in (zip(reversed(self.memory.rewards), reversed(self.memory.is_done))):
            disc_reward = reward + (gamma * disc_reward) * (1 - (int(is_terminal)))
            returns.insert(0, disc_reward)

        returns = torch.tensor(returns, dtype=torch.float)

        for e in range(n_epochs):
            # convert to tensors all the stored information
            # to forward pass them through the models.
            states = torch.tensor(self.memory.states, dtype=torch.float).to(device)
            log_probs = torch.tensor(self.memory.log_probs, dtype=torch.float).to(device)
            actions = torch.tensor(self.memory.actions).to(device)
            selected_actions = torch.zeros(actions.shape[0])
            # compute action probability distribution
            act_distr = self.actor(states)
            # select the actions using the action ids stored in the trajectory
            for idx, act in enumerate(actions):
                selected_actions[idx] = act_distr[idx][act]
            # feedforward pass to the critic net to get the values per batch
            values_ = self.critic(states).squeeze()
            # compute advantages
            advantages = returns - values_
            # compute new prob log.
            new_prob = torch.log(selected_actions)
            # in order to get the correct ratio the exp() operator must be applied
            # because of the log value.
            ratio = torch.exp(new_prob - log_probs)
            # represents the first term in the L_CLIP equation
            clip = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            # multiply ratio and advantages to get the 'weighted probs'
            weighted_prob = advantages * ratio
            # Compute the L_CLIP function. Gradient ascent is applied,
            # so we have to change sign. Also take the mean because it's an expectation.
            # The actor loss
            l_clip = -torch.min(weighted_prob, clip).mean()
            # compute the L_VF, which is the critic loss
            l_vf = self.criterion(returns, values_)
            # combine the two losses to get the final loss L_CLIP_VF
            total_loss = l_clip + (c_1 * l_vf)

            # perform backprop
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            total_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()
        # at the end of the epoch clear the trajectory
        # in the memory
        self.memory.empty_memory()
