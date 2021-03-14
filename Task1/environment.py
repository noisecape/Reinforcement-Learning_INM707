import numpy as np
from utils import Environment, GameDifficulty
from policy import E_Greedy


policy = E_Greedy(0.9)
lr = 0.9  # learning rate
gamma = 0.9  # discount factor
env = Environment(lr, gamma, dimension=13, difficulty=GameDifficulty.EASY)

for episode in range(100):
    reward = 0
    while not env.is_gameover:
        # choose action
        idx_action = policy.take_action(env.agent.current_location, env.q_values)
        # store previous coordinates of the agent
        prev_x, prev_y = env.agent.current_location
        # transition to next state and get immediate reward
        reward = env.step(idx_action)
        # get previous q-value
        prev_q_value = env.q_values[prev_x, prev_y, idx_action]
        # store updated agent location
        agent_x, agent_y = env.agent.current_location
        # compute temporal error difference
        t_d_error = reward + (gamma * np.max(env.q_values[agent_x, agent_y])) - prev_q_value
        # update Q-value of the previous state-action pair.
        updated_q_value = prev_q_value + (lr * t_d_error)
        env.q_values[prev_x, prev_y, idx_action] = updated_q_value
    env.reset()
    print(reward)


