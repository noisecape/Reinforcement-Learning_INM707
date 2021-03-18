import numpy as np
from utils import Environment, GameDifficulty, Experiment
from policy import E_Greedy

def train(episode, env, lr, gamma, policy):
    reward_history = []
    for episode in range(episode):
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
            t_d_error = reward + (gamma * np.max(env.q_values[agent_x, agent_y]) - prev_q_value)
            # update Q-value of the previous state-action pair.
            updated_q_value = prev_q_value + (lr * t_d_error)
            env.q_values[prev_x, prev_y, idx_action] = updated_q_value
        reward_history.append(env.final_reward)
        env.reset()
    return reward_history
policy = E_Greedy(0.9)
env = Environment()
results = train(episode=1000, env=env, lr=0.9, gamma=0.9, policy=policy)
print(results)


# exp_batch = [Experiment(episodes=1000, dimension=10, difficulty=GameDifficulty.HARD, epsilon=0.9, gamma=0.9, lr=0.9),
#              Experiment(episodes=1000, dimension=20, difficulty=GameDifficulty.HARD, epsilon=0.9, gamma=0.9, lr=0.9),
#              Experiment(episodes=1000, dimension=30, difficulty=GameDifficulty.HARD, epsilon=0.9, gamma=0.9, lr=0.9),
#              Experiment(episodes=1000, dimension=40, difficulty=GameDifficulty.HARD, epsilon=0.9, gamma=0.9, lr=0.9)]
# results = []
# for exp in exp_batch:
#     env = Environment(exp.lr, exp.gamma, exp.dimension, exp.difficulty)
#     policy = E_Greedy(exp.epsilon)
#     exp_result = train(exp.episodes, env, exp.lr, exp.gamma, policy=policy)
#     avg_results = exp_result/exp.episodes
#     print(avg_results)
#     results.append(avg_results)
