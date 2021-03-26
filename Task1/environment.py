import numpy as np
from utils import Environment, GameDifficulty, Experiment
from policy import E_Greedy
import matplotlib.pyplot as plt

def train(episode, env, lr, gamma, policy):
    reward_history = []
    for episode in range(episode):
        running_reward = []
        while not env.is_gameover:
            # choose action
            idx_action = policy.take_action(env.agent.current_location, env.q_values)
            # store previous coordinates of the agent
            prev_x, prev_y = env.agent.current_location
            # transition to next state and get immediate reward
            reward = env.step(idx_action)
            running_reward.append(reward)
            # get previous q-value
            prev_q_value = env.q_values[prev_x, prev_y, idx_action]
            # store updated agent location
            agent_x, agent_y = env.agent.current_location
            # compute temporal error difference
            t_d_error = reward + (gamma * np.max(env.q_values[agent_x, agent_y]) - prev_q_value)
            # update Q-value of the previous state-action pair.
            updated_q_value = prev_q_value + (lr * t_d_error)
            env.q_values[prev_x, prev_y, idx_action] = updated_q_value
        # at the end of each game store the average reward
        running_reward.append(env.final_reward)
        if episode % 100 == 0:
            avg_reward = np.mean(running_reward)
            reward_history.append(avg_reward)
        env.reset()
    return reward_history


## HYPERPARAMETERS

episodes = 1000
experiments = [Experiment(episodes=episodes, difficulty=GameDifficulty.EASY),
               Experiment(episodes=episodes, difficulty=GameDifficulty.HARD)]
exp_results = {}
for idx, exp in enumerate(experiments):
    history_rewards = []
    history_q_values = []
    print('Beginning experiment [{}/{}] ...'.format(idx+1, len(experiments)))
    dim_batch = [10, 15, 20, 25]
    for dim in dim_batch:
        env = Environment(dimension=dim, difficulty=exp.difficulty)
        policy = E_Greedy(exp.epsilon)
        rewards = train(episode=exp.episodes, env=env, lr=exp.lr, gamma=exp.gamma, policy=policy)
        history_rewards.append(rewards)

    exp_results[exp.difficulty.name] = {'rewards': history_rewards, 'dim': dim_batch}
    print('Experiment [{}/{}] completed'.format(idx, len(experiments)))


def plot_image(exp_results, exp_name):
    n_experiments = len(exp_results[exp_name]['rewards'])
    dim = exp_results[exp_name]['dim']
    plt.figure(figsize=(10, 8))
    for i in range(n_experiments):
        plt.subplot(2, 2, i+1)
        plt.plot(exp_results[exp_name]['rewards'][i])
        plt.xscale('log')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('{} Results, Dimension {}'.format(exp_name, dim[i]))

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()


for exp_name in ['EASY', 'HARD']:
    plot_image(exp_results, exp_name)


# # plot results
# plt.figure(figsize=(8, 8))
# plt.subplot(2, 2, 1)
# plt.plot(exp_results['EASY']['rewards'][0])
# plt.xscale('log')
# plt.xlabel('Episodes')
# plt.ylabel('Rewards')
# plt.title('{} Results, Dimension {}'.format('EASY', 10))
#
#
# plt.subplot(2, 2, 2)
# plt.plot(exp_results['EASY']['rewards'][1])
# plt.xscale('log')
# plt.xlabel('Episodes')
# plt.ylabel('Rewards')
# plt.title('{} Results, Dimension {}'.format('EASY', 20))
#
#
# plt.subplot(2, 2, 3)
# plt.plot(exp_results['EASY']['rewards'][2])
# plt.xscale('log')
# plt.xlabel('Episodes')
# plt.ylabel('Rewards')
# plt.title('{} Results, Dimension {}'.format('EASY', 30))
#
#
# plt.subplot(2, 2, 4)
# plt.plot(exp_results['EASY']['rewards'][3])
# plt.xscale('log')
# plt.xlabel('Episodes')
# plt.ylabel('Rewards')
# plt.title('{} Results, Dimension {}'.format('EASY', 40))
#
# plt.subplots_adjust(wspace=0.4, hspace=0.4)
#
# plt.show()


