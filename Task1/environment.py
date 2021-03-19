import numpy as np
from utils import Environment, GameDifficulty, Experiment
from policy import E_Greedy
import matplotlib.pyplot as plt

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
        # average final reward over the number of action taken
        reward_history.append(env.final_reward)
        env.reset()
    return reward_history, env.q_values


# policy = E_Greedy(0.1)
# env = Environment(dimension=20)
# results = train(episode=10, env=env, lr=0.9, gamma=0.9, policy=policy)
experiments = [Experiment(difficulty=GameDifficulty.EASY),
               Experiment(difficulty=GameDifficulty.MEDIUM),
               Experiment(difficulty=GameDifficulty.HARD),
               Experiment(difficulty=GameDifficulty.EXTREME)]

exp_results = {}
for exp in experiments:
    history_rewards = []
    history_q_values = []
    for dim in [20, 30, 40, 60]:
        env = Environment(dimension=dim, difficulty=exp.difficulty)
        policy = E_Greedy(exp.epsilon)
        rewards, q_values = train(episode=exp.episodes, env=env, lr=exp.lr, gamma=exp.gamma, policy=policy)
        history_rewards.append(rewards)
        history_q_values.append(q_values)
    exp_results[exp.difficulty.name] = {'q_values': history_q_values, 'rewards': history_rewards}
    print('First experiment completed')


def plot_image(exp_results, exp_name):
    n_experiments = len(exp_results[exp_name]['rewards'])

    plt.figure(figsize=(10, 8))
    for i in range(n_experiments):
        plt.subplot(2, 2, i+1)
        plt.plot(exp_results[exp_name]['rewards'][i])
        plt.xscale('log')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('{} Results, Dimension {}'.format(exp_name, (i+1)*10))

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()


for exp_name in ['EASY', 'MEDIUM']:
    plot_image(exp_results, exp_name)

for exp_name in ['HARD', 'EXTREME']:
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


