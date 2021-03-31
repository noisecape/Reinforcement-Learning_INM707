import numpy as np
from utils import Environment, GameDifficulty, Experiment
from policy import E_Greedy
import matplotlib.pyplot as plt
import os, pickle


def train(episode, env, lr, gamma, policy):
    reward_history = []
    for episode in range(episode):
        running_reward = []
        seed = 0
        while not env.is_gameover:
            # to be able to replicate the same experiment while
            # keeping the stochastic component in the environment
            np.random.seed(seed)
            seed += 1
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
        if episode % 10 == 0:
            avg_reward = np.mean(running_reward)
            reward_history.append(avg_reward)
        env.reset()
    return reward_history, env.q_values


## HYPERPARAMETERS


# hyperparameters
episodes = 10000
gamma = 0.96
dim_batch = [10, 15, 20, 25]
experiments = [Experiment(episodes=episodes, difficulty=GameDifficulty.EASY, gamma=gamma),
               Experiment(episodes=episodes, difficulty=GameDifficulty.HARD, gamma=gamma)]

exp_results = {}
save_path = os.path.join(os.curdir, 'saved_files/difficulty_comparison.pickle')
if not os.path.exists(os.path.join(os.curdir, 'saved_files')):
    os.mkdir(os.path.join(os.curdir, 'saved_files'))

if os.path.exists(save_path):
    # load values
    with open(save_path, 'rb') as file2load:
        print('Loading previous values...')
        exp_results = pickle.load(file2load)
        print('Values Loaded')

else:
    for idx, exp in enumerate(experiments):
        history_rewards = []
        history_q_values = []
        print('Beginning experiment [{}/{}] ...'.format(idx+1, len(experiments)))
        for dim in dim_batch:
            env = Environment(dimension=dim, difficulty=exp.difficulty)
            policy = E_Greedy(exp.epsilon)
            rewards, q_values = train(episode=exp.episodes, env=env, lr=exp.lr, gamma=exp.gamma, policy=policy)
            history_rewards.append(rewards)
            history_q_values.append(q_values)
        exp_results[exp.difficulty.name] = {'rewards': history_rewards, 'dim': dim_batch, 'q_values': history_q_values}
        print('Experiment [{}/{}] completed'.format(idx+1, len(experiments)))
    print('Training Completed')
    file_name = 'difficulty_comparison.pickle'
    print('Saving values...')
    with open(os.path.join(os.curdir, 'saved_files/'+file_name), 'wb') as file2store:
        pickle.dump(exp_results, file2store, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')


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

# plot easy episodes
plot_image(exp_results, exp_name='EASY')
# plot hard episodes
plot_image(exp_results, exp_name='HARD')

# Check how the reward behaves when the dimensions and the eps-values change
# Set an eps value and a dimension. Execute experiments, compute avg to see
# how different eps values affect rewards in different sizes.

# Hyperparameters
eps_values = [.01, .1, .2, .5]
dim = 20
results = {}
gamma = .96
episodes = 100
# in a stochastic environment we want the agent to learn little by little
# lr = 0 -> no learning
# lr = 1 -> learn only the most recent information
lr = .2

save_path = os.path.join(os.curdir, 'saved_files/eps_comparison.pickle')
if os.path.exists(save_path):
    # load values
    with open(save_path, 'rb') as file2load:
        print('Loading previous values...')
        results = pickle.load(file2load)
        print('Values Loaded')
else:
    for idx, eps in enumerate(eps_values):
        print('Beginning experiment [{}/{}] ...'.format(idx+1, len(eps_values)))
        reward_history = []
        name_exp = 'Experiment {}'.format(idx+1)
        policy = E_Greedy(eps)
        env = Environment(dimension=dim, difficulty=GameDifficulty.EASY)
        rewards, _ = train(episodes, env, lr=lr, gamma=gamma, policy=policy)
        reward_history.append(rewards)
        results[name_exp] = {'rewards': reward_history, 'eps': eps_values}
        print('Experiment [{}/{}] completed'.format(idx+1, len(eps_values)))
    file_name = 'eps_comparison.pickle'
    with open(os.path.join(os.curdir, 'saved_files/'+file_name), 'wb') as file2store:
        pickle.dump(results, file2store, protocol=pickle.HIGHEST_PROTOCOL)


# function used to plot results
def plot_eps(results):
    plt.figure(figsize=(10, 8))
    for idx, name in enumerate(results.keys()):
        data = results[name]['rewards'][0]
        eps = results[name]['eps'][idx]
        plt.subplot(2, 2, idx+1)
        plt.plot(data)
        plt.xscale('log')
        plt.xlabel('Reward Samples')
        plt.ylabel('Average Rewards')
        plt.title('{} Results, eps {}'.format(name, eps))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()


# plot rewards vs eps values.
plot_eps(results)

# inference stage

n_games = 500
dim = [10, 15, 20, 25]
difficulty = GameDifficulty.EASY.name
inference_results = {}

print('Start Experiment')
# iterate through all the desired dimensions
for idx in range(len(dim)):
    # init environment
    env = Environment(dim[idx])
    # init name of experiment used to store final info
    # retrieve q_values from previous training
    q_values = exp_results[difficulty]['q_values'][idx]
    winnings = []
    losses = []
    # iterate through all the games
    for game in range(n_games):
        env.reset()
        limit_steps = 0
        while not env.is_gameover:
            # get agent location
            agent_x, agent_y = env.agent.current_location
            # get action idx
            action_idx = np.argmax(q_values[agent_x][agent_y])
            # take a step and get reward
            _ = env.step(action_idx)
            limit_steps += 1
            if limit_steps > 1000:
                env.is_gameover = True
        # check if the agent won
        if env.goal_reached:
            winnings.append(env.final_reward)
        else:  # the agent lost the game
            losses.append(env.final_reward)
        # every 100 games print some useful statistics
        if game % 100 == 0:
            print('Environment {}, Games [{}]/[{}] executed. Wins: {}, Losses: {}'.format(difficulty,
                                                                                          game+1,
                                                                                          n_games,
                                                                                          len(winnings),
                                                                                          len(losses)))
    avg_reward_wins = np.mean(winnings)
    avg_reward_losses = np.mean(losses)
    inference_results[difficulty] = {'Wins': len(winnings),
                                     'Average_reward': avg_reward_wins,
                                     'Losses': len(losses),
                                     'Average_reward': avg_reward_losses}

    print('Environment {}, Total wins: {}, average reward: {}, Total losses: {}, average reward: {}'.format(difficulty,
                                                                                                            len(winnings),
                                                                                                            avg_reward_wins,
                                                                                                            len(losses),
                                                                                                            avg_reward_losses))


