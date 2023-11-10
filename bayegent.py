import numpy as np 
import random 
import matplotlib.pyplot as plt 
from scipy.ndimage import convolve
import constants 
from collections import Counter


class Bayegent:
    def __init__(self, environment, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.environment = environment
        
        self.reset_prior()
        self.posterior = np.zeros((environment.height, environment.width))
        self.likelihood = {} # Dictionary of sensations to position distributions 

        self.qtable = {}
        self.curiosity = constants.curosity

        self.position = self.environment.start_pos
        assert self.environment.grid[self.position] == 0

    def learn_bayesian(self, n_runs=100): 
        for i in range(n_runs): # Run through the maze N times
            position_history, sa_history, posterior_history  = self.run_maze_bayesian(i)

            self.update_qtable(sa_history)
            self.update_likelihood(posterior_history, sa_history)

            print(f'{i+1}/{n_runs} runs complete')
            print(f'Steps taken: {len(position_history)}')
            # TODO: implement bayesian stuff...

        return position_history

    def learn_qtable(self, n_runs=10): 
        for i in range(n_runs): # Run through the maze N times
            self.position = self.environment.start_pos # Reset position
            position_history, sa_history  = self.run_maze_qtable()
            self.update_qtable(sa_history)

            print(len(position_history))
            print(f'{i+1}/{n_runs} runs complete')

        return position_history

    def reset_prior(self):
        self.prior = np.zeros((self.environment.height, self.environment.width))
        self.prior[self.environment.start_pos] = 1 # Reset prior 
    
    def update_prior(self, action):

        spread_factor = constants.confusion

        direction_to_kernel = {
        "U": np.array([[0, spread_factor, 0], [0, 1 - spread_factor, 0], [0, 0, 0]]),
        "D": np.array([[0, 0, 0], [0, 1 - spread_factor, 0], [0, spread_factor, 0]]),
        "L": np.array([[0, 0, 0], [spread_factor, 1 - spread_factor, 0], [0, 0, 0]]),
        "R": np.array([[0, 0, 0], [0, 1 - spread_factor, spread_factor], [0, 0, 0]])
        }

        spread_kernel = direction_to_kernel[action]

        self.prior = convolve(self.prior, spread_kernel, mode='constant', cval=0)
        self.prior = self.prior/np.sum(self.prior)


    def sense(self):
        left_sensor = self.environment.get_distance_to_wall(self.position, 'L')
        right_sensor = self.environment.get_distance_to_wall(self.position, 'R')
        up_sensor = self.environment.get_distance_to_wall(self.position, 'U')
        down_sensor = self.environment.get_distance_to_wall(self.position, 'D')

        return (left_sensor, right_sensor, up_sensor, down_sensor)

    def is_valid_action(self, action):
        return self.environment.get_distance_to_wall(self.position, action) != 0

    def update_position_from_action(self, action):
        # Update position
        r, c = self.position
        if action == 'R':
            self.position = (r, c+1)
        elif action == 'L':
            self.position = (r, c-1)
        elif action == 'U':
            self.position = (r-1, c)
        elif action == 'D':
            self.position = (r+1, c)

    def update_posterior(self, sensor_state):
        '''
        Given a sensor state, returns the distribution over positions in the maze
        '''

        if sensor_state not in self.likelihood:
            self.likelihood[sensor_state] = np.ones(self.environment.dims) / np.prod(self.environment.dims) # Init as uniform distribution

        self.posterior = self.likelihood[sensor_state] * self.prior
        self.posterior /= np.sum(self.posterior) # Normalize


    def take_random_action(self):
        action_space = ['U','D','L','R']
        action = np.random.choice(action_space)
        while not self.is_valid_action(action):
            action_space.remove(action)
            action = np.random.choice(action_space)

        self.update_position_from_action(action)

        return action


    def take_qtable_action(self, sensor_state):
        '''
        Take an unweighted reward-based action
        '''
        action_space = []
        for action in ['U','D','L','R']: 
            if self.is_valid_action(action):
                action_space.append(action)
        
        # Compute rewards using QTable
        reward_for_action = {a: 0 for a in action_space}
        for action in action_space:
            sa_pair = (sensor_state, action)
            if sa_pair in self.qtable:
                reward_for_action[action] += self.qtable[sa_pair]

        # With probability (1-curiosity), pick the action with the highest reward
        if random.random() < (1-self.curiosity):
            max_reward = max(reward_for_action.values())
            possible_best_actions = [a for a in action_space if reward_for_action[a] == max_reward]
            action = np.random.choice(possible_best_actions)
        else:
            action = np.random.choice(action_space)
        
        self.update_position_from_action(action)
        return action

    def take_qtable_weighted_action(self, sensor_state, posterior_distribution):
        '''
        Take a reward-based action weighted with the posterior
        '''
        action_space = []
        for action in ['U','D','L','R']: 
            if self.is_valid_action(action):
                action_space.append(action)
        
        reward_for_action = {a: 0 for a in action_space}

        for (r,c), _ in np.ndenumerate(posterior_distribution): # For each position in the posterior... 
            for action in action_space:
                sa_pair = (sensor_state, action)
                if sa_pair in self.qtable:
                    reward_for_action[action] += self.qtable[sa_pair] * posterior_distribution[r,c] # Weight reward by posterior

        # With probability 1-curiosity, pick the action with the highest reward
        if random.random() < (1-self.curiosity):
            max_reward = max(reward_for_action.values())
            possible_best_actions = [a for a in action_space if reward_for_action[a] == max_reward]
            action = np.random.choice(possible_best_actions)
        else:
            action = np.random.choice(action_space)

        self.update_position_from_action(action)

        return action


    def take_bayesian_action(self, sensor_state):
        action = self.take_qtable_weighted_action(sensor_state, self.posterior)

        return action

    def run_maze_qtable(self):
        '''
        Run the maze once with the current QTable (only RL)
        '''
        position_history = []
        sa_history = []

        position_history.append(self.position)

        while self.position != self.environment.end_pos:
            sensor_state = self.sense()
            action = self.take_qtable_action(sensor_state)
            sa_history.append((sensor_state, action))
            position_history.append(self.position)

        position_history.append(self.environment.end_pos)

        return position_history, sa_history

    def run_maze_bayesian(self, run=0):
        '''
        Run the maze once with the current likelihood distribution and QTable (Bayesian + RL)
        '''
        self.position = self.environment.start_pos # Reset position
        self.reset_prior()

        position_history = []
        sa_history = []
        posterior_history = []

        while self.position != self.environment.end_pos:
            position_history.append(self.position)

            sensor_state = self.sense()
            self.update_posterior(sensor_state) # Multiply prior by likelihood

            if run == 9: # Visualize Likelihood
                # print(self.likelihood[sensor_state], self.position)
                fig, axs = plt.subplots(1,3, figsize=(15,5))

                axs[0].imshow(self.likelihood[sensor_state], cmap='viridis')
                axs[0].scatter(self.position[1], self.position[0], color = 'red', alpha = 0.2)
                axs[0].set_title('Likelihood')

                axs[1].imshow(self.prior, cmap='viridis')
                axs[1].scatter(self.position[1], self.position[0], color = 'red', alpha = 0.2)
                axs[1].set_title('Prior')

                axs[2].imshow(self.posterior, cmap='viridis') 
                axs[2].scatter(self.position[1], self.position[0], color = 'red', alpha = 0.2)
                axs[2].set_title('Posterior')

                plt.suptitle(f'({self.position})')
                # plt.imshow(self.prior, cmap='viridis') 
                plt.pause(0.5)
                plt.clf()

            action = self.take_bayesian_action(sensor_state) # Take the action using the current posterior

            self.update_prior(action) # Update prior after the action (shift n smear)

            posterior_history.append(self.posterior)
            sa_history.append((sensor_state, action))   

        position_history.append(self.environment.end_pos)

        return position_history, sa_history, np.array(posterior_history)

    def update_qtable(self, state_action_list, alpha=0.5, gamma=0.8): # Written by ChatGPT
        """
        Update the Q-table based on a list of state, action tuples.
        
        Parameters:
        - q_table: The Q-table to update.
        - state_action_list: A list of (state, action) tuples from a maze run.
        - alpha: Learning rate.
        - gamma: Discount factor.
        
        Returns:
        - Updated Q-table.
        """
        
        # Define rewards
        step_reward = -0.1
        goal_reward = 1.0
        
        # Iterate backwards through the state-action list
        next_max_q_value = 0  # since there's no next state after reaching the goal
        
        for i, sa_pair in enumerate(reversed(state_action_list)):
            if i == 0:  # Check if it's the goal state
                reward = goal_reward
            else:
                reward = step_reward
            
            # Q-learning update rule
            if sa_pair in self.qtable:
                current_q_value = self.qtable[sa_pair]
                self.qtable[sa_pair] = (1 - alpha) * current_q_value + alpha * (reward + gamma * next_max_q_value)
            else:
                self.qtable[sa_pair] = reward 

            # Update the next_max_q_value for the next iteration
            next_max_q_value = np.max([self.qtable[sa] for sa in self.qtable.keys() if sa[0] == sa_pair[0]])
            
    def get_distribution_mode(self, distribution):
        # Find the index of the maximum value in the 2d array
        max_index = np.unravel_index(np.argmax(distribution), distribution.shape)
        # max_index is a tuple (row, col)
        return max_index


    def update_likelihood(self, posterior_history, sa_history):
        assert len(posterior_history) == len(sa_history)
        unique_sensations = np.unique([sa[0] for sa in sa_history], axis=0)
        unique_sensations = [tuple(s) for s in unique_sensations]

        for s in unique_sensations:
            # Get the unique modes
            posteriors = [posterior_history[i] for i, sa in enumerate(sa_history) if sa[0] == s]
            all_modes_s = Counter([self.get_distribution_mode(post) for post in posteriors])
            unique_modes = list(all_modes_s)

            likelihood = np.zeros(posteriors[0].shape)

            for mode in unique_modes:
                post_sum = np.sum([post for post in posteriors if self.get_distribution_mode(post) == mode], axis=0)
                likelihood += post_sum / all_modes_s[mode] 
            
            self.likelihood[s] = likelihood / len(unique_modes)
        



