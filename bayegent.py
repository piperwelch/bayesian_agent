import numpy as np 
import random 
import matplotlib.pyplot as plt 
from scipy.ndimage import convolve
import constants 


class Bayegent:
    def __init__(self, environment, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.environment = environment
        
        self.prior = np.zeros((environment.height, environment.width))
        self.prior[self.environment.start_pos[0], self.environment.start_pos[1]] = 1 #100% chance we are at the beginning 

        self.posterior = np.zeros((environment.height, environment.width))
        self.likelihood = {} 

        self.qtable = {}
        self.curiosity = constants.curosity

        self.position = self.environment.start_pos
        assert self.environment.grid[self.position] == 0

    def learn(self, N=100): 
        for i in range(N): # Run through the maze N times
            position_history, sa_history, posterior_history  = self.run_maze()

            self.update_qtable(sa_history)
            self.update_likelihood(posterior_history)
            # TODO: implement bayesian stuff...
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

        # plt.imshow(self.prior, cmap='viridis') 
        # plt.scatter(self.position[1], self.position[0], color = 'red', alpha = 0.2)
        # plt.pause(0.5)
        # plt.clf()


    def learn_qtable(self, n_runs=10): 
        for i in range(n_runs): # Run through the maze N times
            self.position = self.environment.start_pos # Reset position
            position_history, sa_history  = self.run_maze_qtable()
            self.update_qtable(sa_history)

            print(len(position_history))
            print(f'{i+1}/{n_runs} runs complete')

        return position_history


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

    def get_posterior(self, sensor_state):
        '''
        Given a sensor state, returns the distribution over positions in the maze
        '''
        # Prior over the positions in the maze

        # Likelihood over all possible sensor states 
        # (probability of observing this sensor state given the current world model...)

        # Likelihood =
        # for each position:
        #   prob of the sensor state? <-- uniform at first 

        # For each timestep, we have a most probable position. 

        # Posterior = 
        # for each position in the prior:
        #   multiply by the likelihood of current sensory state
        
        # First time, the likelihood is uniform...
        # We update the likelihood between maze runs by taking the 

        # 
        return []

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
        self.update_prior(action)
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

        # TODO: make the reward weighted sums over all the positions in the posterior
        #   i.e. For each position in the posterior... 
        for action in action_space:
            sa_pair = (sensor_state, action)
            if sa_pair in self.qtable:
                reward_for_action[action] += self.qtable[sa_pair]

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
        posterior_distribution = self.get_posterior(sensor_state)

        action = self.take_qtable_action(sensor_state, posterior_distribution)

        return posterior_distribution, action

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

    def run_maze(self):
        '''
        Run the maze once with the current likelihood distribution and QTable (Bayesian + RL)
        '''
        position_history = []
        sa_history = []
        posterior_history = []

        while self.position != self.environment.end_pos:
            position_history.append(self.position)

            sensor_state = self.sense()
            posterior_distribution, action = self.take_bayesian_action(sensor_state)
            
            posterior_history.append(posterior_distribution)
            sa_history.append((sensor_state, action))

        position_history.append(self.environment.end_pos)

        return position_history, sa_history, posterior_history


    # def update_qtable(self, sa_history, position_history):
    #     rewards = np.linspace(0, 1, len(sa_history))
    #     for i, sa_pair in enumerate(sa_history):
    #         # if i % 50 == 0:
    #             # print(f'{i}/{len(sa_history)} SA pairs updated...')
    #         if sa_pair in self.qtable:
    #             self.qtable[sa_pair] += rewards[i]
    #         else:
    #             self.qtable[sa_pair] = rewards[i]

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
            


    def update_likelihood(self, posterior_history):
        return self.likelihood



