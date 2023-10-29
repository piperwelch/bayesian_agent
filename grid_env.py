import random
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridMazeEnvironment:
    def __init__(self):
        self.start_pos = None
        self.end_pos = None
        self.width = None
        self.height = None
        self.grid = None

        self.generate_maze()

    def generate_maze(self):
        # 0 is "open", moveable space
        # 1 is a wall
        self.grid = np.array([[0,0,0,0,0],
                              [0,1,1,1,0],
                              [0,0,1,0,0],
                              [0,0,1,0,1],
                              [0,0,1,0,1]])

        self.start_pos = (4, 1)
        self.end_pos = (4, 3)
        self.width = 5
        self.height = 5

    def get_distance_to_wall(self, position, direction):
        r, c = position
        distance = -1 # Starts at -1 because we count the space we are on
        if direction == 'U':
            while (r >= 0) and (self.grid[(r,c)] != 1):
                distance += 1
                r -= 1 
        elif direction == 'D':
            while (r < self.height) and (self.grid[(r,c)] != 1):
                distance += 1
                r += 1 
        elif direction == 'R':
            while (c < self.width) and (self.grid[(r,c)] != 1):
                distance += 1
                c += 1 
        elif direction == 'L':
            while (c >= 0) and (self.grid[(r,c)] != 1):
                distance += 1
                c -= 1

        return distance

    def visualize_agent_trajectory(self, position_history): # Made by ChatGPT
        """ 
        Animate the trajectory of an agent in a grid environment.

        Parameters:
        - grid (list of lists): NxM grid where 0 represents walkable spaces and 1 represents walls.
        - trajectory (list): List of (r, c) tuples representing the agent's position at every timestep.
        """
        N, M = len(self.grid), len(self.grid[0])
        fig, ax = plt.subplots(figsize=(M, N))

        # Draw grid based on 0s (walkable) and 1s (walls)
        for r in range(N):
            for c in range(M):
                color = 'white' if self.grid[r][c] == 0 else 'black'
                ax.add_patch(patches.Rectangle((c, N-r-1), 1, 1, facecolor=color, edgecolor='black'))

        agent_dot, = ax.plot([], [], 'ro')  # Initialize agent dot

        ax.set_xticks(range(M+1))
        ax.set_yticks(range(N+1))
        ax.set_xlim(0, M)
        ax.set_ylim(0, N)
        ax.set_aspect('equal')
        plt.gca().invert_yaxis()  # to match the row-column indexing

        def init():
            agent_dot.set_data([], [])
            return agent_dot,

        def update(frame):
            r, c = position_history[frame]
            agent_dot.set_data(c+0.5, N-r-0.5)  # Convert row, column to y, x and center the agent in the cell
            return agent_dot,

        ani = animation.FuncAnimation(fig, update, frames=len(position_history), init_func=init, blit=True)
        ani.save('random.gif')
        plt.show()


class Bayegent:
    def __init__(self, environment):
        self.environment = environment
        
        self.prior = np.zeros((environment.height, environment.width))
        self.posterior = np.zeros((environment.height, environment.width))
        self.likelihood = {} # Map between sensory states and probabilities (likelihood function)

        self.qtable = {}
        self.curiosity = 0.4

        self.position = self.environment.start_pos
        assert self.environment.grid[self.position] == 0

    def learn(self, N=100): 
        for i in range(N): # Run through the maze N times
            position_history, sa_history, posterior_history  = self.run_maze()

            self.update_qtable(sa_history)
            self.update_likelihood(posterior_history)
            # TODO: implement bayesian stuff...
            
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

        while self.position != self.environment.end_pos:
            position_history.append(self.position)

            sensor_state = self.sense()
            action = self.take_qtable_action(sensor_state)
            
            sa_history.append((sensor_state, action))

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

        return position_history, sa_history, posterior_history

    def update_qtable(self, sa_history):
        rewards = np.linspace(0, 1, len(sa_history))
        for i, sa_pair in enumerate(sa_history):
            # if i % 50 == 0:
                # print(f'{i}/{len(sa_history)} SA pairs updated...')
            if sa_pair in self.qtable:
                self.qtable[sa_pair] += rewards[i]
            else:
                self.qtable[sa_pair] = rewards[i]

    def update_likelihood(self, posterior_history):
        return self.likelihood





environment = GridMazeEnvironment()
agent = Bayegent(environment)

position_history = agent.learn_qtable(n_runs=100)
# position_history, sa_history, posterior_history = agent.run_maze()
# position_history, sa_history = agent.run_maze_qtable()

print(agent.qtable)
print(len(position_history))
environment.visualize_agent_trajectory(position_history)

# learn(agent, environment)