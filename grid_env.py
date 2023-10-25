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

        self.position = self.environment.start_pos
        assert self.environment.grid[self.position] == 0
        self.position_history = [] 

    def learn(self, N=100): 
        for i in range(N): # Run through the maze N times
            self.run_maze()

    def sense(self):
        left_sensor = self.environment.get_distance_to_wall(self.position, 'L')
        right_sensor = self.environment.get_distance_to_wall(self.position, 'R')
        up_sensor = self.environment.get_distance_to_wall(self.position, 'U')
        down_sensor = self.environment.get_distance_to_wall(self.position, 'D')

        return (left_sensor, right_sensor, up_sensor, down_sensor)

    def is_valid_action(self, action):
        return self.environment.get_distance_to_wall(self.position, action) != 0

    def take_random_action(self):
        action_space = ['U','D','L','R']
        action = np.random.choice(action_space)
        while not self.is_valid_action(action):
            action_space.remove(action)
            action = np.random.choice(action_space)

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

        return action

    def take_bayesian_action(self, sensor_state):
        pass

    def run_maze(self):
        while self.position != self.environment.end_pos:
            self.position_history.append(self.position)
            sensor_state = self.sense()
            action = self.take_random_action()

        return self.position_history





environment = GridMazeEnvironment()
environment.generate_maze()
agent = Bayegent(environment)

position_history = agent.run_maze()

environment.visualize_agent_trajectory(position_history)

# learn(agent, environment)