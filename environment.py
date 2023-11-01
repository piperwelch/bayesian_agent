
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mazelib import Maze
from mazelib.generate.Prims import Prims


class GridMazeEnvironment:
    def __init__(self, seed, maze_file = "maze.txt"):
        self.start_pos = None
        self.end_pos = None
        self.width = None
        self.height = None
        self.grid = None
        self.seed = seed 
        #self.read_maze_from_file(maze_file)
        self.random_maze_generation()
        self.height = np.array(self.grid).shape[0]
        self.width = np.array(self.grid).shape[1]

    def random_maze_generation(self):
        maze_dim_x = 3
        maze_dim_y = 3

        m = Maze()
        m.set_seed(self.seed)
        m.generator = Prims(maze_dim_x, maze_dim_y)
        m.generate()
        self.grid = m.grid
        m._generate_inner_entrances()

        self.start_pos, self.end_pos = m.start, m.end


    def read_maze_from_file(self, file_path):
        maze = []
        in_maze_section = False

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("Start:"):
                    self.start_pos = tuple(map(int, line.split("(")[1].split(")")[0].split(",")))
                elif line.startswith("End:"):
                    self.end_pos = tuple(map(int, line.split("(")[1].split(")")[0].split(",")))
                elif line == "Maze:":
                    in_maze_section = True
                elif in_maze_section:
                    row = list(line)
                    row = [int(x) for x in row]
                    maze.append(row)

        self.grid = np.array(maze) 
 

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

    def visualize_agent_trajectory(self, position_history, file_name):
        N, M = len(self.grid), len(self.grid[0])
        fig, ax = plt.subplots(figsize=(M, N))

        # Draw grid based on 0s (walkable) and 1s (walls)
        for r in range(N):
            for c in range(M):
                color = 'white' if self.grid[r][c] == 0 else 'black'
                if r == self.end_pos[0] and c == self.end_pos[1]: 
                    color = 'pink'
                ax.add_patch(patches.Rectangle((c, N-r-1), 1, 1, facecolor=color, edgecolor='black'))

        agent_square = patches.Rectangle((0, 0), 1, 1, facecolor='red')  # Initialize agent square

        ax.set_xticks(range(M+1))
        ax.set_yticks(range(N+1))
        ax.set_xlim(0, M)
        ax.set_ylim(0, N)
        ax.set_aspect('equal')
        plt.gca().invert_yaxis()  # to match the row-column indexing

        def init():
            agent_square.set_xy((0, 0))
            ax.add_patch(agent_square)
            return agent_square,

        def update(frame):
            r, c = position_history[frame]
            agent_square.set_xy((c, N-r-1))
            return agent_square,

        ani = animation.FuncAnimation(fig, update, frames=len(position_history), init_func=init, blit=True)
        ani.save(f'{file_name}.gif')
        plt.show()