import sys 
import argparse

from bayegent import Bayegent
from environment import GridMazeEnvironment


parser = argparse.ArgumentParser()
parser.add_argument('--n_runs', type=int, default=10, help='Number of maze runs for Bayegent. n > 0')
parser.add_argument('--seed', type=int, default=0, help='Seed for RNG')
parser.add_argument("--vis", action="store_true", help="A boolean flag for visualizing") #have to do this to make booleans work 
parser.add_argument('--gif_name', type=str, default='maze_trace', help='Name for saved file of maze run')
parser.add_argument('--show_maze', action="store_true", help='A boolean flag for showing the maze before running the learning algorithm')

args = parser.parse_args()

assert args.n_runs > 0, "n_runs larger must be larger than 0"

exp_parameters = {
    'curiosity': [0.7 for _ in range(args.n_runs)],
    'step_reward': -0.1,
    'goal_reward': 1,
    'learning_rate': 0.5,
    'discount_factor': 0.8,
}

if __name__ == '__main__':
    environment = GridMazeEnvironment(args.seed)
    agent = Bayegent(environment, args.seed, parameters=exp_parameters)

    if args.show_maze:
        environment.show_maze()

    # position_history = agent.learn_qtable(n_runs=args.n_runs)
    position_histories = agent.learn_bayesian(n_runs=args.n_runs)

    if args.vis:
        environment.visualize_agent_trajectory(position_histories[-1], args.gif_name)
