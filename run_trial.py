import argparse
import os
import numpy as np

from bayegent import Bayegent
from environment import GridMazeEnvironment

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('exp_file', type=str, help='Name of experiment file')
parser.add_argument('exp_name', type=str, help='Name of experiment')
parser.add_argument('arm_name', type=str, help='Name of experiment arm')
parser.add_argument('trial', type=int, help='Trial number')
args = parser.parse_args()

# Read the experiment file into exp_arms variable
exp_file = open(args.exp_file)
exp_string = exp_file.read()
exp_arms = eval(exp_string)
exp_file.close()

# Get the parameters for this particular arm
arm_parameters = exp_arms[args.arm_name]

# Create experiment directory if it doesn't already exist
os.makedirs(f'./experiments/{args.exp_name}', exist_ok=True)

# Copy experiment file into experiment directory
os.system(f'cp {args.exp_file} ./experiments/{args.exp_name}')

def main():
    # Create an arm directory if it doesn't already exist
    os.makedirs(f'./experiments/{args.exp_name}/{args.arm_name}', exist_ok=True)

    # Generate curiosity from tuple
    arm_parameters['curiosity'] = np.linspace(arm_parameters['curiosity'][0], arm_parameters['curiosity'][1], arm_parameters['n_runs'])

    environment = GridMazeEnvironment(arm_parameters['seed'])
    agent = Bayegent(environment, arm_parameters['seed'], parameters=arm_parameters)
    
    if arm_parameters['bayesian']:
        agent.learn_bayesian(n_runs=arm_parameters['n_runs'])
    else:
        agent.learn_qtable(n_runs=arm_parameters['n_runs'])

    seed = arm_parameters['seed']
    agent.pickle_agent(f'experiments/{args.exp_name}/{args.arm_name}/{args.arm_name}_seed{seed}.pkl')
    

if __name__ == '__main__':
    main()
