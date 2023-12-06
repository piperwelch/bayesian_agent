import os
import sys 
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('exp_file')
parser.add_argument('exp_name')

args = parser.parse_args()

# Create experiment directory if it doesn't exist
if not os.path.exists('./experiments'):
    os.system('mkdir experiments')
if not os.path.exists(f'./experiments/{args.exp_name}'):
    os.system(f'mkdir ./experiments/{args.exp_name}')

# Read the experiment file into exp_arms variable
exp_file = open(args.exp_file)
exp_string = exp_file.read()
exp_arms = eval(exp_string)
exp_file.close()

# assert args.n_runs > 0, "n_runs larger must be larger than 0"

if __name__ == '__main__':
    for i, arm in enumerate(exp_arms):
        subprocess.Popen(['python3', 'run_trial.py', args.exp_file, args.exp_name, arm, str(i)])

