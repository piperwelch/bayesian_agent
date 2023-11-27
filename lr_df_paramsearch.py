import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import concurrent
import pickle

from bayegent import Bayegent
from environment import GridMazeEnvironment

def run_one_seed(seed, parameters, n_runs=100):
    environment = GridMazeEnvironment(seed)
    agent = Bayegent(environment, seed, parameters=parameters)

    all_position_histories = agent.learn_bayesian(n_runs)

    return all_position_histories


# Function to run one combination of lr and df
def run_one_combination(curiosities, lr, df, n_runs=100):
    print(f'LR: {lr}, DF: {df}')
    start = time.time()
    # for seed in range(50):
    #     print(f'{seed}/50')
    position_histories = run_one_seed(0, 
                                        parameters={
                                            'curiosity': curiosities,
                                            'step_reward': -0.1,
                                            'goal_reward': 1,
                                            'learning_rate': lr,
                                            'discount_factor': df,
                                        },
                                        n_runs=n_runs)
    end = time.time()

    path_lengths = [len(pl) for pl in position_histories]
    return lr, df, path_lengths, end-start

def run_experiment(lr_interval, df_interval, curiosity_interval=(0.2,0.9), gridsize=10, n_runs=100):
    curiosities = list(reversed(np.linspace(curiosity_interval[0], curiosity_interval[1], n_runs)))
    learning_rates = list(reversed(np.linspace(lr_interval[0], lr_interval[1], gridsize)))
    discount_factors = list(reversed(np.linspace(df_interval[0], df_interval[1], gridsize)))

    # Dictionaries to store results
    last_path_lengths = {lr: {df: 0 for df in discount_factors} for lr in learning_rates}
    min_path_lengths = {lr: {df: 0 for df in discount_factors} for lr in learning_rates}
    running_times = {lr: {df: 0 for df in discount_factors} for lr in learning_rates}

    # Use ProcessPoolExecutor to parallelize
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_one_combination, curiosities, lr, df) for lr in learning_rates for df in discount_factors]

        for future in concurrent.futures.as_completed(futures):
            lr, df, path_lengths, run_time = future.result()
            running_times[lr][df] = run_time
            last_path_lengths[lr][df] = path_lengths[-1]
            min_path_lengths[lr][df] = min(path_lengths)

    return last_path_lengths, min_path_lengths, running_times

last_path_lengths, min_path_lengths, running_times = run_experiment((0.3,0.8), (0.3,0.8), gridsize=7)

with open('last_path_lengths_lrdf.pkl', 'wb') as pf:
    pickle.dump(last_path_lengths, pf, protocol=pickle.HIGHEST_PROTOCOL)

with open('min_path_lengths_lrdf.pkl', 'wb') as pf:
    pickle.dump(min_path_lengths, pf, protocol=pickle.HIGHEST_PROTOCOL)

with open('running_times_lrdf.pkl', 'wb') as pf:
    pickle.dump(running_times, pf, protocol=pickle.HIGHEST_PROTOCOL)