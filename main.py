from bayegent import Bayegent
from environment import GridMazeEnvironment
import sys 

seed = 0
vis_bool = True
gif_name = "maze_trace"
n_runs = 10 


if len(sys.argv) == 5: 

    seed = int(sys.argv[1])
    n_runs = int(sys.argv[2]) #must be more than 0
    if n_runs <=0: n_runs=1
    vis_bool = int(sys.argv[3])
    gif_name = sys.argv[4]


environment = GridMazeEnvironment()
agent = Bayegent(environment, seed)

position_history = agent.learn_qtable(n_runs=n_runs)

if vis_bool:
    environment.visualize_agent_trajectory(position_history, gif_name)
