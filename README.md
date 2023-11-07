# Bayegent

Kam Bielawski + Piper Welch's Bayesian Statistics project from F2023. 

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)


## Getting Started

To run, please run main.py from the command line. We have 4 optional command line args: --n_runs, --seed, --vis, and --gif_name. If --vis is not present, the maze trace will not be replayed or saved to a file. If no command line arguments are given, then defaults will be used. 

```bash
# 2 ways to run our project
python3 main.py 

python3 main.py --n_runs 3 --seed 5 --vis -gif_name file
```

## Prerequisites
argparse 1.1 