from ai import *

args = {
    'num_iterations': 1,
    'num_simulation_iterations': 128,
    'max_time_steps': 600,
    'num_parallel_systems': 100,
    'num_epochs': 32,
    'batch_size': 128
}

system = Space()

learn(args, system)