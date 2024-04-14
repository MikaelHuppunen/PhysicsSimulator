from ai import *

args = {
    'num_iterations': 2,
    'num_simulation_iterations': 16,
    'max_time_steps': 6,
    'num_parallel_systems': 100,
    'num_epochs': 16,
    'batch_size': 128
}

system = Space()

learn(args, system)