from ai import *

args = {
    'num_iterations': 1,
    'num_simulation_iterations': 16384,
    'max_time_steps': 10,
    'num_parallel_systems': 100,
    'num_epochs': 32,
    'batch_size': 128,
    'read_from_file': True
}

system = Space()

learn(args, system)