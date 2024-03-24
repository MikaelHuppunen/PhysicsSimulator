from ai import *

args = {
    'num_iterations': 1,
    'num_simulation_iterations': 2,
    'num_parallel_systems': 100,
    'num_epochs': 2,
    'batch_size': 128
}

system = Space()

learn(args, system)