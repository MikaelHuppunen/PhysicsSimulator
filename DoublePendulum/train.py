from ai import *

args = {
    'num_iterations': 8,
    'num_selfPlay_iterations': 32,
    'num_parallel_systems': 100,
    'num_epochs': 4,
    'batch_size': 128
}

system = DoublePendulum()

learn(args, system)