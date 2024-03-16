from ai import *

args = {
    'num_iterations': 1,
    'num_selfPlay_iterations': 4096,
    'num_parallel_systems': 100,
    'num_epochs': 32,
    'batch_size': 128
}

system = DoublePendulum()

learn(args, system)