from ai import *

args = {
    'C': 2,
    'num_searches': 1,
    'num_iterations': 8,
    'num_selfPlay_iterations': 10,
    'num_parallel_systems': 100,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

system = DoublePendulum()

learn(args, system)