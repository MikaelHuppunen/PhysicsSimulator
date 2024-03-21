import ai
import numpy as np

space = ai.Space()

mass_grid = space.get_initial_mass_state()
momentum_grid = space.get_initial_momentum_state()

for i in range(space.row_count):
    print(np.around(mass_grid[i], 2))
print("mass")

for i in range(space.dimensions):
    for j in range(space.row_count):
        print(momentum_grid[i,j])
    print(f"mometum[{i}]")