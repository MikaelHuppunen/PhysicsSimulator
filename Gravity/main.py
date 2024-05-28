from gravity import approximate, total_energy
import ai
import numpy as np
import pygame
from copy import copy, deepcopy

def print_mass_grid(space, mass_grid):
    for i in range(space.row_count):
        print(np.around(mass_grid[i], 2))
    print("mass")

def print_momentum_grid(space, momentum_grid):
    for i in range(space.dimensions):
        for j in range(space.row_count):
            print(np.around(momentum_grid[i,j],3))
        print(f"mometum[{i}]")

space = ai.Space()

mass, position, velocity, radius = space.get_initial_state()
ai_mass, ai_position, ai_velocity, ai_radius = deepcopy(mass), deepcopy(position), deepcopy(velocity), deepcopy(radius)

total_energy_at_start = total_energy(mass, velocity, position, space.gravitational_constant)

args = {
    'search': True,
}
model_dict = "./Gravity/models/model_0_Space.pt"

pygame.init()

width, height = 800, 700
screen = pygame.display.set_mode((width, height))

time_step = 0

waiting = True
while(waiting):
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            waiting = False
            break
        break

running = True
while running:
    pygame.time.Clock().tick(5)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
    screen.fill((255, 255, 255))
    for i in range(len(mass)):
        pygame.draw.circle(screen, (255,0,0), (position[i][0]/space.scale+width/2, position[i][1]/space.scale+height/2), 10)
        pygame.draw.circle(screen, (0,0,0), (ai_position[i][0]/space.scale+width/2, ai_position[i][1]/space.scale+height/2), 10)
    space.simulate_next_state(mass, velocity, position, radius)
    ai_position, ai_velocity = ai.play(args, space, model_dict, ai_mass, ai_velocity, ai_position, ai_radius)

    simulation_total_energy = total_energy(mass, velocity, position, space.gravitational_constant)
    ai_total_energy = total_energy(ai_mass, ai_velocity, ai_position, space.gravitational_constant)
    print(f"simulation: {float('%.2g' % (100*(simulation_total_energy-total_energy_at_start)/total_energy_at_start))}%, ai: {float('%.2g' % (100*(ai_total_energy-total_energy_at_start)/total_energy_at_start))}%")
    
    pygame.display.flip()

pygame.quit()