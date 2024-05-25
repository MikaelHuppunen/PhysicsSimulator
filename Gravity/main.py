from gravity import approximate
import ai
import numpy as np
import pygame

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

args = {
    'search': True,
}
model_dict = "./Gravity/models/model_0_Space.pt"

pygame.init()

width, height = 800, 700
screen = pygame.display.set_mode((width, height))
scale = 5e8

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
    pygame.time.Clock().tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
    screen.fill((255, 255, 255))
    for i in range(len(mass)):
        pygame.draw.circle(screen, (0,0,0), (position[i][0]/scale+width/2, position[i][1]/scale+height/2), 10)
    #space.simulate_next_state(mass, velocity, position, radius)
    position, velocity = ai.play(args, space, model_dict, mass, velocity, position)
    
    pygame.display.flip()

pygame.quit()