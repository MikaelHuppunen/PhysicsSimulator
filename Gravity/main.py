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

mass = space.get_initial_mass()
position = space.get_initial_position()
velocity = space.get_initial_velocity()
radius = space.get_initial_radius()

mass_grid = space.get_mass_grid(mass, position)
momentum_grid = space.get_momentum_grid(mass, position, velocity)

mass_grid = space.get_mass_grid(mass, position)
momentum_grid = space.get_momentum_grid(mass, position, velocity)
gravitational_field_grid = space.get_gravitational_field_grid(mass, position)
gravitational_field_derivative_grid = space.get_gravitational_field_derivative_grid(mass, position, velocity)

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

def draw_grid(space, grid, width, height):
    for y in range(space.row_count):
        for x in range(space.column_count):
            if(grid[y,x] > 0):
                color = 255-(grid[y,x]-np.min(grid))/(np.max(grid)-np.min(grid))*255
                pygame.draw.rect(screen, (color,color,color), (x/space.column_count*width, y/space.row_count*height, np.ceil(width/space.column_count), np.ceil(height/space.row_count)))

def draw_grid2(space, grid, width, height):
    max_amplitude = np.max(abs(grid-0.5))
    for y in range(space.row_count):
        for x in range(space.column_count):
            color = ((grid[y,x]-0.5)/max_amplitude+1)*255/2
            pygame.draw.rect(screen, (color,color,color), (x/space.column_count*width, y/space.row_count*height, np.ceil(width/space.column_count), np.ceil(height/space.row_count)))

running = True
while running:
    pygame.time.Clock().tick(6)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
    screen.fill((255, 255, 255))
    draw_grid2(space, gravitational_field_derivative_grid, width, height)
    #draw_grid(space, gravitational_field_grid, width, height)
    #space.simulate_next_state(mass, velocity, position, radius)
    #mass_grid = space.get_mass_grid(mass, position)
    #gravitational_field_grid = space.get_gravitational_field_grid(mass, position)
    #gravitational_field_derivative_grid = space.get_gravitational_field_derivative_grid(mass, position, velocity)
    gravitational_field_grid, gravitational_field_derivative_grid = ai.play(args, space, model_dict, gravitational_field_grid, gravitational_field_derivative_grid)

    pygame.display.flip()

pygame.quit()