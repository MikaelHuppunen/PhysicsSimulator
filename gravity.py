import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
import time
import copy

G = 6.67384e-11
#mass of sun:1.9891e30, mass of earth:5.9e24
mass = [1.9891e30, 1.9891e30, 5.9e24, 5.9e24, 5.9e24, 5.9e24]
#mass of sun:6.957e8, mass of earth:6.372e6
radius = [6.957e8, 6.957e8, 6.372e6,6.372e6,6.372e6,6.372e6]
#aphelion velocity 2.929e4
velocity = [[0,-1e4,0],[0,1e4,0], [-2e4,2e4,0],[-2e4,-2e4,0],[2e4,2e4,0],[2e4,-2e4,0]]
#aphelion distance 1.5210e11
position = [[-1.5210e11,0,0],[1.5210e11,0,0],[2.5e11,2.5e11,0],[2.17e11,-2.17e11,0],[-1.8510e11,1.8510e11,0],[-1.5210e11,-1.5210e11,0]]
t = 0
scale = 1e9 #meter/pixel
upscale = [5e1,5e1,5e2,5e2,5e2,5e2] #shown radius/real radius
dt = 60 #time interval
past_positions = [copy.deepcopy(position)]
mode = 1 #0 for real time, 1 for precalculated
simulation_duration = 1e8
time_multiplier = 5e6
simulation_interval = max(1,int(time_multiplier/(60*dt)))
print(f"time multiplier: {round(60*dt*simulation_interval,1)}")

def distance(coordinate1, coordinate2):
    return np.sqrt((coordinate1[0]-coordinate2[0])**2+(coordinate1[1]-coordinate2[1])**2+(coordinate1[2]-coordinate2[2])**2)

def gravity(index):
    global mass, G, position
    acceleration = [0,0,0]
    for i in range(len(mass)):
        if i != index:
            for axel in range(3):
                acceleration[axel] += -G*mass[i]/(max(distance(position[index],position[i]), radius[index]+radius[i])**3)\
                    *(position[index][axel]-position[i][axel])
    return acceleration


def approximate(delta_t):
    global mass, velocity, position, t
    for i in range(len(mass)):
        acceleration = gravity(i)
        for axel in range(3):
            velocity[i][axel] += delta_t*acceleration[axel]
            position[i][axel] += delta_t*velocity[i][axel]
    t += delta_t

def total_energy():
    energy = 0
    for i in range(len(mass)):
        energy += 1/2*mass[i]*(velocity[i][0]**2+velocity[i][1]**2+velocity[i][2]**2)
        for j in range(len(mass)-i-1):
            energy += -G*mass[i]*mass[i+j+1]/(distance(position[i],position[i+j+1]))
    return energy

def simulate(duration, delta_t):
    global t, past_positions, simulation_interval
    start_time = time.time()
    while t < duration:
        for i in range(simulation_interval):
            approximate(delta_t)
        print(f"{round(t*100/duration,1)}%", end='\r')
        past_positions += [copy.deepcopy(position)]
    print("100")
    print(f"{time.time()-start_time}s")

if mode == 1:
    start_energy = total_energy()
    simulate(simulation_duration,dt)

pygame.init()

screen = pygame.display.set_mode((800, 700))

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
    if mode == 0:
        for i in range(simulation_interval):
            approximate(dt)
        #print(total_energy())
        for i in range(len(mass)):
            pygame.draw.circle(screen, (0,0,0), (position[i][0]/scale+400,position[i][1]/scale+350), int(upscale[i]*radius[i]/scale))
    else:
        if time_step < len(past_positions) - 1:
            screen.fill((255, 255, 255))
            for i in range(len(mass)):
                pygame.draw.circle(screen, (0,0,0), (past_positions[time_step][i][0]/scale+400,past_positions[time_step][i][1]/scale+350)\
                    , int(upscale[i]*radius[i]/scale))
            time_step += 1
        else:
            print(f"delta_energy = {100*(total_energy()-start_energy)/start_energy}%")
            running = False

    pygame.display.flip()

pygame.quit()
plt.show()