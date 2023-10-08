import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *

G = 6.67384e-11
mass = [1.9891e30, 5.9e24]
radius = [6.957e8, 6.372e6]
velocity = [[0,0,0],[0,2.929e4,0]]
position = [[0,0,0],[1.5210e11,0,0]]
time = 0
scale = 1e9 #meter/pixel
upscale = [5e1,1e3] #shown radius/real radius
dt = 60 #time interval
n_interval = 1000 #time interval/frame

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
    global mass, velocity, position, time
    for i in range(len(mass)):
        acceleration = gravity(i)
        for axel in range(3):
            velocity[i][axel] += delta_t*acceleration[axel]
            position[i][axel] += delta_t*velocity[i][axel]
    time += delta_t

def total_energy():
    energy = 0
    for i in range(len(mass)):
        energy += 1/2*mass[i]*(velocity[i][0]**2+velocity[i][1]**2+velocity[i][2]**2)
        for j in range(len(mass)-i-1):
            energy += -G*mass[i]*mass[i+j+1]/(distance(position[i],position[i+j+1]))
    return energy

pygame.init()

screen = pygame.display.set_mode((800, 700))

running = True
while running:
    for i in range(n_interval):
        approximate(dt)
    print(total_energy())
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
    screen.fill((255, 255, 255))
    for i in range(len(mass)):
        pygame.draw.circle(screen, (0,0,0), (position[i][0]/scale+400,position[i][1]/scale+350), int(upscale[i]*radius[i]/scale))

    pygame.display.flip()

pygame.quit()
plt.show()