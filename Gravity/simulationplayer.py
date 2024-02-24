import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
import time
import copy
import json

file_path = "./Gravity/simulation2.txt"

with open(file_path, 'r') as file:
    past_positions = json.loads(file.read())

radius = [6.957e8, 6.957e8,6.372e6,6.957e8,6.372e6,6.372e6,6.372e6,6.372e6,6.372e6,6.372e6,6.372e6]
scale = 2e9 #meter/pixel
upscale = [1e1,1e1,4e2,1e1,4e2,4e2,4e2,4e2,4e2,4e2,4e2] #shown radius/real radius

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
    if time_step < len(past_positions) - 1:
        screen.fill((255, 255, 255))
        for i in range(len(radius)):
            pygame.draw.circle(screen, (0,0,0), (past_positions[time_step][i][0]/scale+400,past_positions[time_step][i][1]/scale+350)\
                , int(upscale[i]*radius[i]/scale))
        time_step += 1
    else:
        running = False

    pygame.display.flip()

pygame.quit()