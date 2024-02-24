import pygame
import numpy as np
from doublependulum import approximate

class DoublePendulum:
    def __init__(self): #run when class is initiated
        self.g = 9.81
        self.l = 1 #used when length1 == length2
        self.length1 = 1
        self.length2 = 1
        self.mass1 = 3
        self.mass2 = 1
        self.theta_1 = 1.3
        self.theta_2 = 1.1
        self.dot_theta_1 = 5
        self.dot_theta_2 = 5

doublependulum = DoublePendulum()

mode = 1
time = 0
x0 = 400
y0 = 350
l_multiplier = 300/(doublependulum.length1+doublependulum.length2)

pygame.init()

screen = pygame.display.set_mode((800, 700))

running = True
while running:
    for i in range(10):
        doublependulum.theta_1, doublependulum.theta_2, doublependulum.dot_theta_1, doublependulum.dot_theta_2, time = approximate(doublependulum.theta_1, doublependulum.theta_2, 0.0001, mode, doublependulum.dot_theta_1, doublependulum.dot_theta_2, time, doublependulum.mass1, doublependulum.mass2)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
    screen.fill((255, 255, 255))
    pos1 = (np.cos(doublependulum.theta_1-np.pi/2)*doublependulum.length1*l_multiplier+x0, -np.sin(doublependulum.theta_1-np.pi/2)*doublependulum.length1*l_multiplier+y0)
    pygame.draw.line(screen, (0,0,0), (x0, y0), pos1, 2)
    pygame.draw.circle(screen, (0,0,0), pos1, int(15*doublependulum.mass1**(1/3)))
    pos2 = (np.cos(doublependulum.theta_2-np.pi/2)*doublependulum.length2*l_multiplier+np.cos(doublependulum.theta_1-np.pi/2)*doublependulum.length1*l_multiplier+x0\
            , -np.sin(doublependulum.theta_2-np.pi/2)*doublependulum.length2*l_multiplier-np.sin(doublependulum.theta_1-np.pi/2)*doublependulum.length1*l_multiplier+y0)
    pygame.draw.line(screen, (0,0,0), pos1, pos2, 2)
    pygame.draw.circle(screen, (0,0,0), pos2, int(15*doublependulum.mass2**(1/3)))

    pygame.display.flip()

pygame.quit()