import pygame
import numpy as np
from doublependulum import all_angle_approximate as approximate
from ai import *

theta_1 = 1.3
theta_2 = 1.1
dot_theta_1 = 5
dot_theta_2 = 5

doublependulum = DoublePendulum()

time = 0
x0 = 400
y0 = 350
l_multiplier = 300/(doublependulum.length1+doublependulum.length2)

args = {
    'C': 2,
    'num_searches': 5000,
    'dirichlet_epsilon': 0.1,
    'dirichlet_alpha': 0.3,
    'search': True,
    'temperature': 0,
}
model_dict = "./DoublePendulum/models/model_7_DoublePendulum.pt"

pygame.init()

screen = pygame.display.set_mode((800, 700))
state = doublependulum.get_initial_state()

running = True
for aaaaaaaaaaaaaaaa in range(1000):
    ptheta_1, ptheta_2, pdot_theta_1, pdot_theta_2 = theta_1, theta_2, dot_theta_1, dot_theta_2
    for i in range(10):
        theta_1, theta_2, dot_theta_1, dot_theta_2, time = approximate(theta_1, theta_2, 0.0001, dot_theta_1, dot_theta_2, time, doublependulum.mass1, doublependulum.mass2)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

    state = play(args, doublependulum, model_dict, state)
    screen.fill((255, 255, 255))
    ai_theta_1, ai_theta_2, ai_dot_theta_1, ai_dot_theta_2 = state[0,0], state[0,1], state[1,0], state[1,1]

    pos1 = (np.cos(theta_1-np.pi/2)*doublependulum.length1*l_multiplier+x0, -np.sin(theta_1-np.pi/2)*doublependulum.length1*l_multiplier+y0)
    pygame.draw.line(screen, (0,0,0), (x0, y0), pos1, 2)
    pygame.draw.circle(screen, (0,0,0), pos1, int(15*doublependulum.mass1**(1/3)))
    pos2 = (np.cos(theta_2-np.pi/2)*doublependulum.length2*l_multiplier+np.cos(theta_1-np.pi/2)*doublependulum.length1*l_multiplier+x0\
            , -np.sin(theta_2-np.pi/2)*doublependulum.length2*l_multiplier-np.sin(theta_1-np.pi/2)*doublependulum.length1*l_multiplier+y0)
    pygame.draw.line(screen, (0,0,0), pos1, pos2, 2)
    pygame.draw.circle(screen, (0,0,0), pos2, int(15*doublependulum.mass2**(1/3)))

    ai_pos1 = (np.cos(ai_theta_1-np.pi/2)*doublependulum.length1*l_multiplier+x0, -np.sin(ai_theta_1-np.pi/2)*doublependulum.length1*l_multiplier+y0)
    pygame.draw.line(screen, (255,0,0), (x0, y0), ai_pos1, 2)
    pygame.draw.circle(screen, (255,0,0), ai_pos1, int(15*doublependulum.mass1**(1/3)))
    ai_pos2 = (np.cos(ai_theta_2-np.pi/2)*doublependulum.length2*l_multiplier+np.cos(ai_theta_1-np.pi/2)*doublependulum.length1*l_multiplier+x0\
            , -np.sin(ai_theta_2-np.pi/2)*doublependulum.length2*l_multiplier-np.sin(ai_theta_1-np.pi/2)*doublependulum.length1*l_multiplier+y0)
    pygame.draw.line(screen, (255,0,0), ai_pos1, ai_pos2, 2)
    pygame.draw.circle(screen, (255,0,0), ai_pos2, int(15*doublependulum.mass2**(1/3)))

    pygame.display.flip()

pygame.quit()