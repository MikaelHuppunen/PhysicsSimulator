import numpy as np
import pygame
from pygame.locals import *

g = 9.81
l = 1 #used when l1 == l2
l1 = 1
l2 = 1
m1 = 3
m2 = 1
theta_1 = 1.3
theta_2 = 1.1
dot_theta_1 = 5
dot_theta_2 = 5
t = 0
x0 = 400
y0 = 350
l_multiplier = 300/(l1+l2)
mode = 1 #0 for small angle, 1 for all angle
plotmode = 1 #0 for over time and 1 for parametric

def small_angle_approximate(angle1, angle2, delta_t):
    global dot_theta_1, dot_theta_2, theta_1, theta_2, t
    ddot_angle1 = g/(3*l)*angle2-4*g/(3*l)*angle1
    ddot_angle2 = 4*g/(3*l)*angle1-4*g/(3*l)*angle2
    dot_theta_1 += delta_t*ddot_angle1
    dot_theta_2 += delta_t*ddot_angle2
    theta_1 += delta_t*dot_theta_1
    theta_2 += delta_t*dot_theta_2
    if theta_1 > np.pi:
        theta_1 -= 2*np.pi
    if theta_2 > np.pi:
        theta_2 -= 2*np.pi
    if theta_1 < -np.pi:
        theta_1 += 2*np.pi
    if theta_2 < -np.pi:
        theta_2 += 2*np.pi
    t += delta_t

def all_angle_approximate(angle1, angle2, delta_t, dot_theta_1, dot_theta_2, t, m1, m2):
    ddot_angle1 = -(g*(m1*np.sin(angle1)-m2*(np.cos(angle1-angle2)*np.sin(angle2)-np.sin(angle1)))+(l1*np.cos(angle1-angle2)*dot_theta_1**2+l2*dot_theta_2**2)*m2*np.sin(angle1-angle2))/(l1*(m1+m2*(np.sin(angle1-angle2))**2))
    ddot_angle2 = (g*(m1+m2)*(np.sin(angle1)*np.cos(angle1-angle2)-np.sin(angle2))+(l1*(m1+m2)*dot_theta_1**2+l2*m2*np.cos(angle1-angle2)*dot_theta_2**2)*np.sin(angle1-angle2))/(l2*(m1+m2*(np.sin(angle1-angle2))**2))
    dot_theta_1 += delta_t*ddot_angle1
    dot_theta_2 += delta_t*ddot_angle2
    angle1 += delta_t*dot_theta_1
    angle2 += delta_t*dot_theta_2
    return [angle1, angle2, dot_theta_1, dot_theta_2]

def approximate(angle1, angle2, delta_t, dot_theta_1, dot_theta_2, time, m1, m2, mode = 1):
    if mode == 0:
        small_angle_approximate(angle1, angle2, delta_t)
    if mode == 1:
        angle1, angle2, dot_theta_1, dot_theta_2 = all_angle_approximate(angle1, angle2, delta_t, dot_theta_1, dot_theta_2, t, m1, m2)
    if angle1 > np.pi:
        angle1 -= 2*np.pi
    if angle2 > np.pi:
        angle2 -= 2*np.pi
    if angle1 < -np.pi:
        angle1 += 2*np.pi
    if angle2 < -np.pi:
        angle2 += 2*np.pi
    time += delta_t

    return [angle1, angle2, dot_theta_1, dot_theta_2, time]

if __name__ == "__main__":
    pygame.init()

    screen = pygame.display.set_mode((800, 700))

    running = True
    while running:
        for i in range(10):
            theta_1, theta_2, dot_theta_1, dot_theta_2, t = approximate(theta_1, theta_2, 0.0001, mode, dot_theta_1, dot_theta_2, t, m1, m2)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        screen.fill((255, 255, 255))
        pos1 = (np.cos(theta_1-np.pi/2)*l1*l_multiplier+x0, -np.sin(theta_1-np.pi/2)*l1*l_multiplier+y0)
        pygame.draw.line(screen, (0,0,0), (x0, y0), pos1, 2)
        pygame.draw.circle(screen, (0,0,0), pos1, int(15*m1**(1/3)))
        pos2 = (np.cos(theta_2-np.pi/2)*l2*l_multiplier+np.cos(theta_1-np.pi/2)*l1*l_multiplier+x0\
                , -np.sin(theta_2-np.pi/2)*l2*l_multiplier-np.sin(theta_1-np.pi/2)*l1*l_multiplier+y0)
        pygame.draw.line(screen, (0,0,0), pos1, pos2, 2)
        pygame.draw.circle(screen, (0,0,0), pos2, int(15*m2**(1/3)))

        pygame.display.flip()

    pygame.quit()