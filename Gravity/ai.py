import numpy as np
import random

print(np.__version__)

import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0) #set the same seed for pytorch every time to ensure reproducibility

import random
import math
import time
from gravity import approximate
from copy import copy, deepcopy

def print_policy_heatmap(policy):
    for i in range(0, 64, 8):
        slice_rounded = np.around(policy[i:i+8], decimals=8)
        print(" ".join(["{:7.2f}".format(item) for item in slice_rounded]))

def time_left(args, simulation_timer, iteration, simulation_iteration, training_timer, epoch):
    simulation_time_left = simulation_timer*(args['num_iterations']*args['num_simulation_iterations']-iteration*args['num_simulation_iterations']-simulation_iteration-1+0.01)/(iteration*args['num_simulation_iterations']+simulation_iteration+1+0.01)
    training_time_left = training_timer*(args['num_iterations']*args['num_epochs']-iteration*args['num_epochs']-epoch-1+0.01)/(iteration*args['num_epochs']+epoch+1+0.01)
    seconds = simulation_time_left+training_time_left
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return '{:02d}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))

def distance(position1, position2):
    return max(np.sqrt((position1[0]-position2[0])**2+(position1[1]-position2[1])**2),1)

class Space:
    def __init__(self):
        self.gravitational_constant = 6.67384e-11
        self.action_size = 8
        self.scale = 5e8
        self.max_mass = 1e40
        self.max_distance = 1e20
        self.speed_of_light = 299792458.0
        self.dimensions = 2
        self.time_step = 600
        self.time_steps_per_step = 600

    def __repr__(self):
        return "Space"
    
    def get_initial_mass(self):
        mass = [1.9891e30, 5.9e24]
        return mass
    
    def get_initial_position(self, angle):
        position = [[-np.cos(angle)*4.51e5,-np.sin(angle)*4.51e5,0]]
        position += [[np.cos(angle)*1.5210e11,np.sin(angle)*1.5210e11,0]]
        return position
    
    def get_initial_velocity(self, angle):
        velocity = [[-np.cos(angle)*8.69e-2,-np.sin(angle)*8.69e-2,0]]
        velocity += [[np.cos(angle)*2.929e4,np.sin(angle)*2.929e4,0]]
        return velocity
    
    def get_initial_radius(self):   
        radius = [6.957e8,6.372e6]
        return radius
    
    def get_initial_state(self):
        random_angle = random.uniform(-math.pi, math.pi)
        mass = self.get_initial_mass()
        position = self.get_initial_position(random_angle)
        velocity = self.get_initial_velocity(random_angle+np.pi/2)
        radius = self.get_initial_radius()

        return mass, position, velocity, radius
    
    def normalize_mass(self, mass):
        normalized_mass = np.zeros(len(mass))
        for i in range(len(mass)):
            normalized_mass[i] = np.log(mass[i]+1)/np.log(self.max_mass)
        return normalized_mass
    
    def normalize_distance(self, position):
        normalized_distance = np.zeros(len(position))
        for i in range(len(position)):
            normalized_distance[i] = distance([0,0,0], position[i])/self.scale
        return normalized_distance
    
    def normalize_velocity(self, velocity):
        normalized_velocity = np.zeros(len(velocity)*len(velocity[0]))
        for i in range(len(velocity)):
            for j in range(len(velocity[0])):
                normalized_velocity[len(velocity[0])*i+j] = np.sign(velocity[i][j])*np.log(abs(velocity[i][j])+1)/np.log(self.speed_of_light)
        return normalized_velocity

    def get_encoded_state(self, position, velocity):
        normalized_distance = self.normalize_distance(position)
        radial_velocity = np.array(self.get_radial_velocity(position, velocity))
        angular_velocity = np.array(self.get_angular_velocity(position, velocity))

        encoded_state = np.concatenate((normalized_distance,np.array(self.get_angles(position)),radial_velocity,angular_velocity)).astype(np.float32)
        return encoded_state
    
    def simulate_next_state(self, mass, velocity, position, radius):
        for i in range(self.time_steps_per_step):
            approximate(self.time_step, mass, velocity, position, radius, self.gravitational_constant)

    def simulate_action(self, mass, velocity, position, radius):
        old_distances = self.normalize_distance(position)
        old_angles = self.get_angles(position)
        old_radial_velocity = self.get_radial_velocity(position, velocity)
        old_angular_velocity = self.get_angular_velocity(position, velocity)

        self.simulate_next_state(mass, velocity, position, radius)

        angles = self.get_angles(position)
        distances = self.normalize_distance(position)
        radial_velocity = self.get_radial_velocity(position, velocity)
        angular_velocity = self.get_angular_velocity(position, velocity)

        distance_action = np.array(distances)-np.array(old_distances)
        angle_action = np.array(angles)-np.array(old_angles)
        radial_velocity_action = np.array(radial_velocity)-np.array(old_radial_velocity)
        angular_velocity_action = np.array(angular_velocity)-np.array(old_angular_velocity)

        angle_action = (angle_action + np.pi) % (2 * np.pi) - np.pi

        action = np.concatenate((distance_action, angle_action, radial_velocity_action, angular_velocity_action))
        return action
    
    def get_next_state(self, velocity, position, action):
        distance_action = action[0:2]
        angle_action = action[2:4]
        radial_velocity_action = action[4:6]
        angular_velocity_action = action[6:8]

        angles = self.get_angles(position)
        radial_velocities = self.get_radial_velocity(position, velocity)
        angular_velocities = self.get_angular_velocity(position, velocity)
        for i in range(len(position)):
            new_angle = angles[i]+angle_action[i]
            radial_velocity = self.scale*(radial_velocities[i]+radial_velocity_action[i])/(self.time_step*self.time_steps_per_step)
            angular_velocity = (angular_velocities[i]+angular_velocity_action[i])/(self.time_step*self.time_steps_per_step)
            
            distance_to_origin = distance([0,0,0], position[i])
            distance_to_origin += distance_action[i]*self.scale

            position[i][0] = distance_to_origin*np.cos(new_angle)
            position[i][1] = distance_to_origin*np.sin(new_angle)
            velocity[i][0] = radial_velocity*np.cos(new_angle)-distance_to_origin*angular_velocity*np.sin(new_angle)
            velocity[i][1] = radial_velocity*np.sin(new_angle)+distance_to_origin*angular_velocity*np.cos(new_angle)
        
        return position, velocity
    
    def get_angles(self, position):
        angles = []
        for i in range(len(position)):
            angles += [np.arctan2(position[i][1], np.sign(position[i][0])*max(abs(position[i][0])+1,1))]
        return angles
    
    def get_angular_velocity(self, position, velocity):
        angular_velocities = []
        for i in range(len(position)):
            angular_velocities += [self.time_step*self.time_steps_per_step*(position[i][0]*velocity[i][1]-position[i][1]*velocity[i][0])/(np.linalg.norm(position[i])**2)]
        
        return angular_velocities
    
    def get_radial_velocity(self, position, velocity):
        radial_velocities = []
        for i in range(len(position)):
            radial_velocities += [self.time_step*self.time_steps_per_step*np.dot(velocity[i], position[i])/(self.scale*np.linalg.norm(position[i]))]
        
        return radial_velocities

class ResNet(nn.Module):
    def __init__(self, system, num_resBlocks, num_hidden, device, number_of_inputs):
        super().__init__() #initiates the parent class
        
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Linear(number_of_inputs, num_hidden),
            nn.BatchNorm1d(num_hidden), #normalize output
            nn.ReLU()
        )
        
        #create a backbone for neural network by creating multiple Resblocks
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Linear(num_hidden, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, system.action_size) #linear transformation(input size, output size)
        )
        
        self.to(device) #move to the device where you want to run the operations
    
    #iterate through the neural network
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        return policy

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.fc = nn.Linear(num_hidden, num_hidden)
        self.bn = nn.BatchNorm1d(num_hidden)
        
    def forward(self, x):
        x = F.relu(self.bn(self.fc(x)))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class GravityAI:
    def __init__(self, model, optimizer, system, args):
        self.model = model
        self.optimizer = optimizer
        self.system = system
        self.args = args

    @torch.no_grad() 
    def simulation(self):
        memory = []
        mass, position, velocity, radius = self.system.get_initial_state()
        time_step_count = 0
        
        while True:
            action = self.system.simulate_action(mass, velocity, position, radius)
            
            memory.append((mass, velocity, position, action))

            time_step_count += 1
            
            if time_step_count >= self.args['max_time_steps']:
                returnMemory = []
                for hist_mass, hist_velocity, hist_position, hist_action in memory:
                    returnMemory.append((
                        self.system.get_encoded_state(hist_position, hist_velocity),
                        hist_action
                    ))
                return returnMemory
                
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets = zip(*sample)
            
            state, policy_targets = np.array(state), np.array(policy_targets)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)

            out_policy = self.model(state)
            squared_difference = (policy_targets-out_policy) ** 2
            loss = squared_difference.sum()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(loss.item())
    
    def learn(self):
        start = time.time()
        simulation_timer = 0.0
        training_timer = 0.0
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for simulation_iteration in range(self.args['num_simulation_iterations']):
                simulation_start = time.time()
                memory += self.simulation()
                simulation_timer += time.time()-simulation_start
                print(f"{iteration+1}/{self.args['num_iterations']}: {100*(simulation_iteration+1)/self.args['num_simulation_iterations']}%, estimated time left: {time_left(self.args, simulation_timer, iteration, simulation_iteration, training_timer, 0)}s")
                
            self.model.train()
            for epoch in range(self.args['num_epochs']):
                training_start = time.time()
                self.train(memory)
                training_timer += time.time()-training_start
                print(f"{100*(epoch+1)/self.args['num_epochs']}%, estimated time left: {time_left(self.args, simulation_timer, iteration, self.args['num_simulation_iterations']-1, training_timer, epoch)}s")
            
            torch.save(self.model.state_dict(), f"./Gravity/models/model_{iteration}_{self.system}.pt")
            #torch.save(self.optimizer.state_dict(), f"./Gravity/models/optimizer_{iteration}_{self.system}.pt")

def learn(args, system):
    model = ResNet(system, 8, 64, device=device, number_of_inputs=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    gravityai = GravityAI(model, optimizer, system, args)
    start_time = time.time()
    gravityai.learn()
    print(f"learning time: {time.time()-start_time}s")

@torch.no_grad()
def play(args, system, model_dict, mass, velocity, position):
    model = ResNet(system, 8, 64, device=device, number_of_inputs=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #load previously learned values
    model.load_state_dict(torch.load(model_dict, map_location=device))
    model.eval() #playing mode

    action_probs = model(
        torch.tensor(system.get_encoded_state(position, velocity), device=model.device).unsqueeze(0)
    )
    action = action_probs.squeeze(0).cpu().numpy()
    #action = action_probs.reshape(state.shape)
    
    position, velocity = system.get_next_state(velocity, position, action)

    return position, velocity