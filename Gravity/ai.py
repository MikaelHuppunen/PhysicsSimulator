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
        slice_rounded = np.around(policy[i:i+8], decimals=2)
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
        self.row_count = 20
        self.column_count = 20
        self.action_size = 12
        self.grid_width = 4e11
        self.meters_per_pixel = self.grid_width/self.column_count
        self.max_mass = 1e40
        self.max_distance = 1e20
        self.speed_of_light = 299792458.0
        self.dimensions = 2
        self.time_step = 600

    def __repr__(self):
        return "Space"
    
    def get_initial_mass(self):
        mass = [1.9891e30, 5.9e24]
        return mass
    
    def get_initial_position(self):
        position = [[0,0,0],[1.5210e11,0,0]]
        return position
    
    def get_initial_velocity(self):
        velocity = [[0,0,0],[0,2.929e4,0]]
        return velocity
    
    def get_initial_radius(self):   
        radius = [6.957e8,6.372e6]
        return radius
    
    def normalize_mass(self, mass):
        normalized_mass = np.zeros(len(mass))
        for i in range(len(mass)):
            normalized_mass[i] = np.log(mass[i]+1)/np.log(self.max_mass)
        return normalized_mass
    
    def normalize_position(self, position):
        normalized_position = np.zeros(len(position)*len(position[0]))
        for i in range(len(position)):
            for j in range(len(position[0])):
                normalized_position[len(position[0])*i+j] = np.sign(position[i][j])*np.log(abs(position[i][j])+1)/np.log(self.max_distance)
        return normalized_position
    
    def normalize_velocity(self, velocity):
        normalized_velocity = np.zeros(len(velocity)*len(velocity[0]))
        for i in range(len(velocity)):
            for j in range(len(velocity[0])):
                normalized_velocity[len(velocity[0])*i+j] = np.sign(velocity[i][j])*np.log(abs(velocity[i][j])+1)/np.log(self.speed_of_light)
        return normalized_velocity

    def get_encoded_state(self, mass, velocity, position):
        normalized_mass = self.normalize_mass(mass)
        normalized_position = self.normalize_position(position)
        normalized_velocity = self.normalize_velocity(velocity)

        encoded_state = np.concatenate((normalized_mass, normalized_position, normalized_velocity)).astype(np.float32)
        return encoded_state
    
    def simulate_next_state(self, mass, velocity, position, radius):
        for i in range(600):
            approximate(self.time_step, mass, velocity, position, radius, self.gravitational_constant)

    def simulate_action(self, mass, velocity, position, radius):
        old_velocity = deepcopy(velocity)
        old_position = deepcopy(position)
        self.simulate_next_state(mass, velocity, position, radius)
        position_action = (self.normalize_position(position)-self.normalize_position(old_position))
        velocity_action = (self.normalize_velocity(velocity)-self.normalize_velocity(old_velocity))
        action = np.concatenate((position_action,velocity_action))
        return action
    
    def get_next_state(self, velocity, position, action):
        position_action = action[0:6]
        velocity_action = action[6:12]
        position_action = position_action.reshape((2,3))
        velocity_action = velocity_action.reshape((2,3))
        return position+position_action, velocity+velocity_action
    
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
        mass = self.system.get_initial_mass()
        position = self.system.get_initial_position()
        velocity = self.system.get_initial_velocity()
        radius = self.system.get_initial_radius()
        time_step_count = 0
        
        while True:
            action = self.system.simulate_action(mass, velocity, position, radius)
            
            memory.append((mass, velocity, position, action))

            time_step_count += 1
            
            if time_step_count >= self.args['max_time_steps']:
                returnMemory = []
                for hist_mass, hist_velocity, hist_position, hist_action in memory:
                    returnMemory.append((
                        self.system.get_encoded_state(hist_mass, hist_velocity, hist_position),
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
            print(policy_targets)

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
    model = ResNet(system, 8, 64, device=device, number_of_inputs=14)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    gravityai = GravityAI(model, optimizer, system, args)
    start_time = time.time()
    gravityai.learn()
    print(f"learning time: {time.time()-start_time}s")

@torch.no_grad()
def play(args, system, model_dict, mass, velocity, position):
    model = ResNet(system, 8, 64, device=device, number_of_inputs=14)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #load previously learned values
    model.load_state_dict(torch.load(model_dict, map_location=device))
    model.eval() #playing mode

    action_probs = model(
        torch.tensor(system.get_encoded_state(mass, velocity, position), device=model.device).unsqueeze(0)
    )
    action = action_probs.squeeze(0).cpu().numpy()
    #action = action_probs.reshape(state.shape)
    
    velocity, position = system.get_next_state(velocity, position, action)

    return velocity, position