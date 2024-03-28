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

def print_policy_heatmap(policy):
    for i in range(0, 64, 8):
        slice_rounded = np.around(policy[i:i+8], decimals=2)
        print(" ".join(["{:7.2f}".format(item) for item in slice_rounded]))

def time_left(args, simulation_timer, iteration, simulation_iteration, training_timer, epoch):
    simulation_time_left = simulation_timer*(args['num_iterations']*args['num_simulation_iterations']-iteration*args['num_simulation_iterations']-simulation_iteration-1+0.01)/(iteration*args['num_simulation_iterations']+simulation_iteration+1+0.01)
    training_time_left = training_timer*(args['num_iterations']*args['num_epochs']-iteration*args['num_epochs']-epoch-1+0.01)/(iteration*args['num_epochs']+epoch+1+0.01)
    return round(simulation_time_left+training_time_left, 2)

def distance(position1, position2):
    return max(np.sqrt((position1[0]-position2[0])**2+(position1[1]-position2[1])**2),1)

class Space:
    def __init__(self):
        self.gravitational_constant = 6.67384e-11
        self.row_count = 20
        self.column_count = 20
        self.action_size = self.row_count*self.column_count
        self.grid_width = 4e11
        self.meters_per_pixel = self.grid_width/self.column_count
        self.max_mass = 1e40
        self.speed_of_light = 299792458.0
        self.dimensions = 2
        self.time_step = 600

    def __repr__(self):
        return "Space"
    
    def get_initial_mass(self):
        mass = [1.9891e30,5.9e24]
        return mass
    
    def get_initial_position(self):
        position = [[2e11,2e11,0],[3.5210e11,2e11,0]]
        return position
    
    def get_initial_velocity(self):
        velocity = [[0,0,0],[0,2.929e4,0]]
        return velocity
    
    def get_initial_radius(self):   
        radius = [6.957e8,6.372e6]
        return radius
    
    def get_mass_grid(self, mass, position):
        mass_grid = np.zeros((self.row_count, self.column_count))
        for i in range(len(mass)):
            x, y = int(position[i][0]/self.meters_per_pixel),int(position[i][1]/self.meters_per_pixel)
            if x < self.column_count and y < self.row_count:
                mass_grid[y,x] += np.log(mass[i])/np.log(self.max_mass)
        
        return mass_grid
    
    def get_gravitational_field_grid(self, mass, position):
        gravitational_field_grid = np.zeros((self.row_count, self.column_count))
        for i in range(len(mass)):
            x, y = int(position[i][0]/self.meters_per_pixel),int(position[i][1]/self.meters_per_pixel)
            for grid_y in range(self.column_count):
                for grid_x in range(self.row_count):
                    gravitational_field_grid[grid_y,grid_x] += mass[i]/((self.meters_per_pixel*distance((x,y),(grid_x, grid_y)))**2)      

        for grid_y in range(self.column_count):
            for grid_x in range(self.row_count):
                gravitational_field_grid[grid_y,grid_x] = np.log(gravitational_field_grid[grid_y,grid_x])/np.log(self.max_mass/self.meters_per_pixel)
        return gravitational_field_grid
    
    def get_gravitational_field_derivative_grid(self, mass, position, velocity):
        gravitational_field_grid = np.zeros((self.row_count, self.column_count))
        for i in range(len(mass)):
            x, y = int(position[i][0]/self.meters_per_pixel),int(position[i][1]/self.meters_per_pixel)
            for grid_y in range(self.column_count):
                for grid_x in range(self.row_count):
                    gravitational_field_grid[grid_y,grid_x] += -2*np.sign(x-grid_x)*max(abs(x-grid_x),1)*velocity[i][0]+np.sign(y-grid_y)*max(abs(y-grid_y),1)*velocity[i][1]*mass[i]/((self.meters_per_pixel*distance((x,y),(grid_x, grid_y))**2)**2)      

        for grid_y in range(self.column_count):
            for grid_x in range(self.row_count):
                gravitational_field_grid[grid_y,grid_x] = (np.sign(gravitational_field_grid[grid_y,grid_x])*np.log(abs(gravitational_field_grid[grid_y,grid_x])+1)/np.log(2*self.speed_of_light*self.max_mass/self.meters_per_pixel)+1)/2
        return gravitational_field_grid
    
    def get_momentum_grid(self, mass, position, velocity):
        momentum_grid = []
        for i in range(self.dimensions):
            momentum_grid += [np.ones((self.row_count, self.column_count))/2]
            for j in range(len(mass)):
                x, y = int(position[j][0]/self.meters_per_pixel),int(position[j][1]/self.meters_per_pixel)
                if x < self.column_count and y < self.row_count:
                    momentum_grid[i][y,x] += (np.sign(velocity[j][i])*np.log10(mass[j]*abs(velocity[j][i])+1)/np.log10(self.max_mass*self.speed_of_light))/2
        
        return np.array(momentum_grid)
    
    def get_encoded_state(self, mass_grid, momentum_grid0, momentum_grid1):
        encoded_state = np.array([mass_grid, momentum_grid0, momentum_grid1]).astype(np.float32)
        
        return encoded_state
    
    def simulate_next_state(self, mass, velocity, position, radius):
        for i in range(2500):
            approximate(self.time_step, mass, velocity, position, radius, self.gravitational_constant)

    def simulate_action(self, mass, velocity, position, radius, mass_grid, momentum_grid):
        self.simulate_next_state(mass, velocity, position, radius)
        new_mass_grid = self.get_mass_grid(mass, position)
        new_momentum_grid = self.get_momentum_grid(mass, position, velocity)
        mass_action = new_mass_grid-mass_grid
        momentum_action = new_momentum_grid-momentum_grid
        return mass_action, momentum_action

class ResNet(nn.Module):
    def __init__(self, system, num_resBlocks, num_hidden, device, number_of_input_channels):
        super().__init__() #initiates the parent class
        
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(number_of_input_channels, num_hidden, kernel_size=3, padding=1), #convolutional layer(input channels, output channels, size of layer, match the input size with output size)
            nn.BatchNorm2d(num_hidden), #normalize output
            nn.ReLU()
        )
        
        #create a backbone for neural network by creating multiple Resblocks
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead_mass = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(), #takes each element from a tensor to an array
            nn.Linear(32 * system.row_count * system.column_count, system.action_size) #linear transformation(input size, output size)
        )

        self.policyHead_momentum0 = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(), #takes each element from a tensor to an array
            nn.Linear(32 * system.row_count * system.column_count, system.action_size) #linear transformation(input size, output size)
        )

        self.policyHead_momentum1 = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(), #takes each element from a tensor to an array
            nn.Linear(32 * system.row_count * system.column_count, system.action_size) #linear transformation(input size, output size)
        )
        
        self.to(device) #move to the device where you want to run the operations
    
    #iterate through the neural network
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        mass_policy = self.policyHead_mass(x)
        momentum_policy0 = self.policyHead_momentum0(x)
        momentum_policy1 = self.policyHead_momentum1(x)
        return mass_policy, momentum_policy0, momentum_policy1
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
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
        mass_grid = self.system.get_mass_grid(mass, position)
        momentum_grid = self.system.get_momentum_grid(mass, position, velocity)
        time_step_count = 0
        
        while True:
            mass_action, momentum_action = self.system.simulate_action(mass, velocity, position, radius, mass_grid, momentum_grid)
            
            memory.append((mass_grid, momentum_grid[0], momentum_grid[1], mass_action.flatten(), momentum_action[0].flatten(), momentum_action[1].flatten()))

            mass_grid = mass_grid + mass_action
            momentum_grid = momentum_grid + momentum_action
            mass_grid = np.clip(mass_grid, 0, 1)
            momentum_grid = np.clip(momentum_grid, 0, 1)

            time_step_count += 1
            
            if time_step_count >= self.args['max_time_steps']:
                returnMemory = []
                for hist_mass_grid, hist_momentum_grid0, hist_momentum_grid1, hist_mass_action, hist_momentum_action0, hist_momentum_action1 in memory:
                    returnMemory.append((
                        self.system.get_encoded_state(hist_mass_grid, hist_momentum_grid0, hist_momentum_grid1),
                        hist_mass_action,
                        hist_momentum_action0,
                        hist_momentum_action1
                    ))
                return returnMemory
                
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, mass_policy_targets, momentum_policy_targets0, momentum_policy_targets1 = zip(*sample)
            
            state, mass_policy_targets, momentum_policy_targets0, momentum_policy_targets1 = np.array(state), np.array(mass_policy_targets), np.array(momentum_policy_targets0), np.array(momentum_policy_targets1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            mass_policy_targets = torch.tensor(mass_policy_targets, dtype=torch.float32, device=self.model.device)
            momentum_policy_targets0 = torch.tensor(momentum_policy_targets0, dtype=torch.float32, device=self.model.device)
            momentum_policy_targets1 = torch.tensor(momentum_policy_targets1, dtype=torch.float32, device=self.model.device)
            
            out_mass_policy, out_momentum_policy0, out_momentum_policy1 = self.model(state)
            mass_squared_difference = (mass_policy_targets-out_mass_policy) ** 2
            momentum_squared_difference0 = (momentum_policy_targets0-out_momentum_policy0) ** 2
            momentum_squared_difference1 = (momentum_policy_targets1-out_momentum_policy1) ** 2

            loss = mass_squared_difference.sum() + momentum_squared_difference0.sum() + momentum_squared_difference1.sum()
            
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
                print(f"{100*(epoch+1)/self.args['num_epochs']}%, estimated time left: {time_left(self.args, simulation_timer, iteration, 0, training_timer, epoch)}s")
            
            torch.save(self.model.state_dict(), f"./Gravity/models/model_{iteration}_{self.system}.pt")
            #torch.save(self.optimizer.state_dict(), f"./Gravity/models/optimizer_{iteration}_{self.system}.pt")

def learn(args, system):
    model = ResNet(system, 4, 256, device=device, number_of_input_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    gravityai = GravityAI(model, optimizer, system, args)
    start_time = time.time()
    gravityai.learn()
    print(f"learning time: {time.time()-start_time}s")

@torch.no_grad()
def play(args, system, model_dict, mass_grid, momentum_grid):
    model = ResNet(system, 4, 256, device=device, number_of_input_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #load previously learned values
    model.load_state_dict(torch.load(model_dict, map_location=device))
    model.eval() #playing mode

    mass_action_probs, momentum_action_probs0, momentum_action_probs1 = model(
        torch.tensor(system.get_encoded_state(mass_grid, momentum_grid[0], momentum_grid[1]), device=model.device).unsqueeze(0)
    )
    mass_action_probs = mass_action_probs.squeeze(0).cpu().numpy()
    momentum_action_probs0 = momentum_action_probs0.squeeze(0).cpu().numpy()
    momentum_action_probs1 = momentum_action_probs1.squeeze(0).cpu().numpy()
    mass_action = mass_action_probs.reshape(mass_grid.shape)
    momentum_action0 = momentum_action_probs0.reshape(momentum_grid[0].shape)
    momentum_action1 = momentum_action_probs1.reshape(momentum_grid[1].shape)
    
    mass_grid = mass_grid + mass_action
    momentum_grid = momentum_grid + np.array([momentum_action0, momentum_action1])
    mass_grid = np.clip(mass_grid, 0, 1)
    momentum_grid = np.clip(momentum_grid, 0, 1)

    return mass_grid, momentum_grid