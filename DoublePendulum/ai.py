from doublependulum import all_angle_approximate as approximate
import numpy as np

print(np.__version__)

import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0) #set the same seed for pytorch every time to ensure reproducibility

import random
import math
import time

def print_policy_heatmap(policy):
    for i in range(0, 64, 8):
        slice_rounded = np.around(policy[i:i+8], decimals=2)
        print(" ".join(["{:7.2f}".format(item) for item in slice_rounded]))

class DoublePendulum:
    def __init__(self):
        self.g = 9.81
        self.l = 1 #used when l1 == l2
        self.length1 = 1
        self.length2 = 1
        self.mass1 = 3
        self.mass2 = 1
        self.row_count = 2
        self.column_count = 2
        self.action_size = self.row_count*self.column_count
        self.max_time_steps = 1000
        self.max_angular_speed = 10000

    def __repr__(self):
        return "DoublePendulum"
    
    def get_initial_state(self):
        return np.array([[1.3,1.1],[5.0,5.0]])
    
    def get_next_state(self, state, action):
        state = state+action
        state[0,0] = (state[0,0] + np.pi) % (2 * np.pi) - np.pi
        state[0,1] = (state[0,1] + np.pi) % (2 * np.pi) - np.pi
        #limit the angular speeds from growing too much
        if state[1,0] > self.max_angular_speed:
            state[1,0] = self.max_angular_speed
        if state[1,0] < -self.max_angular_speed:
            state[1,0] = -self.max_angular_speed
        if state[1,1] > self.max_angular_speed:
            state[1,1] = self.max_angular_speed
        if state[1,1] < -self.max_angular_speed:
            state[1,1] = -self.max_angular_speed
            
        return state
    
    def get_encoded_state(self, state):
        encoded_state = np.array([state]).astype(np.float32)
        
        return encoded_state
    
    def is_terminal(self, time_step):
        return (time_step >= self.max_time_steps)
    
    def simulate_action(self, state):
        theta_1, theta_2, dot_theta_1, dot_theta_2 = state[0,0], state[0,1], state[1,0], state[1,1]
        for i in range(10):
            theta_1, theta_2, dot_theta_1, dot_theta_2, _ = approximate(theta_1, theta_2, 0.0001, dot_theta_1, dot_theta_2, 0, self.mass1, self.mass2)
        return np.array([[theta_1, theta_2],[dot_theta_1, dot_theta_2]])-state

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
        
        self.policyHead = nn.Sequential(
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
        policy = self.policyHead(x)
        return policy
        
        
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
    
class AIPendulum:
    def __init__(self, model, optimizer, system, args):
        self.model = model
        self.optimizer = optimizer
        self.system = system
        self.args = args

    @torch.no_grad() 
    def selfPlay(self):
        memory = []
        state = self.system.get_initial_state()
        move_count = 0
        
        while True:
            action = self.system.simulate_action(state)
            action_probs = action.flatten()
            
            memory.append((state, action_probs))
            
            state = self.system.get_next_state(state, action)

            move_count += 1
            
            is_terminal = self.system.is_terminal(move_count)
            
            if is_terminal:
                returnMemory = []
                for hist_state, hist_action_probs in memory:
                    returnMemory.append((
                        self.system.get_encoded_state(hist_state),
                        hist_action_probs
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
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()
                print(f"{iteration+1}/{self.args['num_iterations']}: {100*(selfPlay_iteration+1)/self.args['num_selfPlay_iterations']}%, estimated time left: {round((time.time()-start)*(self.args['num_iterations']*self.args['num_selfPlay_iterations']-iteration*self.args['num_selfPlay_iterations']-selfPlay_iteration-1+0.01)/(iteration*self.args['num_selfPlay_iterations']+selfPlay_iteration+1+0.01),2)}s")
                
            self.model.train()
            for epoch in range(self.args['num_epochs']):
                self.train(memory)
                print(f"{100*(epoch+1)/self.args['num_epochs']}%")
            
            torch.save(self.model.state_dict(), f"./DoublePendulum/models/model_{iteration}_{self.system}.pt")
            torch.save(self.optimizer.state_dict(), f"./DoublePendulum/models/optimizer_{iteration}_{self.system}.pt")

def learn(args, system):
    model = ResNet(system, 4, 64, device=device, number_of_input_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    aipendulum = AIPendulum(model, optimizer, system, args)
    start_time = time.time()
    aipendulum.learn()
    print(f"learning time: {time.time()-start_time}s")

@torch.no_grad()
def play(args, system, model_dict, state):
    model = ResNet(system, 4, 64, device=device, number_of_input_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #load previously learned values
    model.load_state_dict(torch.load(model_dict, map_location=device))
    model.eval() #playing mode

    action_probs = model(
        torch.tensor(system.get_encoded_state(state), device=model.device).unsqueeze(0)
    )
    action_probs = action_probs.squeeze(0).cpu().numpy()
    action = action_probs.reshape(state.shape)
    
    state = system.get_next_state(state, action)

    return state