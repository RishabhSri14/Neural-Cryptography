import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torchsummary import summary
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt


class Alice(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Linear(config['Data_bits']+config['Key_bits'], config['Data_bits']+ config['Key_bits']),
            nn.Tanh(),
            nn.Unflatten(1, (config['Data_bits']+config['Key_bits'], 1)),
        )
        self.RNN1 = nn.RNN(input_size=1,hidden_size=64,num_layers=2,batch_first=True)
        self.RNN2 = nn.RNN(input_size=64,hidden_size=128,num_layers=1,batch_first=True)
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.p2 = nn.Sequential(
            nn.Linear(128, 4096),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(4096,config['Data_bits']),
            )
        h0 = torch.zeros(2,config['batch_size'],64).to(self.device)
        self.hidden0=h0
        h1 = torch.zeros(1,config["batch_size"],128).to(self.device)
        self.hidden1=h1
        
    def forward(self, x):
        # x = torch.cat((x, key), dim=1)
        x = self.p1(x)
        x, _ = self.RNN1(x,self.hidden0)
        x, _ = self.RNN2(x,self.hidden1)
        x = self.p2(x[:, -1, :])
        return x
        
        
        
class Bob(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Linear(config['Cipher_bits']+config['Key_bits'], config['Cipher_bits']+ config['Key_bits']),
            nn.Tanh(),
            nn.Unflatten(1, (config['Cipher_bits']+config['Key_bits'], 1)),
        )
        self.RNN1 = nn.RNN(input_size=1,hidden_size=64,num_layers=2,batch_first=True)
        self.RNN2 = nn.RNN(input_size=64,hidden_size=128,num_layers=1,batch_first=True)
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.p2 = nn.Sequential(
            nn.Linear(128, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, config['Data_bits']),
            )
        h0 = torch.zeros(2,config['batch_size'],64).to(self.device)
        self.hidden0=h0
        h1 = torch.zeros(1,config["batch_size"],128).to(self.device)
        self.hidden1=h1
        
    def forward(self, x):
        # x = torch.cat((cipher, key), dim=1)
        x = self.p1(x)
        x, _ = self.RNN1(x,self.hidden0)
        x, _ = self.RNN2(x,self.hidden1)
        x = self.p2(x[:, -1, :])
        return x
        
        
class Eve(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Linear(config['Cipher_bits'], config['Cipher_bits']+ config['Key_bits']),
            nn.ReLU(),
            nn.Unflatten(1, (config['Cipher_bits']+config['Key_bits'], 1)),
        )
        self.RNN1 = nn.RNN(input_size=1,hidden_size=64,num_layers=2,batch_first=True)
        self.RNN2 = nn.RNN(input_size=64,hidden_size=128,num_layers=1,batch_first=True)
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.p2 = nn.Sequential(
            nn.Linear(128, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, config['Data_bits']),
            )
        h0 = torch.zeros(2,config['batch_size'],64).to(self.device)
        self.hidden0=h0
        h1 = torch.zeros(1,config["batch_size"],128).to(self.device)
        self.hidden1=h1
        
    def forward(self, x):
        # x = cipher
        x = self.p1(x)
        x, _ = self.RNN1(x,self.hidden0)
        x, _ = self.RNN2(x,self.hidden1)
        x = self.p2(x[:, -1, :])
        return x
        
        
   