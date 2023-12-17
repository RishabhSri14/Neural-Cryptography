import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

with open("config.yaml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)
name = config['base_model']

if name=='LSTM':
    from model_LSTM import Alice, Bob, Eve
if name=='RNN':
    from model_RNN import Alice, Bob, Eve
if name=='GRU':
    from model_GRU import Alice, Bob, Eve
wandb.init(project="SMAI_Project",entity="rishabhsri14", name=name+f"epochs={config['epochs']}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
number_of_batches = 2**config['Data_bits']/config['batch_size']

alice = Alice(config).to(device)
bob = Bob(config).to(device)
eve = Eve(config).to(device)

AB_loss = []
B_loss = []
E_loss = []
# ABE_optimizer = optim.Adam(list(alice.parameters())+list(bob.parameters())+list(eve.parameters()), lr=config['lr'])
# A_optimizer = optim.Adam(alice.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
# B_optimizer = optim.Adam(bob.parameters(), lr=config['lr'],weight_decay=config['weight_decay'])
# E_optimizer = optim.Adam(eve.parameters(), lr=config['lr'],weight_decay=config['weight_decay'])

# ABE_optimizer = optim.RMSprop(list(alice.parameters())+list(bob.parameters()), lr=config['lr'])
A_optimizer = optim.RMSprop(alice.parameters(), lr=config['lr'])
B_optimizer = optim.RMSprop(bob.parameters(), lr=config['lr'])
E_optimizer = optim.RMSprop(eve.parameters(), lr=config['lr'])

# ABE_optimizer = optim.SGD(list(alice.parameters())+list(bob.parameters())+list(eve.parameters()), lr=0.001, momentum=0.9)
# A_optimizer = optim.SGD(alice.parameters(), lr=config['lr'],momentum=0.9)
# B_optimizer = optim.SGD(bob.parameters(), lr=config['lr'], momentum=0.9)
# E_optimizer = optim.SGD(eve.parameters(), lr=config['lr'], momentum=0.9)
creiterion = nn.L1Loss()
for epochs in (range(config['epochs'])):
    AB_epoch_loss = []
    B_epoch_loss = []
    E_epoch_loss = []
    for batch in tqdm(range(int(number_of_batches))):
        #Training Alice and Bob together
        alice.train()
        bob.train()
        eve.train()
        key = (torch.randint(0, 2, (config['batch_size'], config['Key_bits']))*2-1).float().to(device)
        data = (torch.randint(0, 2, (config['batch_size'], config['Data_bits']))*2-1).float().to(device)
        cat = torch.cat((data,key),dim=1)
        cipher = alice(torch.cat((data,key),dim=1))
        adv = eve(cipher)
        dec = bob(torch.cat((cipher,key),dim=1).to(device))
        bob_loss = creiterion(dec,data)/2
        eve_loss = creiterion(adv,data)/2
        # ab_loss = bob_loss + ((config['Data_bits']/2 - eve_loss)**2)/((config['Data_bits']/2)**2)
        ab_loss = bob_loss + (1 - eve_loss**2)
        # ABE_optimizer.zero_grad()
        A_optimizer.zero_grad()
        B_optimizer.zero_grad()
        ab_loss.backward()
        nn.utils.clip_grad_value_(alice.parameters(), 1)
        nn.utils.clip_grad_value_(bob.parameters(), 1)
        
        A_optimizer.step()
        B_optimizer.step()
        # ABE_optimizer.step()
        
        
        # Training the adversary
        
        for i in range(2):
            key = (torch.randint(0, 2, (config['batch_size'], config['Key_bits']))*2-1).float().to(device)
            data = (torch.randint(0, 2, (config['batch_size'], config['Data_bits']))*2-1).float().to(device)
            with torch.no_grad():
                cipher = alice(torch.cat((data,key),dim=1).to(device))
            adv = eve(cipher)
            eve_loss = creiterion(adv,data)/2
            E_optimizer.zero_grad()
            eve_loss.backward()
            nn.utils.clip_grad_value_(eve.parameters(), 1)
            E_optimizer.step()
        
        AB_epoch_loss.append(ab_loss.item())
        B_epoch_loss.append(bob_loss.item())
        E_epoch_loss.append(eve_loss.item())
        
        AB_loss.append(ab_loss.item())
        B_loss.append(bob_loss.item())
        E_loss.append(eve_loss.item())
        wandb.log({"AB_loss": ab_loss.item(), "B_loss": bob_loss.item(), "E_loss": eve_loss.item()})
    print("Epoch: {}, AB_loss: {}, B_loss: {}, E_loss: {}".format(epochs,np.mean(AB_epoch_loss),np.mean(B_epoch_loss),np.mean(E_epoch_loss)))
                

print(len(AB_loss))
plt.figure(figsize=(7, 4))
plt.plot(AB_loss, label='A-B')
plt.plot(B_loss, label='Bob')
plt.plot(E_loss, label='E_loss')
plt.xlabel("Iterations", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.legend(fontsize=13)
plt.savefig("LSTM_loss.png")
    
torch.save(alice,f"./model/{name}/alice.pt")
torch.save(bob,f"./model/{name}/bob.pt")
torch.save(eve,f"./model/{name}/eve.pt")

wandb.save(f"./model/{name}/alice.pt")
wandb.save(f"./model/{name}/bob.pt")
wandb.save(f"./model/{name}/eve.pt")
    
    
