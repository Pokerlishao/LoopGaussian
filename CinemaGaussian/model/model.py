import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from .poinetnet_autoencoder import GaussianAutoEncoder
from .motion_field import EulerMotion_Kriging, EulerMotion_MLP

def train_SG(data, task_name, skip_train=False, num_epochs = 10_000):
    channel = data.shape[1]
    if not skip_train:
        writer = SummaryWriter(f'./output/{task_name}/PAE_model')
        print(f'Tensorboard log save to: ./output/{task_name}/PAE_model')
        
        model = GaussianAutoEncoder(input_channels=channel, useSTN=False)
        model.to(data.device)
        model.train()
        
        update_tqdm = 50
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        with tqdm(total=num_epochs,desc='Training PAE') as tqdm_:
            for epoch in range(num_epochs):           
                point_data, _ = model(data)
                loss = nn.MSELoss()(point_data, data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # tqdm processing bar  
                if(epoch % update_tqdm == 0):
                    tqdm_.set_postfix(loss='{:.8f}'.format(loss.item()))
                    tqdm_.update(update_tqdm)
                    writer.add_scalar(f'PAE_{task_name}', loss.item(), epoch)
        torch.save(model, f'./output/{task_name}/pae.pth')
    else:
        model = torch.load(f'./output/{task_name}/pae.pth')
        model.to(data.device)
    return model

def train_Motion(pos, vel, task_name, skip_train=False, num_epochs = 50_000):        
    if not skip_train:
        writer = SummaryWriter(f'./output/{task_name}/motion_model')
        print(f'Tensorboard log save to: ./output/{task_name}/motion_model')
        
        model = EulerMotion_MLP()
        model.to(pos.device)
        model.train()
        
        
        update_tqdm = 50
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        with tqdm(total=num_epochs,desc='Training Motion MLP') as tqdm_:
            for epoch in range(num_epochs):           
                pvel = model(pos)
                loss = nn.MSELoss()(vel, pvel)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # tqdm processing bar  
                if(epoch % update_tqdm == 0):
                    tqdm_.set_postfix(loss='{:.8f}'.format(loss.item()))
                    tqdm_.update(update_tqdm)
                    writer.add_scalar(f'Motion_{task_name}', loss.item(), epoch)
        torch.save(model, f'./output/{task_name}/motion_model.pth')
    else:
        model = torch.load(f'./output/{task_name}/motion_model.pth')
        model.to(pos.device)
    return model


class MLP(nn.Module):
    def __init__(self, input_dim = 3, hidden_dims = [64,128,256,256], output_dim = 14):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.hidden_layers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dims[0])])
        for i in range(1, len(self.hidden_dims)):
            self.hidden_layers.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)
        self.activation_function = nn.ReLU()
  

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation_function(layer(x))
        x = self.output_layer(x)
        return x



