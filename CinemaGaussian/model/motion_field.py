import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import inv
import numpy as np
# from skgstat import Variogram

from scipy.interpolate import Rbf

from CinemaGaussian.utils.PositionEncoding import PositionalEncoder


class EulerMotion_MKriging(nn.Module):
    def __init__(self, coordinates, values):
        super(EulerMotion_MKriging, self).__init__()
        self.coordinates = coordinates
        self.values = values
        self.sill = torch.var(values)
        self.V = self.spherical_model(torch.cdist(coordinates, coordinates)) #semivariance
        self.n = values.shape[0]
        self.device = coordinates.device
        

    def forward(self, pos):
        l_column = torch.ones((self.n,1)).to(self.device)
        l_row = torch.ones((1, self.n+1)).to(self.device)
        l_row[0,self.n] = 0
        r_row = torch.ones((1, pos.shape[0])).to(self.device)
        semi_variance = self.spherical_model(torch.cdist(self.coordinates, pos))
        l_matrix = torch.vstack([torch.hstack([self.V,l_column]), l_row])
        # r_vector = torch.tensor([semi_variance,torch.tensor(1).to(self.device)])
        r_vector = torch.vstack([semi_variance, r_row])
        weights = torch.linalg.solve(l_matrix, r_vector)

        interpolated_values = weights[:-1].T @ self.values + weights[-1]

        return interpolated_values
        
    
    def spherical_model(self, h , range_ = 1., nugget=0):
        """
        Spherical variogram model.
        """
        range_ = torch.tensor(range_, dtype=h.dtype, device=h.device)
        condition = h > range_
        result = torch.where(condition, self.sill, nugget + self.sill * (1.5 * (h / range_) - 0.5 * (h**3.0) / (range_**3) ))

        return result

class EulerMotion_Kriging(nn.Module):
    def __init__(self, pos, vel, model = 'spherical'):
        super(EulerMotion_Kriging, self).__init__()
        self.modelx = EulerMotion_MKriging(pos, vel[:,0])
        self.modely = EulerMotion_MKriging(pos, vel[:,1])
        self.modelz = EulerMotion_MKriging(pos, vel[:,2])
    
    def forward(self, pos, is_curl = False):

        if is_curl:
            delta = 1e-6
            nx1 = pos.clone().detach()
            nx2 = pos.clone().detach()
            ny1 = pos.clone().detach()
            ny2 = pos.clone().detach()
            nz1 = pos.clone().detach()
            nz2 = pos.clone().detach()
            nx1[:,0] += delta
            nx2[:,0] -= delta
            ny1[:,1] += delta
            ny2[:,1] -= delta
            nz1[:,2] += delta
            nz2[:,2] -= delta
            x1 = self.modelx(nx1)
            x2 = self.modelx(nx2)
            y1 = self.modelx(ny1)
            y2 = self.modelx(ny2)
            z1 = self.modelx(nz1)
            z2 = self.modelx(nz2)
            dF_dx = (x1 - x2) / (2 * delta)
            dF_dy = (y1 - y2) / (2 * delta)
            dF_dz = (z1 - z2) / (2 * delta)
            vel = torch.stack([dF_dy - dF_dz, dF_dz - dF_dx, dF_dx - dF_dy], dim=1)
            vel = F.normalize(vel,p=2., dim=1)
            # print(vel)
        else:
            x = self.modelx(pos)
            y = self.modely(pos)
            z = self.modelz(pos)
            vel = torch.stack([x,y,z],dim=1)
        return vel



class EulerMotion_RBF(nn.Module):
    def __init__(self, pos, vel, sigma = 0.1):
        super(EulerMotion_RBF, self).__init__()
        self.pos = pos.cpu().numpy()
        self.vel = vel.cpu().numpy()
        self.sigma = sigma
        
    
    def forward(self, pos):
        # distances = torch.cdist(self.pos, pos)
        # X = distances.lu_solve(self.vel) #[N, 3]
        # radial_basis_values = self.gaussian_rbf(distances, self.sigma)
        # A = torch.cat([radial_basis_values, torch.ones_like(radial_basis_values[:, 0:1])], dim=1)
        # # Solve the linear system to obtain weights

        # weights = torch.linalg.lstsq(A, self.vel).solution[:A.size(1)]

        # interpolation_result = torch.matmul(self.gaussian_rbf(torch.cdist(self.pos, pos), self.sigma), weights[:-1])
        # print(interpolation_result.shape)
        # return interpolation_result
        
        device = pos.device
        pos = pos.cpu().numpy()
        rbf = Rbf(self.pos[:,0], self.pos[:,1], self.pos[:,2], 
                  self.vel,mode='N-D',function='gaussian')

        
        res = rbf(pos[:, 0], pos[:, 1], pos[:, 2])
        return torch.tensor(res, device=device, dtype=torch.float32)
        
        
    def gaussian_rbf(self, r, sigma):
        """Gaussian Radial Basis Function"""
        return torch.exp(-0.5 * (r / sigma)**2)

'''
    To calculate the motion of whole Gaussians
'''
class EulerMotion_MLP(nn.Module):
    '''
    Args:
        input:
            [num_vertices, 3]
        output:
            [num_vertices, 3]
        Misc:
            add position encoding?
    '''
    def __init__(self):
        super(EulerMotion_MLP, self).__init__()
        self.pe = PositionalEncoder(d_input=3, n_freqs=4)
        self.input_dim = 27  # add time dimension (2 * n_freqs +1) * 3
        self.hidden_dims = [128,64]
        self.output_dim = 3
        self.hidden_layers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dims[0])])
        for i in range(1, len(self.hidden_dims)):
            self.hidden_layers.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)
        self.activate_function = nn.ReLU()


    def forward(self, point):
        # t = torch.tensor(t,dtype=torch.float32,device=point.device).expand(point.shape[0], 1)
        # todo: add a position encoding for t?
        # point = torch.cat([point, t], dim=1)
        point = self.pe(point)
        for layer in self.hidden_layers:
            point = self.activate_function(layer(point))
        xyz_ = self.output_layer(point)
        return xyz_