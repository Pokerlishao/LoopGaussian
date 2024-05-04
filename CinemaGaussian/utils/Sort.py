import torch
import math
from torch import nn


def cartesian_to_polar(x, y, z, cx, cy, cz):
    rho = torch.sqrt((x - cx)**2 + (y - cy)**2)
    phi = torch.atan2(y - cy, x - cx)
    theta = torch.atan2(torch.sqrt((x - cx)**2 + (y - cy)**2), z - cz)
    return rho, phi, theta

def polar_to_cartesian(rho, phi, theta, cx, cy, cz):
    x = rho * torch.cos(phi) * torch.sin(theta) + cx
    y = rho * torch.sin(phi) * torch.sin(theta) + cy
    z = rho * torch.cos(theta) + cz
    return x, y, z


def sort_clockwise_spiral(pc):
    xyz = pc[:,:3]
    centroid = torch.mean(xyz, dim=0)
    rhos, phis, thetas = cartesian_to_polar(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                                        centroid[0], centroid[1], centroid[2])
    
    sorted_indices = torch.argsort(phis)
    thetas = thetas[sorted_indices] * 20
    sorted_indices = torch.argsort(thetas.ceil(), stable=True)
    # sorted_points_polar = torch.stack((rhos[sorted_indices], phis[sorted_indices], thetas[sorted_indices]), dim=1)
    
    # sorted_points_cartesian = polar_to_cartesian(sorted_points_polar[:, 0], 
    #                                              sorted_points_polar[:, 1],
    #                                              sorted_points_polar[:, 2],
    #                                              centroid[0], centroid[1], centroid[2])
    
    return sorted_indices
    
