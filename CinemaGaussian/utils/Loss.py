import torch
from torch import nn



def chamfer_disntance(pc1, pc2):
    # Expand dimensions to broadcast
    pc1_b = pc1.unsqueeze(1) # M * 1 * 3
    pc2_b = pc2.unsqueeze(0) # 1 * N * 3
    
    # calculate distance
    distance = torch.sum((pc1_b - pc2_b) ** 2, dim=2)
    min_dist1, _ = torch.min(distance, dim=1)
    min_dist2, _ = torch.min(distance, dim=0)
    chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)
    
    return chamfer_dist
    