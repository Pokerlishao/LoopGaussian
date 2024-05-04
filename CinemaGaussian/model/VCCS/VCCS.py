import torch
import torch.nn as nn
import torch.nn.functional as F


class VCCS(nn.Module):
    '''
    Args:
        pc: point cloud
        r_voxel: the radius of voxel
        r_seed: the radius of seed voxel 
        r_search: the search radius
    https://pcl.readthedocs.io/projects/tutorials/en/master/supervoxel_clustering.html#supervoxel-clustering
    '''

    def __init__(self, r_voxel = 0.01, r_seed = 0.1, r_search=0.05):
        super(VCCS, self).__init__()
        self.r_voxel = r_voxel
        self.r_seed = r_seed
        self.r_search = 0.5 * r_seed
        self.N = 0.05 * r_search * r_search * 3.14159265 / (r_voxel * r_voxel) # 判断噪声的阈值


    def select_seed(self, pc):
        '''
        Args:
            point_cloud: shape == [N, 3]. Just need position.
        Output:
            seed_voxel_indices: shape == [num_seed, 3]. Coordinates after vectorization.
        '''
        # Normalize the point cloud to [0,1]
        min_point = torch.min(pc, dim=0)[0] # no need for indices
        max_point = torch.max(pc, dim=0)[0]
        normalized_pc = (pc - min_point) / (max_point - min_point)

        # Compute voxel indices
        point_indices = torch.floor(normalized_pc / self.r_voxel).int()
        voxel_indices = point_indices.unique(dim=0) # [N, 3]

        super_voxel_indices = torch.floor(normalized_pc / self.r_seed).int().unique(dim=0)  #计算哪些超体素被占据
        center_voxel = super_voxel_indices * self.r_seed + self.r_seed * 0.5   #计算超体素中心点坐标
        center_voxel_indices = torch.floor(center_voxel / self.r_voxel).int()    #超体素中心的体素化坐标
        
        distances = torch.cdist(center_voxel_indices.float(), voxel_indices.float())
        closest_indices = torch.argmin(distances, dim=1)
        seed_indices = voxel_indices[closest_indices]
        
        # filter seed point

        return seed_indices
    
    def filter_seed_points(self):
        
        pass
        
    def forward(self, pc):
        xyz = pc[:,:3]
        indices = self.select_seed(xyz)
        return indices
        