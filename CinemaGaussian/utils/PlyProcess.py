import torch
import torch.nn as nn
import numpy as np
from plyfile import PlyData, PlyElement
import os
from os import makedirs


def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    for i in range(45):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l


def save_ply(features, path, max_degree = 3):
    '''
    Args:
        features: torch.tensor
            features.shape = [num_vertices, 59]
            in each row: position(3), opacities(1), diffuse color(3), sh_color(45), scales(3), rots(4)
        path: string
            the save path
    '''
    makedirs(os.path.dirname(path), exist_ok=True)
    feature_sh_len = ((max_degree + 1) ** 2 - 1) * 3

    xyz = features[:,:3].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = features[:,3:6].detach().cpu().numpy()
    f_rest = features[:,6:6+feature_sh_len].detach().cpu().numpy()
    opacities = features[:,6+feature_sh_len:7+feature_sh_len].detach().cpu().numpy()
    scale = features[:,7+feature_sh_len:10+feature_sh_len].detach().cpu().numpy()
    rotation = features[:,10+feature_sh_len:14+feature_sh_len].detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def load_ply(max_sh_degree: int, path):
    '''
        Returns:   features: torch.tensor
            features.shape = [num_vertices, 59]
            in each row: position(3), diffuse color(3), sh_color(45), opacities(1), scales(3), rots(4)
                            [:,:3]      [:,3:6]           [:,6:51]     [:,51:52]   [:,52:55] [:,55:59]
    '''
    plydata = PlyData.read(path)
    num_vertices = plydata['vertex'].data.shape[0]
    # Position
    # xyz.shape = [num_vertices , 3]
    xyz = torch.stack((torch.tensor(plydata.elements[0]["x"]),
                    torch.tensor(plydata.elements[0]["y"]),
                    torch.tensor(plydata.elements[0]["z"])),  axis=1)

    # diffuse Color
    # features_dc.shape = [num_vertices, 3, 1]
    features_dc = torch.stack((torch.tensor(plydata.elements[0]["f_dc_0"]),
                    torch.tensor(plydata.elements[0]["f_dc_1"]),
                    torch.tensor(plydata.elements[0]["f_dc_2"])),  axis=1).unsqueeze(-1)

    # Spherical harmonic
    # features_extra.shape = [num_vertices, 3, 15] (sh_degree+1) ** 2 - 1
    feature_sh_len = ((max_sh_degree + 1) ** 2 - 1) * 3
    features_extra = torch.empty(num_vertices, feature_sh_len)
    for idx in range(feature_sh_len):
        features_extra[:,idx] = torch.tensor(plydata.elements[0]["f_rest_" + str(idx)])
    features_extra = features_extra.reshape(num_vertices, 3, -1)

    # Opacity
    # opacities.shape = [num_vertices, 1]    
    opacities = torch.tensor(plydata.elements[0]["opacity"]).unsqueeze(-1)

    # Scale
    # scales.shape = [num_vertices, 3]
    scales = torch.empty(num_vertices, 3)
    for idx in range(3):
        scales[:,idx] = torch.tensor(plydata.elements[0]["scale_" + str(idx)])
    
    # Rotation
    # rots.shape = [num_vertices, 4]
    rots = torch.empty(num_vertices, 4)
    for idx in range(3):
        rots[:,idx] = torch.tensor(plydata.elements[0]["rot_" + str(idx)])

    # features [num_vertices, 59]
    if feature_sh_len==0:
        res = torch.cat([xyz, features_dc[:,:,0], opacities,scales,rots], dim=1)
    else:
        res = torch.cat([xyz, features_dc[:,:,0], features_extra.reshape(-1,feature_sh_len),opacities,scales,rots], dim=1)
    return res


def load_pcl_ply(path, max_degree = 3) :
    plydata = PlyData.read(path)
    
    
def save_pcl_ply(features, path, max_degree = 3):
    makedirs(os.path.dirname(path), exist_ok=True)
    xyz = features[:,:3].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = features[:,3:6].detach().cpu().numpy()
    f_dc = f_dc *  0.28209479177387814 + 0.5
    f_dc = (f_dc * 255).astype(np.uint8)
    f_rest = features[:,6:51].detach().cpu().numpy()
    opacities = torch.sigmoid(features[:,51:52]).detach().cpu().numpy()
    opacities = (opacities * 255).astype(np.uint8)
    scale = torch.exp(features[:,52:55]).detach().cpu().numpy()
    rotation = torch.nn.functional.normalize(features[:,55:59]).detach().cpu().numpy()

    
    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), 
                  ('f_dc_0', 'u1'), ('f_dc_1', 'u1'), ('f_dc_2', 'u1'), 
                  ('f_rest_0', 'f4'), ('f_rest_1', 'f4'), ('f_rest_2', 'f4'), ('f_rest_3', 'f4'), ('f_rest_4', 'f4'), ('f_rest_5', 'f4'), ('f_rest_6', 'f4'), ('f_rest_7', 'f4'), ('f_rest_8', 'f4'), ('f_rest_9', 'f4'), ('f_rest_10', 'f4'), ('f_rest_11', 'f4'), ('f_rest_12', 'f4'), ('f_rest_13', 'f4'), ('f_rest_14', 'f4'), ('f_rest_15', 'f4'), ('f_rest_16', 'f4'), ('f_rest_17', 'f4'), ('f_rest_18', 'f4'), ('f_rest_19', 'f4'), ('f_rest_20', 'f4'), ('f_rest_21', 'f4'), ('f_rest_22', 'f4'), ('f_rest_23', 'f4'), ('f_rest_24', 'f4'), ('f_rest_25', 'f4'), ('f_rest_26', 'f4'), ('f_rest_27', 'f4'), ('f_rest_28', 'f4'), ('f_rest_29', 'f4'), ('f_rest_30', 'f4'), ('f_rest_31', 'f4'), ('f_rest_32', 'f4'), ('f_rest_33', 'f4'), ('f_rest_34', 'f4'), ('f_rest_35', 'f4'), ('f_rest_36', 'f4'), ('f_rest_37', 'f4'), ('f_rest_38', 'f4'), ('f_rest_39', 'f4'), ('f_rest_40', 'f4'), ('f_rest_41', 'f4'), ('f_rest_42', 'f4'), ('f_rest_43', 'f4'), ('f_rest_44', 'f4'), 
                  ('opacity', 'u1'), 
                  ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'), 
                  ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)