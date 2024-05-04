import os
import struct
import torch
import torchvision
from torch import nn
from tqdm import trange

from CinemaGaussian.rendering.camera_manager import CameraManeger
from CinemaGaussian.rendering.renderer import render_picture

def visualization_clustering(clusters):
    for cluster in clusters:
        color = torch.randn(3).to(cluster.device)
        cluster[:,3:6] = color
    render_data = torch.cat(clusters, dim=0)
    # save_ply(render_data, './output/cluster/point_cloud.ply')
    
    print(f'Render: {render_data.shape} in {render_data.device}')

    cameras = CameraManeger()    
    cameras.create_rotation_cameras(3., 1.3,np.array([0.,0.,0.]), unity=False) #ficus
    images = []
    
    for i in trange(144):
        image = render_picture(cameras, render_data, frame=0) 
        torchvision.utils.save_image(image.permute(2,0,1), f'./output/cluster/cluster_{i}.png')
    images.append(image)

    images = torch.stack(images, dim=0).detach().cpu()
    images = (images * 255).type(torch.uint8)
    save_path = './output/visual/cluster_visual.mp4'
    torchvision.io.write_video(save_path, images, fps = 24)
    torch.cuda.empty_cache()
    print(f'Save video to {save_path}. Total frames: {images.shape[0]}. Resolution: {images.shape[1]} x {images.shape[2]}')


'''
There are a few bugs that need to be worked out, now temporarily using VCCS (https://github.com/yblin/Supervoxel-for-3D-point-clouds)
'''
def supervoxel_cluster(path, data, resolution = 0.2, k_neighbor = 26):
    ''' supervoxel clustering '''
    # data = data.to(device)
    temp_ = './CinemaGaussian/lib/temp.bin'
    os.system(f'./CinemaGaussian/lib/svcluster {path} {temp_} {resolution} {k_neighbor}')
    
    with open(temp_, 'rb') as f:
        idx = f.read()
        index_cluster = struct.unpack('i' * (len(idx) // 4), idx)
    os.system(f'rm {temp_}')
    num_cluster = max(index_cluster) + 1
    print(f'Read clusters: {num_cluster} clusters of {len(index_cluster)} points.')
    
    GaussianCluster = [[] for _ in range(num_cluster)]
    for i in range(len(index_cluster)):
        id = index_cluster[i]
        GaussianCluster[id].append(data[i])
    GaussianCluster = [torch.stack(cluster) for cluster in GaussianCluster]
    return GaussianCluster