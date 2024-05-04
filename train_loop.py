import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from tqdm import tqdm, trange
from argparse import ArgumentParser

from CinemaGaussian.utils.PlyProcess import load_ply, save_ply
from CinemaGaussian.model import EulerMotion_Kriging, train_SG, train_Motion
from CinemaGaussian.utils import supervoxel_cluster, visualization_clustering
from CinemaGaussian.rendering import CameraManeger, render_video, render_picture

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pre_euler_integral(xyz, model, T , smooth = torch.tensor([0.1,0.1,0.1])):
    f_xyz = torch.empty(T, *xyz.shape).to(xyz.device)
    b_xyz = torch.empty(T, *xyz.shape).to(xyz.device)
    f_xyz[0] = xyz
    b_xyz[0] = xyz
    
    smooth = smooth.to(device)
    with torch.no_grad():
        for i in range(1, T):
            f_xyz[i] = f_xyz[i-1] + model(f_xyz[i-1]) * smooth
            b_xyz[i] = b_xyz[i-1] - model(b_xyz[i-1]) * smooth
    return f_xyz, b_xyz

def get_no_sh_point(pc):
    return torch.cat([pc[:,:6], pc[:,-8:]], dim=1)


def main(args):
    
    '''
        Load data
    '''
    data = load_ply(args.max_degree, args.ply_path).to(device)
    # if unity: data[:,:3] = data[:,[0,2,1]]
    print(f'Origin Gaussians number: {data.shape[0]}')
    fea_data = get_no_sh_point(data)
    
    '''
        Mask
    '''
    if args.use_mask:
        mask = torch.load(args.mask_path)
        data = data[mask]
        static_data = data[~mask]
    

    '''
        Train GaussianAutoencoder
    '''
    sg_model = train_SG(fea_data, args.task_name, args.skip_SG, num_epochs=args.SG_epoch)    
    # data = torch.cat([point_data[:,:6], data[:,6:51],point_data[:,6:]], dim = 1)
    

    '''
        Supervoxel cluster  & global features
    '''
    ax_len = torch.max(data[:,:3], dim=0)[0] - torch.min(data[:,:3], dim=0)[0]
    clu_resolution = torch.max(ax_len) / 25
    print(f'Scene resolution for cluster: {clu_resolution}')
    GaussianCluster = supervoxel_cluster(args.ply_path, data, resolution=clu_resolution)
    # visualization_clustering(GaussianCluster)
    center_point = torch.stack([torch.mean(clu[:,:3], dim=0) for clu in GaussianCluster])

    fea_clus = sg_model.get_globalfea(GaussianCluster)
  
    '''
        calculate cosine similarity
    '''
    feature_vectors_normalized = F.normalize(fea_clus, p=2, dim=1)
    similarity_matrix = torch.matmul(feature_vectors_normalized, feature_vectors_normalized.T)
    similarity_matrix.fill_diagonal_(-1)
    nearest_indices = torch.argmax(similarity_matrix, dim=1)
    

    '''
        Eulerian motion field
    '''
    use_addition = True
    dst_vel = center_point - center_point[nearest_indices]
    dst_pos = center_point
    if use_addition:
        Kri_model = EulerMotion_Kriging(center_point, dst_vel)
        # Kri_model = EulerMotion_RBF(center_point, dst_vel)
        add_pos = data[:,:3]
        add_vel = Kri_model(add_pos)
        dst_pos = torch.cat([center_point,add_pos],dim=0)
        dst_vel = torch.cat([dst_vel,add_vel],dim=0)
    dst_vel = torch.sigmoid(dst_vel)
    
    motion_model = train_Motion(dst_pos, dst_vel, args.task_name, args.skip_motion, num_epochs=args.motion_epoch)

    # # # Render
    cameras = CameraManeger()    
    cameras.create_rotation_cameras(3., 1.3,np.array([0.,0.,0.]))
    
    ori_pos = data[:,:3].clone().detach()
    images = []
    T = args.frames
    
    smooth = 1.2 / T * torch.exp(-ax_len) *  torch.tensor([0.5,0.5,2.3]).to(device)
    # smooth = torch.tensor([0.09,0.09,0.09]).to(device) * clu_resolution * 0.87
    print(f'smooth: {smooth}')
    f_pos, b_pos = pre_euler_integral(ori_pos, motion_model, T+1, smooth=smooth)#eflag flag 0.15  ficus 0.09
    torch.save(f_pos, f"./output/{args.task_name}/f_pos.pt")
    torch.save(b_pos, f"./output/{args.task_name}/b_pos.pt")

    with torch.no_grad():
        for t in trange(T, desc='Render'): 
            alpha = t / T
            del_pos = (1-alpha) * f_pos[t] + alpha * b_pos[T-t]
            data[:,:3] = del_pos
            if args.use_mask:
                data = torch.cat([data, static_data], dim=0)
            images.append(render_picture(cameras, data, frame=0))

    images = torch.stack(images, dim=0).detach().cpu()
    images = (images * 255).type(torch.uint8)
    save_path = f'./output/{args.task_name}/visual.mp4'
    torchvision.io.write_video(save_path, images, fps = 24)
    torch.cuda.empty_cache()
    print(f'Save video to {save_path}. Total frames: {images.shape[0]}. Resolution: {images.shape[1]} x {images.shape[2]}')

if __name__ == '__main__':
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--task_name", default='Test', type=str)
    parser.add_argument("--GS_iterations", default=50_000, type=int)
    parser.add_argument("--max_degree", default=3, type=int)
    parser.add_argument("--ply_path", default='', type=str)
    parser.add_argument("--frames", default=48, type=int, help='Total frames')    
    parser.add_argument("--use_mask", action='store_true')
    parser.add_argument("--mask_path", default='', type=str)
    parser.add_argument("--skip_SG", action="store_true", help='Skip train Gaussian autoencoder')
    parser.add_argument("--SG_epoch", default=10_000, type=int, help='Number of iterations for Gaussian autoencoder training')
    parser.add_argument("--skip_motion", action="store_true", help='Skip train Eulerian motion field')
    parser.add_argument("--motion_epoch", default=50_000, type=int, help='Number of iterations for Eulerian motion field training')

    args = parser.parse_args()
    
    if args.ply_path == '':
        args.ply_path = f'./output/{args.task_name}/point_cloud/iteration_{args.GS_iterations}/point_cloud.ply'
    if args.use_mask:
        assert args.mask_path != ''
    
    
    main(args)