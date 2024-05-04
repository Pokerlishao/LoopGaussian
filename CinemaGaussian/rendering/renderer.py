import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torchvision
import os, gc
from os import makedirs
from tqdm import trange

from CinemaGaussian.rendering import Camera, camera_manager
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class Renderer(nn.Module):
    def __init__(self):
        super(Renderer, self).__init__


    # This piece of code references the
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py#L18
    def render(self, camera: Camera, gaussians_pc, sh_degree = 3,bg_color = torch.tensor([1.,1.,1.], dtype=torch.float32), scaling_modifier = 1.0):
        '''
        Args:
            camera: Camera
            gaussians_pc: torch.tensor, shape = [num_vertices, 59]
            bg_color = torch.tensor([0,0,0])
            scaling_modifier = 1.0
        '''
        # Set up rasterization configuration
        # torch.cuda.empty_cache()
        # print('##########')
        # print( torch.cuda.is_available())
        # print(bg_color.device)
        # print(gaussians_pc.device)  
        tanfovx = math.tan(camera.FoVx * 0.5)
        tanfovy = math.tan(camera.FoVy * 0.5)
        bg_color = bg_color.to(gaussians_pc.device)
        
 
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=camera.world_view_matrix,
            projmatrix=camera.MVP_Matrix,
            sh_degree=sh_degree,
            campos=camera.camera_position,
            prefiltered=False,
            debug=False
        )
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(gaussians_pc[:,:3], dtype=torch.float32, requires_grad=True, device=gaussians_pc.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points
        dc = gaussians_pc[:,3:6].reshape([-1,3,1]).transpose(1, 2).contiguous()
        rest = gaussians_pc[:,6:51].reshape([-1,3,15]).transpose(1, 2).contiguous()
        
        means3D = gaussians_pc[:,:3]
        shs = torch.cat([dc, rest], dim=1)
        # shs = gaussians_pc[:,3:51].reshape([-1,16,3]) error
        opacity = torch.sigmoid(gaussians_pc[:,51:52])
        scales = torch.exp(gaussians_pc[:,52:55])
        rotations = torch.nn.functional.normalize(gaussians_pc[:,55:59])
        
        # Leave all the below to rasterizer to calculate. Consider adding pre-calculations later
        # Excatly one of either SHs or precomputed colors need to be provided

        colors_precomp = None
        cov3D_precomp = None
                
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        return rendered_image
    
# def render_video(filename, cameras, gaussians, path = './output/', extention = '.mp4', fps = 24):
#     images = []
#     renderer = Renderer()
#     for i in trange(cameras.getMaxFrame, desc="Rendering video"):
#         rendering = renderer.render(cameras.getCamera(i), gaussians)
#         # torchvision.utils.save_image(rendering, f"./output/{i}_.png")
#         images.append(rendering.permute(1,2,0))
    
#     images = torch.stack(images, dim=0).detach().cpu()
#     images = (images * 255).type(torch.uint8)
#     save_path = path + filename + extention
#     torchvision.io.write_video(save_path, images, fps)
#     print(f'Save video to {save_path}. Total frames: {images.shape[0]}. Resolution: {images.shape[1]} x {images.shape[2]}')
    

def render_video(filename, cameras, gaussians, path = './output/', extention = '.mp4', fps = 24, motion_model = None, motion_features = None):
    images = []
    renderer = Renderer()
    # xyz_ = torch.randn_like(gaussians[:,:3]) * 0.001
    with torch.no_grad():
        for i in trange(cameras.getRenderFrame, desc="Rendering video"):
            if motion_model is None:
                rendering = renderer.render(cameras.getCamera(i), gaussians)
            else:
                xyz_ = motion_model(gaussians[:,:3],i)
                gaussians[:,:3] += xyz_
                rendering = renderer.render(cameras.getCamera(i), gaussians)
            # torchvision.utils.save_image(rendering, f"./output/{i}_.png")
            images.append(rendering.permute(1,2,0))
   
    images = torch.stack(images, dim=0).detach().cpu()
    images = (images * 255).type(torch.uint8)
    save_path = path + filename + extention
    torchvision.io.write_video(save_path, images, fps)
    torch.cuda.empty_cache()
    print(f'Save video to {save_path}. Total frames: {images.shape[0]}. Resolution: {images.shape[1]} x {images.shape[2]}')
    
def render_picture(cameras, gaussians, frame = 0, save_picture = False, path = './output/'):
    renderer = Renderer()
    with torch.no_grad():
        rendering = renderer.render(cameras.getCamera(frame), gaussians)
    image = rendering.permute(1,2,0)
    return image