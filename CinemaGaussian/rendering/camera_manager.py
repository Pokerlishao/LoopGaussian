import os
import random
import json
import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# This piece of code references the
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/__init__.py#L21

class CameraManeger():
    def __init__(self):
        self.cameras = []
        self.frame = 0
        self.maxFrames = 0
        self.renderFrames = 0
        
    def getCamera(self, frame):
        assert self.maxFrames>0
        return self.cameras[frame % self.maxFrames] #Loop
    
    @property
    def getMaxFrame(self):
        return self.maxFrames
    
    @property
    def getRenderFrame(self):
        return self.renderFrames
        
    def load_from_nerf(self, path, mode = 0, extension='.png'):
        transformsfiles = ["transforms_train.json", "transforms_test.json","transforms_val.json"]
        with open(os.path.join(path, transformsfiles[mode])) as json_file:
            contents = json.load(json_file)
            fovx = contents["camera_angle_x"]

            frames = contents["frames"]
            self.maxFrames = len(frames)
            self.renderFrames = self.maxFrames
            for idx, frame in enumerate(frames):
                image_path = os.path.join(path, frame["file_path"] + extension)

                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                image = Image.open(image_path)
                # print(f'image size: {image.size}')

                fovy = self.focal2fov(self.fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
                self.cameras.append(Camera(width=image.size[0], height=image.size[1],
                                           R=R, T=T, FoVx=FovX, FoVy=FovY, uid = idx, file_path=image_path))

    
    def load_from_colmap(self, path):
        pass
    
    def create_rotation_cameras(self, radius, cheight, center, angular_vel = 5., 
                                image_width = 1000, image_height = 1000,
                                FovX = 0.8722, FovY = 0.8722, maxFrame = 72, renderFrame = 72, unity = False):
        # FoV is default set to 50°
        self.maxFrames = maxFrame
        self.renderFrames = renderFrame
        for idx in range(maxFrame):
            angle = idx * angular_vel / 180.0 * math.pi
            positon = np.array([radius * math.cos(angle), cheight, radius * math.sin(angle)]) if unity else\
                np.array([radius * math.cos(angle),  radius * math.sin(angle), cheight])
            view_dir = center - positon
            view_dir = view_dir / np.linalg.norm(view_dir)
            up = np.array([0.,1.,0.]) if unity else np.array([0.,0.,1.]) 

            # 摄像机空间遵守 OpenGL 约定：摄像机前方为负 Z 轴。这与 Unity 的约定不同，在 Unity 中，摄像机前方为 正 Z 轴。
            right = np.cross(view_dir, up)
            right = right / np.linalg.norm(right)
            new_up = np.cross(right, view_dir)
            c2w = np.eye(4)
            # Y dowm and Z forward
            R = np.column_stack((right, -new_up, view_dir))
            c2w[:3,:3] = R
            c2w[:3,3] = positon
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])
            T = w2c[:3, 3]
            self.cameras.append(Camera(width=image_width, height=image_height,
                                           R=R, T=T, FoVx=FovX, FoVy=FovY, uid = idx, file_path=None))

    def create_static_cameras(self, cam_position, center, 
                                    image_width = 1000, image_height = 1000,
                                    FovX = 0.8722, FovY = 0.8722, renderFrame = 72):
        self.maxFrames = 1
        self.renderFrames = renderFrame
        view_dir = center - cam_position
        view_dir = view_dir / np.linalg.norm(view_dir)
        up = np.array([0.,1.,0.]) # Z up 
        right = np.cross(up, view_dir)
        right = right / np.linalg.norm(right)
        new_up = np.cross(view_dir, right)
        c2w = np.eye(4)
        # Y dowm and Z forward
        R = np.column_stack((right, -new_up, view_dir))
        c2w[:3,:3] = R
        c2w[:3,3] = cam_position
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]
        self.cameras.append(Camera(width=image_width, height=image_height,
                                           R=R, T=T, FoVx=FovX, FoVy=FovY, uid = 0, file_path=None))
    
    def fov2focal(self, fov, pixels):
        return pixels / (2 * math.tan(fov / 2))

    def focal2fov(self, focal, pixels):
        return 2*math.atan(pixels/(2*focal))

    
class Camera(nn.Module):
    def __init__(self, width, height, R, T, FoVx, FoVy, uid = 0, scale = 1.0, file_path = None):
        self.uid = uid
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.R = R # row matrix
        self.T = T
        self.image_width = width
        self.image_height = height
        self.znear = 0.01      
        self.zfar = 100.0
        
        self.world_view_matrix = self.getWorld2View(R, T).transpose(0, 1).cuda()
        self.projection_matrix = self.getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.MVP_Matrix = (self.world_view_matrix.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_position = self.world_view_matrix.inverse()[3, :3]
        
        # print(self.camera_position)

        self.file_path = file_path

    def getWorld2View(self, R, T, scale=1.0):
        '''
        Notice: R is a row matrix.
        '''
        W2C = np.zeros((4, 4))
        W2C[:3, :3] = R.transpose()
        W2C[:3, 3] = T
        W2C[3, 3] = 1.0
        return torch.tensor(W2C, dtype=torch.float32)
    
    def getProjectionMatrix(self, znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P
        
