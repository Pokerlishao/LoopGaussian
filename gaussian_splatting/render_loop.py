#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm, trange
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel



# def change_backcolor(image):
#     black = image.sum(dim = 0).eq(torch.zeros(image.shape[1:], device=image.device))
#     image[:, black] = 1.
#     return image

    

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    T = 48
    f_pos = torch.load(f"./output/f_pos.pt")
    b_pos = torch.load(f"./output/b_pos.pt")
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # rendering = render(view, gaussians, pipeline, background)["render"]
        # gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        images = []
        for t in trange(T, desc='Render'):
            alpha = t / T
            del_pos = (1-alpha) * f_pos[t] + alpha * b_pos[T-t]
            # gaussians._xyz[mask] = 1e8
            # gaussians._scaling[mask] = -1e12
            view.image_height = 1000
            view.image_width = 1000
            rendering = render(view, gaussians, pipeline, background)["render"] # 3, 800, 800
            # rendering =  change_backcolor(rendering)
            torchvision.utils.save_image(rendering, f'./output/flow/render_{t}.png')
            images.append(rendering.permute(1,2,0))
        images = torch.stack(images, dim=0).detach().cpu()
        images = (images * 255).type(torch.uint8)
        save_path = f'./output/{idx}.mp4'
        torchvision.io.write_video(save_path, images, fps = 24)
        torch.cuda.empty_cache()
        print(f'Save video to {save_path}. Total frames: {images.shape[0]}. Resolution: {images.shape[1]} x {images.shape[2]}')


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        dataset.sh_degree = 3
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1., 1., 1.] 
        # if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)