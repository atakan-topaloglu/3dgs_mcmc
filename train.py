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

import os
import json
import torch
from random import randint
import numpy as np
from utils.loss_utils import l1_loss, ssim, cauchy_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import build_scaling_rotation
import math

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from torchvision.transforms.functional import gaussian_blur
from lpipsPyTorch import LPIPS

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        chkpnt_data = torch.load(checkpoint)
        # Handle new checkpoint format with difficulty scores
        if len(chkpnt_data) == 3 and isinstance(chkpnt_data[2], torch.Tensor):
            model_params, first_iter, synthetic_view_difficulties_chkpnt = chkpnt_data
            # Defer loading scores until after they are initialized
        else: # Backwards compatibility for old checkpoints
            model_params, first_iter = chkpnt_data
            synthetic_view_difficulties_chkpnt = None
            print("[Warning] Loaded old checkpoint format. Re-initializing difficulty scores.")
        gaussians.restore(model_params, opt)
    else:
        synthetic_view_difficulties_chkpnt = None
    
    lpips_vgg = LPIPS(net_type='vgg').to("cuda")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    gt_viewpoint_stack = None
    synth_viewpoint_stack = None

    train_gt_cameras = scene.getTrainGTCameras().copy()
    train_synthetic_cameras = scene.getTrainSyntheticCameras().copy()

    use_split_sampling = opt.gt_synth_ratio >= 0 and len(train_gt_cameras) > 0 and len(train_synthetic_cameras) > 0
    ema_loss_for_log = 0.0
    ema_synthetic_loss_for_log = 0.0
    ema_depth_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    print(args.cap_max)
    print(args.opacity_reg)
    print(args.scale_reg)
    print(args.noise_lr)
    print(args.gt_synth_ratio)

    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()
        Ll1 = torch.tensor(0.0, device="cuda")

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if use_split_sampling:
            # Decide whether to use GT or synthetic camera
            if opt.gt_synth_ratio == 0:
                prob_gt = 1
            else: # > 0
                progress = iteration / opt.iterations
                prob_synth = 6*progress*((1-progress)**3)
                prob_gt = np.clip(1-prob_synth, 0, 1)

            if torch.rand(1).item() < prob_gt:
                # Pick from GT
                if not gt_viewpoint_stack:
                    gt_viewpoint_stack = train_gt_cameras.copy()
                viewpoint_cam = gt_viewpoint_stack.pop(randint(0, len(gt_viewpoint_stack) - 1))
            else:
                              # Pick from synthetic using adaptive sampling
                if not synth_viewpoint_stack:
                    synth_viewpoint_stack = train_synthetic_cameras.copy()
                viewpoint_cam = synth_viewpoint_stack.pop(randint(0, len(synth_viewpoint_stack) - 1))
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        # depth_image = render_pkg["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = 0.0

        if viewpoint_cam.is_synthetic:
            target_image = gt_image # Blurring removed


            if viewpoint_cam.attention_map is not None:
                # C = 0.2 
                # error = image - target_image
                # cauchy_per_pixel = torch.log1p((error / C)**2)
                # cauchy_term = (cauchy_per_pixel * viewpoint_cam.attention_map).mean()
                l1_term = l1_loss(image, target_image)
                l1_term = (l1_term * viewpoint_cam.attention_map).mean()
            else:
                # cauchy_term = cauchy_loss(image, target_image)
                l1_term = l1_loss(image, target_image)

            ssim_term = 1.0 - ssim(image, target_image)
            lpips_term = lpips_vgg(image.unsqueeze(0), target_image.unsqueeze(0)).mean()

            synth_loss = (1.0 - opt.lambda_dssim_synth - opt.lambda_lpips) * l1_term + \
                         opt.lambda_dssim_synth * ssim_term + \
                         opt.lambda_lpips * lpips_term
            
            loss += synth_loss

        else: # Not synthetic
            Ll1 = l1_loss(image, gt_image)
            ssim_term = 1.0 - ssim(image, gt_image)

            gt_loss = (1.0 - opt.lambda_dssim_gt) * Ll1 + \
                      opt.lambda_dssim_gt  * ssim_term
            loss += gt_loss
 
         # Depth Loss
        depth_image = render_pkg["depth"]

        # Depth Loss
        depth_l1_weight_schedule = get_expon_lr_func(lr_init=opt.depth_l1_weight_init, lr_final=opt.depth_l1_weight_final, max_steps=opt.iterations)
        depth_loss = 0.0
        if depth_l1_weight_schedule(iteration) > 0 and viewpoint_cam.depth_reliable and not viewpoint_cam.is_synthetic:
            inv_depth = depth_image
            mono_invdepth = viewpoint_cam.invdepthmap
            depth_mask = viewpoint_cam.depth_mask
            depth_loss = torch.abs((inv_depth - mono_invdepth) * depth_mask).mean() * depth_l1_weight_schedule(iteration)
            loss += depth_loss


        loss = loss + args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
        loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if viewpoint_cam.is_synthetic:
                ema_synthetic_loss_for_log = 0.4 * loss.item() + 0.6 * ema_synthetic_loss_for_log
            else:
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_depth_loss_for_log = 0.4 * depth_loss.item() if isinstance(depth_loss, torch.Tensor) else 0.4 * depth_loss + 0.6 * ema_depth_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Synthetic Loss": f"{ema_synthetic_loss_for_log:.{7}f}", "Depth": f"{ema_depth_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                gaussians.relocate_gs(dead_mask=dead_mask)
                gaussians.add_new_gs(cap_max=args.cap_max)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)

                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))
                
                noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 15_000, 22_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 15_000, 22_000,30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
