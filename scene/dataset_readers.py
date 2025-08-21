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
import sys
import glob
from PIL import Image
from typing import NamedTuple, Dict, Any
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import re

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    depth_path: str
    depth_params: dict
    attention_map_path: str
    width: int
    height: int
    is_test: bool
    is_synthetic: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depths_folder, depths_params, synthetic_dir, synth_attention_dir, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        
        height_gt = intr.height
        width_gt = intr.width

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x_gt = intr.params[0]
            focal_length_y_gt = intr.params[0]
        elif intr.model=="PINHOLE":
            focal_length_x_gt = intr.params[0]
            focal_length_y_gt = intr.params[1]
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        FovY_gt = focal2fov(focal_length_y_gt, height_gt)
        FovX_gt = focal2fov(focal_length_x_gt, width_gt)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))

        if not os.path.exists(image_path):
            print(f"\n[Warning] Image {extr.name} not found at '{image_path}', skipping.")
            continue

        is_test = extr.name in test_cam_names_list
    
        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_stem = extr.name[:-n_remove]
        depth_params_cam = None
        if depths_params is not None:
            try:
                depth_params_cam = depths_params[depth_stem]
            except KeyError:
                print("\n", key, "not found in depths_params")
                pass
                
        depth_path = ""
        if depths_folder:
            base_depth_path = os.path.join(depths_folder, depth_stem)
            for ext in ['.png', '.exr', '.npy']:
                potential_path = base_depth_path + ext
                if os.path.exists(potential_path):
                    depth_path = potential_path
                    break

        gt_cam_info = CameraInfo(uid=extr.id, R=np.transpose(qvec2rotmat(extr.qvec)), T=np.array(extr.tvec),
                                 FovY=FovY_gt, FovX=FovX_gt, image_path=image_path, image_name=extr.name,
                                 width=width_gt, height=height_gt, is_synthetic=False, is_test=is_test, attention_map_path="",
                                 depth_path=depth_path, depth_params=depth_params_cam)
        cam_infos.append(gt_cam_info)

        if synthetic_dir:
            synth_image_path = os.path.join(synthetic_dir, os.path.basename(extr.name))
            if os.path.exists(synth_image_path):
                synth_img = Image.open(synth_image_path)
                width_synth, height_synth = synth_img.size
                
                # DON'T TOUCH THIS, THIS WORKS.
                FovY_synth = focal2fov(focal_length_y_gt * height_synth / height_gt, height_synth)
                FovX_synth = focal2fov(focal_length_x_gt * width_synth / width_gt, width_synth)

                synth_attention_path = os.path.join(synth_attention_dir, os.path.basename(extr.name)) if synth_attention_dir else ""
                if synth_attention_path and not os.path.exists(synth_attention_path):
                    print(f"\n[Warning] Attention map for {extr.name} not found at '{synth_attention_path}', skipping.")
                    synth_attention_path = ""

                synth_cam_info = CameraInfo(uid=extr.id, R=np.transpose(qvec2rotmat(extr.qvec)), T=np.array(extr.tvec),
                                            FovY=FovY_synth, FovX=FovX_synth, image_path=synth_image_path, image_name=extr.name,
                                            width=width_synth, height=height_synth, is_synthetic=True, is_test=False,
                                            depth_path="", depth_params=None, attention_map_path=synth_attention_path)
                cam_infos.append(synth_cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, llffhold=8, init_type="sfm", num_pts=100000, num_train_views=-1, train_on_test_synth=False, synth_attention_dir="", random_init_ratio=0.2):
    sparse_dir_name = str(num_train_views) if num_train_views != -1 else "0"
    sparse_path = os.path.join(path, "sparse", sparse_dir_name)
    if not os.path.exists(sparse_path):
        raise FileNotFoundError(f"The specified sparse directory does not exist: {sparse_path}")
    print(f"Reading COLMAP sparse model from: {sparse_path}")
 
    try:
        cameras_extrinsic_file = os.path.join(sparse_path, "images.bin")
        cameras_intrinsic_file = os.path.join(sparse_path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        print(f"Read {len(cam_extrinsics)} extrinsics and {len(cam_intrinsics)} intrinsics from {cameras_extrinsic_file} and {cameras_intrinsic_file}")
    except:
        cameras_extrinsic_file = os.path.join(sparse_path, "images.txt")
        cameras_intrinsic_file = os.path.join(sparse_path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        print(f"Read {len(cam_extrinsics)} extrinsics and {len(cam_intrinsics)} intrinsics from {cameras_extrinsic_file} and {cameras_intrinsic_file}")
    depth_params_file = os.path.join(sparse_path, "depth_params.json")



    if num_train_views != -1:
        potential_dir = os.path.join(path, f"synthetic_{num_train_views}")
        if os.path.isdir(potential_dir):
            synthetic_dir = potential_dir
            print(f"Selected synthetic directory based on num_train_views: {synthetic_dir}")
        else:
            raise ValueError(f"Synthetic directory {potential_dir} not found")

    # Fallback to finding any synthetic directory if specific one isn't found or num_train_views is not set
    # if not synthetic_dir:
    #     all_dirs = glob.glob(os.path.join(path, "synthetic_*"))
    #     synth_dirs = [d for d in all_dirs if re.fullmatch(r'synthetic_\d{1,2}', os.path.basename(d))]
    #     if synth_dirs:
    #         synthetic_dir = sorted(synth_dirs)[0]
    #         print(f"Found fallback synthetic directory: {synthetic_dir}")

    # If synth_attention_dir is provided as an argument, derive the path automatically.
    actual_synth_attention_dir = ""
    if synth_attention_dir  and synthetic_dir:
        actual_synth_attention_dir = synthetic_dir + "_attention_maps"
        print(f"Automatically determined attention map directory: {actual_synth_attention_dir}")

    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = "images" if images == None else images
    all_cam_names = sorted([cam_extrinsics[cam_id].name for cam_id in cam_extrinsics])
    real_cam_names = sorted(list(set(all_cam_names)))
    test_cam_names_list = []
    
    if eval:
        test_txt_path = os.path.join(sparse_path, "test.txt")
        if os.path.exists(test_txt_path):
            print(f"Found test.txt at {test_txt_path}, using for test set.")
            with open(test_txt_path, 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
        else:
            print("test.txt not found, falling back to llffhold split.")
            test_cam_names_list = [name for idx, name in enumerate(real_cam_names) if idx % llffhold == 0]

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir),
                                           depths_folder=os.path.join(path, depths) if depths != "" else "",
                                           depths_params=depths_params, 
                                           synthetic_dir=synthetic_dir,
                                            synth_attention_dir=actual_synth_attention_dir,
                                            test_cam_names_list=test_cam_names_list)
    
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    real_cam_infos = [c for c in cam_infos if not c.is_synthetic]
    synthetic_cam_infos = [c for c in cam_infos if c.is_synthetic]

    train_cam_infos_full_real = [c for c in real_cam_infos if not c.is_test]
    test_cam_infos = [c for c in real_cam_infos if c.is_test]


    if not train_on_test_synth:
        test_image_names = {c.image_name for c in test_cam_infos}
        original_synth_count = len(synthetic_cam_infos)
        synthetic_cam_infos = [c for c in synthetic_cam_infos if c.image_name not in test_image_names]
        filtered_count = original_synth_count - len(synthetic_cam_infos)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} synthetic views that correspond to test viewpoints. Use --train_on_test_synth to include them.")

    if num_train_views != -1 and len(train_cam_infos_full_real) > num_train_views:
        print(f"Downsampling real training cameras from {len(train_cam_infos_full_real)} to {num_train_views}")
        indices = np.linspace(0, len(train_cam_infos_full_real) - 1, num_train_views, dtype=int)
        train_cam_infos_real = [train_cam_infos_full_real[i] for i in indices]
    else:
        train_cam_infos_real = train_cam_infos_full_real

    train_cam_infos = train_cam_infos_real + synthetic_cam_infos

    print("Test cameras: " + ", ".join(sorted([c.image_name for c in test_cam_infos])))
    print("Train cameras (GT): " + ", ".join(sorted([c.image_name for c in train_cam_infos_real])))
    if synthetic_cam_infos:
        print("Train cameras (synthetic): " + ", ".join(sorted([c.image_name for c in synthetic_cam_infos])))

    nerf_normalization = getNerfppNorm(train_cam_infos_real)

    # Point cloud generation
    ply_path = "" # Start with no path
    if init_type == "sfm":
        ply_path = os.path.join(sparse_path, "points3D.ply")
        # bin_path = os.path.join(sparse_path, "points3D.bin")
        txt_path = os.path.join(sparse_path, "points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
                print(f"txt path is {txt_path}")
            storePly(ply_path, xyz, rgb)
    
    elif init_type == "random":
        ply_path = os.path.join(path, "random.ply")
        if not os.path.exists(ply_path):
            print(f"Generating random point cloud ({num_pts})...")
            xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"] * 3 * 2 - (nerf_normalization["radius"] * 3)
            shs = np.random.random((xyz.shape[0], 3)) / 255.0
            storePly(ply_path, xyz, SH2RGB(shs) * 255)
        else:
            print(f"Loading random point cloud from {ply_path}")
    elif init_type == "hybrid":
        ply_path = os.path.join(path, f"hybrid_init_{random_init_ratio:.2f}.ply")
        if not os.path.exists(ply_path):
            print(f"Generating hybrid point cloud and saving to {ply_path}")
            # 1. Load SfM points
            sfm_ply_path = os.path.join(sparse_path, "points3D.ply")
            bin_path = os.path.join(sparse_path, "points3D.bin")
            txt_path = os.path.join(sparse_path, "points3D.txt")
            if not os.path.exists(sfm_ply_path):
                print("Converting point3d.bin to .ply for hybrid initialization.")
                try:
                    xyz, rgb, _ = read_points3D_binary(bin_path)
                except:
                    xyz, rgb, _ = read_points3D_text(txt_path)
                storePly(sfm_ply_path, xyz, rgb)
            
            sfm_pcd = fetchPly(sfm_ply_path)
            sfm_points = sfm_pcd.points
            sfm_colors = sfm_pcd.colors

            # 2. Generate random points
            num_sfm_points = sfm_points.shape[0]
            num_random_points = int(num_sfm_points * random_init_ratio)
            print(f"Hybrid initialization: {num_sfm_points} SfM points, augmenting with {num_random_points} random points.")

            scene_center = -nerf_normalization["translate"]
            scene_radius = nerf_normalization["radius"]
            random_points = (np.random.random((num_random_points, 3)) - 0.5) * (2 * scene_radius * 1.5) + scene_center
            random_colors = np.random.random((num_random_points, 3))

            combined_points = np.concatenate((sfm_points, random_points), axis=0)
            combined_colors = np.concatenate((sfm_colors, random_colors), axis=0)
            storePly(ply_path, combined_points, combined_colors * 255)
        else:
            print(f"Loading existing hybrid point cloud from {ply_path}")
    else:
        print("Please specify a correct init_type: random, sfm, or hybrid")
        exit(0)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def _read_blender_cameras_from_transforms(path, transforms_filename, white_background, is_test_set, depths="", depths_params=None, extension=".png"):
    """
    Helper function to read cameras from a single Blender-style transforms file.
    """
    cam_infos = []

    
    transforms_path = os.path.join(path, transforms_filename)
    if not os.path.exists(transforms_path):
        print(f"[Warning] Transforms file not found at {transforms_path}")
        return []

    with open(transforms_path) as json_file:
         contents = json.load(json_file)
         fovx = contents["camera_angle_x"]
 
         frames = contents["frames"]
         for idx, frame in enumerate(frames):
            # Handle file paths that may or may not have the extension
            relative_path = frame["file_path"]
            if not relative_path.endswith(extension):
                relative_path += extension
            image_path = os.path.join(path, relative_path)

            if not os.path.exists(image_path):
                print(f"[Warning] Image {relative_path} not found at '{image_path}', skipping.")
                continue

            image_name = Path(image_path).stem
            image = Image.open(image_path)
            width, height = image.size
 
             # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
             # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
 

            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy 
            FovX = fovx
 
            depth_path_cam = ""
            depth_params_cam = None
            if depths:
                # Construct depth path, assuming it's in a parallel folder `depths`
                # with the same filename stem but possibly different extension.
                depth_filename_stem = os.path.splitext(os.path.basename(frame["file_path"]))[0]
                base_depth_path = os.path.join(path, depths, depth_filename_stem)
                for ext in ['.png', '.exr', '.npy']:
                    potential_path = base_depth_path + ext
                    if os.path.exists(potential_path):
                        depth_path_cam = potential_path
                        break
            
            if depths_params is not None:
                try:
                    depth_params_cam = depths_params[image_name]
                except KeyError:
                    pass


            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image_path=image_path,
                                        image_name=image_name, width=width, height=height,
                                        is_synthetic=False, is_test=is_test_set,
                                        depth_path=depth_path_cam, depth_params=depth_params_cam, attention_map_path=""))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, num_train_views=-1, train_on_test_synth=False, synth_attention_dir="", init_type="random", num_pts=100_000, extension=".png"):
    print("Reading Blender-style data with sparse/synthetic setup...")

    depths_params = None
    if depths:
        depth_params_file = os.path.join(path, "depth_params.json")
        if os.path.exists(depth_params_file):
            print(f"Found depth_params.json at {depth_params_file}")
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params if "scale" in depths_params[key]])
            if (all_scales > 0).sum() > 0:
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                if "med_scale" not in depths_params[key]:
                    depths_params[key]["med_scale"] = med_scale
        else:
            print(f"WARNING: --depths specified but depth_params.json not found in scene root '{path}'.")
            print("You can generate it by running 'python utils/make_depth_scale.py --base_dir <path_to_scene> --depths_dir <path_to_depths>'.")
            print("Continuing without depth supervision.")
            depths = "" # Disable depth if params not found

    # Find synthetic directory
    if num_train_views != -1:
        potential_dir = os.path.join(path, f"synthetic_{num_train_views}")
        if os.path.isdir(potential_dir):
            synthetic_dir = potential_dir
            print(f"Selected synthetic directory based on num_train_views: {synthetic_dir}")

    # Fallback to finding any synthetic directory if specific one isn't found or num_train_views is not set
    if not synthetic_dir:
        synth_dirs = [d for d in glob.glob(os.path.join(path, "synthetic_*")) if re.fullmatch(r'synthetic_\d+', os.path.basename(d))]
        if synth_dirs:
            synthetic_dir = sorted(synth_dirs)[0]
            print(f"Found fallback synthetic directory: {synthetic_dir}")

    # If synth_attention_dir is provided as an argument, derive the path automatically.
    actual_synth_attention_dir  = ""
    if synth_attention_dir  and synthetic_dir:
        actual_synth_attention_dir = synthetic_dir + "_attention_maps"
        print(f"Automatically determined attention map directory: {actual_synth_attention_dir}")

    # Load ground truth training cameras
    train_transforms_file = "transforms.json" if os.path.exists(os.path.join(path, "transforms.json")) else "transforms_train.json"
    print(f"Reading GT training cameras from: {train_transforms_file}")
    gt_train_cameras = _read_blender_cameras_from_transforms(path, train_transforms_file, white_background, is_test_set=False, depths=depths, depths_params=depths_params, extension=extension)

    # Load test cameras if in eval mode
    test_cam_infos = []
    if eval:
        print("Reading test cameras from: transforms_test.json")
        test_cam_infos = _read_blender_cameras_from_transforms(path, "transforms_test.json", white_background, is_test_set=True, depths=depths, depths_params=depths_params, extension=extension)

    # Load synthetic training cameras, linked to the GT training cameras
    synthetic_cam_infos = []
    if synthetic_dir:
        for gt_cam in gt_train_cameras:
            synth_image_path = os.path.join(synthetic_dir, os.path.basename(gt_cam.image_path))
            if os.path.exists(synth_image_path):
                synth_img = Image.open(synth_image_path)
                synth_width, synth_height = synth_img.size

                # Recalculate FoV for synthetic image if its aspect ratio is different
                FovY_synth = focal2fov(fov2focal(gt_cam.FovX, gt_cam.width) * (synth_height / gt_cam.height), synth_height)
                FovX_synth = focal2fov(fov2focal(gt_cam.FovX, gt_cam.width) * (synth_width / gt_cam.width), synth_width)

                synth_attention_path = os.path.join(actual_synth_attention_dir, os.path.basename(gt_cam.image_path)) if actual_synth_attention_dir else ""
                if synth_attention_path and not os.path.exists(synth_attention_path):
                    synth_attention_path = ""

                synthetic_cam_infos.append(CameraInfo(uid=gt_cam.uid, R=gt_cam.R, T=gt_cam.T,
                                                      FovY=FovY_synth, FovX=FovX_synth, image_path=synth_image_path,
                                                     image_name=gt_cam.image_name, width=synth_width, height=synth_height,
                                                      is_synthetic=True, is_test=False,
                                                      depth_path="", depth_params=None, attention_map_path=synth_attention_path))

    # Downsample GT training views if requested
    if num_train_views != -1 and len(gt_train_cameras) > num_train_views:
        print(f"Downsampling real training cameras from {len(gt_train_cameras)} to {num_train_views}")
        indices = np.linspace(0, len(gt_train_cameras) - 1, num_train_views, dtype=int)
        train_cam_infos_real = [gt_train_cameras[i] for i in indices]
    else:
        train_cam_infos_real = gt_train_cameras

    # Combine GT and synthetic views for the final training set
    train_cam_infos = train_cam_infos_real + synthetic_cam_infos

    print("--- Dataset Summary ---")
    print(f"Total GT Training Cameras: {len(train_cam_infos_real)}")
    print(f"Total Synthetic Training Cameras: {len(synthetic_cam_infos)}")
    print(f"Total Test Cameras: {len(test_cam_infos)}")
    print("-----------------------")

    # Use GT training cameras for scene normalization
    nerf_normalization = getNerfppNorm(train_cam_infos_real)

    # Point cloud generation
    ply_path = os.path.join(path, "points3d.ply")
    if (init_type == "random" or not os.path.exists(ply_path)) and not os.path.exists(ply_path):
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}