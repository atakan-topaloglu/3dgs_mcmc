# 3dgs-mcmc/utils/make_depth_scale.py
import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
import os
from read_write_model import read_model, qvec2rotmat

def get_scales(key, cameras, images, points3d_ordered, args):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = image_meta.point3D_ids

    mask = pts_idx >= 0
    # This check might be too strict if point indices are not contiguous
    # mask *= pts_idx < len(points3d_ordered)

    valid_indices = np.where(mask)[0]
    pts_idx_valid = pts_idx[valid_indices]
    
    # Ensure all point indices are within the bounds of points3d_ordered
    valid_pts_mask = pts_idx_valid < len(points3d_ordered)
    pts_idx_valid = pts_idx_valid[valid_pts_mask]
    valid_xys = image_meta.xys[valid_indices[valid_pts_mask]]


    if len(pts_idx_valid) > 0:
        pts = points3d_ordered[pts_idx_valid]
    else:
        pts = np.array([[0, 0, 0]])

    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2] 
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    depth_path = f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png"
    if not os.path.exists(depth_path):
        depth_path = f"{args.depths_dir}/{image_meta.name}"

    invmonodepthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if invmonodepthmap is None:
        print(f"Warning: could not read depth map {depth_path}")
        return None
    
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0) & 
        (maps[..., 1] >= 0) & 
        (maps[..., 0] < invmonodepthmap.shape[1]) & 
        (maps[..., 1] < invmonodepthmap.shape[0]) & (invcolmapdepth > 0))
    
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        scale = s_colmap / s_mono if s_mono > 1e-8 else 0.0
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', help="path to scene folder")
    parser.add_argument('--depths_dir', help="path to depth maps")
    parser.add_argument('--model_type', default="bin", choices=['bin', 'txt'])
    args = parser.parse_args()

    is_blender = os.path.exists(os.path.join(args.base_dir, "transforms.json"))


    if is_blender:
        print("Blender scene detected. Assuming depth maps are metric. Generating dummy depth_params.json.")
        depth_params = {}
        
        transform_files = []
        for f in ["transforms.json", "transforms_test.json", "transforms_val.json"]:
            p = os.path.join(args.base_dir, f)
            if os.path.exists(p):
                transform_files.append(p)
        
        for transform_file in transform_files:
            with open(transform_file, 'r') as f:
                meta = json.load(f)
            for frame in meta['frames']:
                image_name = os.path.splitext(os.path.basename(frame['file_path']))[0]
                depth_params[image_name] = {"scale": 1.0, "offset": 0.0}
        
        output_path = os.path.join(args.base_dir, "depth_params.json")
        with open(output_path, "w") as f:
            json.dump(depth_params, f, indent=2)
        
        print(f"depth_params.json created at {output_path}.")
    else:
        colmap_sparse_dir = os.path.join(args.base_dir, "sparse", "0")
        if not os.path.exists(colmap_sparse_dir):
            print(f"Error: 'sparse/0' directory not found in '{args.base_dir}'. This does not seem to be a COLMAP scene.")
            exit(1)

        cam_intrinsics, images_metas, points3d = read_model(colmap_sparse_dir, ext=f".{args.model_type}")
        pts_indices = np.array([points3d[key].id for key in points3d])
        pts_xyzs = np.array([points3d[key].xyz for key in points3d])
        
        max_id = pts_indices.max() if len(pts_indices) > 0 else -1
        points3d_ordered = np.zeros([max_id + 1, 3])
        points3d_ordered[pts_indices] = pts_xyzs


        depth_param_list = Parallel(n_jobs=-1, backend="threading")(
                delayed(get_scales)(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas
            )
        depth_params = {
                depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
                for depth_param in depth_param_list if depth_param is not None
            }

        with open(os.path.join(colmap_sparse_dir, "depth_params.json"), "w") as f:
                json.dump(depth_params, f, indent=2)

        print("depth_params.json created.")
