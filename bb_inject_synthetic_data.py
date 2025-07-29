import os
import argparse
import pycolmap
from pathlib import Path
import math

def create_synthetic_camera(ref_camera: pycolmap.Camera, 
                            new_id: int, 
                            synth_width: int, 
                            synth_height: int) -> pycolmap.Camera:
    """
    Creates a new COLMAP camera by scaling the intrinsics of a reference camera.
    """
    print(f"\nCreating synthetic camera with ID {new_id} based on reference camera with dimensions {ref_camera.width}x{ref_camera.height}")
    
    synth_cam = pycolmap.Camera(ref_camera.todict())
    synth_cam.camera_id = new_id
    synth_cam.width = synth_width
    synth_cam.height = synth_height

    if ref_camera.width > 0 and ref_camera.height > 0:
        w_ratio = synth_width / ref_camera.width
        h_ratio = synth_height / ref_camera.height
    else:
        w_ratio = 1.0
        h_ratio = 1.0

    if len(ref_camera.focal_length_idxs()) == 1:
        avg_ratio = (w_ratio + h_ratio) / 2.0
        synth_cam.focal_length = ref_camera.focal_length * avg_ratio
        print(f"  - Scaled single focal length by average ratio {avg_ratio:.4f}")
    elif len(ref_camera.focal_length_idxs()) == 2:
        synth_cam.focal_length_x = ref_camera.focal_length_x * w_ratio
        synth_cam.focal_length_y = ref_camera.focal_length_y * h_ratio
        print(f"  - Scaled fx by {w_ratio:.4f}, fy by {h_ratio:.4f}")

    if len(ref_camera.principal_point_idxs()) > 0:
        synth_cam.principal_point_x = ref_camera.principal_point_x * w_ratio
        synth_cam.principal_point_y = ref_camera.principal_point_y * h_ratio
        print(f"  - Scaled principal point with ratios w: {w_ratio:.4f}, h: {h_ratio:.4f}")

    print("  - Synthetic camera created:")
    print(f"    ID: {synth_cam.camera_id}, Model: {synth_cam.model.name}, "
          f"Width: {synth_cam.width}, Height: {synth_cam.height}")
    print(f"    Params: {synth_cam.params_to_string()}")
    
    return synth_cam

def main():
    """
    Main function to parse arguments and run the injection process.
    """
    parser = argparse.ArgumentParser(
        description="Inject synthetic images into a COLMAP reconstruction."
    )
    parser.add_argument(
        "input_dir", type=Path,
        help="Path to the input COLMAP model directory (containing cameras.txt, etc.)"
    )
    parser.add_argument(
        "output_dir", type=Path,
        help="Path to the output directory to save the modified model."
    )
    parser.add_argument(
        "--width", type=int, required=True,
        help="Width of the synthetic images in pixels."
    )
    parser.add_argument(
        "--height", type=int, required=True,
        help="Height of the synthetic images in pixels."
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found at {args.input_dir}")
        return

    print(f"Loading reconstruction from {args.input_dir}...")
    try:
        rec = pycolmap.Reconstruction(args.input_dir)
        print("Reconstruction loaded successfully.")
        print(rec.summary())
    except Exception as e:
        print(f"Error loading reconstruction: {e}")
        return

    if not rec.cameras:
        print("Error: No cameras found in the reconstruction.")
        return
        
    # --- MODIFICATION START ---
    # Permanently divide the first camera's parameters by 4.
    cam_to_modify_id = min(rec.cameras.keys())
    cam_to_modify = rec.cameras[cam_to_modify_id]
    
    print(f"\nPermanently modifying camera ID {cam_to_modify_id} by dividing by 4...")
    print("  - Before modification:")
    print(f"    ID: {cam_to_modify.camera_id}, Model: {cam_to_modify.model.name}, "
          f"Width: {cam_to_modify.width}, Height: {cam_to_modify.height}")
    print(f"    Params: {cam_to_modify.params_to_string()}")

    # Scale dimensions, rounding up to the nearest integer
    cam_to_modify.width = int(math.ceil(cam_to_modify.width / 4.0))
    cam_to_modify.height = int(math.ceil(cam_to_modify.height / 4.0))

    # Scale intrinsics by the same factor of 4
    if len(cam_to_modify.focal_length_idxs()) == 1:
        cam_to_modify.focal_length /= 4.0
    elif len(cam_to_modify.focal_length_idxs()) == 2:
        cam_to_modify.focal_length_x /= 4.0
        cam_to_modify.focal_length_y /= 4.0

    if len(cam_to_modify.principal_point_idxs()) > 0:
        cam_to_modify.principal_point_x /= 4.0
        cam_to_modify.principal_point_y /= 4.0
    
    print("  - After modification:")
    print(f"    ID: {cam_to_modify.camera_id}, Model: {cam_to_modify.model.name}, "
          f"Width: {cam_to_modify.width}, Height: {cam_to_modify.height}")
    print(f"    Params: {cam_to_modify.params_to_string()}")
    # --- MODIFICATION END ---

    # 1. Create the synthetic camera, using the now-modified camera as the reference.
    new_cam_id = (max(rec.cameras.keys()) + 1) if rec.cameras else 1
    synth_cam = create_synthetic_camera(cam_to_modify, new_cam_id, args.width, args.height)
    rec.add_camera(synth_cam)

    # 2. Create a NEW, simple rig for our synthetic camera
    print("\nCreating a new rig for the synthetic images...")
    new_rig_id = (max(rec.rigs.keys()) + 1) if rec.rigs else 1
    synth_rig = pycolmap.Rig()
    synth_rig.rig_id = new_rig_id
    # A sensor_t is a struct with type and id
    synth_sensor = pycolmap.sensor_t()
    synth_sensor.type = pycolmap.SensorType.CAMERA
    synth_sensor.id = synth_cam.camera_id
    synth_rig.add_ref_sensor(synth_sensor)
    rec.add_rig(synth_rig)
    print(f"  - Created new rig with ID {new_rig_id} for camera ID {new_cam_id}")

    # 3. Inject synthetic images
    print("\nInjecting synthetic images...")
    original_images = list(rec.images.values())
    max_image_id = max(rec.images.keys()) if rec.images else 0
    max_frame_id = max(rec.frames.keys()) if rec.frames else 0
    
    for image in original_images:
        max_image_id += 1
        max_frame_id += 1

        synth_image = pycolmap.Image()
        synth_image.image_id = max_image_id
        synth_image.camera_id = synth_cam.camera_id
        p = Path(image.name)
        synth_image.name = f"{p.stem}_synthetic{p.suffix}"
        clean_points2D = [pycolmap.Point2D(p2d.xy) for p2d in image.points2D]
        synth_image.points2D = clean_points2D

        synth_frame = pycolmap.Frame()
        synth_frame.frame_id = max_frame_id
    
        synth_frame.rig_id = synth_rig.rig_id

        pose = image.cam_from_world()
        synth_frame.rig_from_world = pose
        synth_image.frame_id = synth_frame.frame_id
        synth_frame.add_data_id(synth_image.data_id)

        rec.add_frame(synth_frame)
        rec.add_image(synth_image)
        
        for p2d_idx, p2d in enumerate(image.points2D):
            if p2d.has_point3D():
                track_el = pycolmap.TrackElement(synth_image.image_id, p2d_idx)
                rec.add_observation(p2d.point3D_id, track_el)
    
    print(f"Injected {len(original_images)} synthetic images.")

    # 4. Write the final reconstruction
    print(f"\nWriting modified reconstruction to {args.output_dir}...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rec.write(args.output_dir)
    
    print("\nFinal reconstruction summary:")
    print(rec.summary())
    print(f"\nProcess complete. Modified files are in {args.output_dir}")


if __name__ == "__main__":
    main()