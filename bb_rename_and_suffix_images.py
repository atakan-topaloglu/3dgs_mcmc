import os
import json
import shutil
import argparse
import re

def rename_and_suffix_images(split_file_path, transforms_file_path, input_dir, dataset_root_dir):
    """
    Renames images based on a transforms file and saves them to a synthetic
    data directory (e.g., 'synthetic_12'), which is determined from the
    split file name.

    This function reads a list of 'test_ids' from a split file, finds the
    corresponding 'file_path' for each ID in a transforms file, and copies
    the images from the input directory to a 'synthetic_XX' directory within
    the dataset root. The number XX is parsed from the split file name.
    The new filenames will match the original filenames from the transforms file,
    without any suffix.

    Args:
        split_file_path (str): Path to the train_test_split[...].json file.
        transforms_file_path (str): Path to the transforms.json file.
        input_dir (str): Path to the directory containing the original images.
        dataset_root_dir (str): Path to the root of the dataset, where the
                                'synthetic_XX' directory will be created.

    Raises:
        FileNotFoundError: If any of the input files or the input directory do not exist.
        ValueError: If the number of images in the input directory does not match the
                    number of 'test_ids', or if the number cannot be parsed from the split file name.
        KeyError: If a 'test_id' from the split file cannot be found in the
                  transforms file.
    """
    # --- 1. Validate and Load Inputs ---
    print("Starting image processing...")

    # Validate input paths
    for path in [split_file_path, transforms_file_path, input_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: The input path '{path}' does not exist.")

    # Validate output directory as per requirements
    match = re.search(r'train_test_split_(\d+)\.json$', os.path.basename(split_file_path))
    if not match:
        raise ValueError(f"Could not extract view count from split file name: '{os.path.basename(split_file_path)}'. "
                         "Expected format: train_test_split_XX.json")
    num_views = match.group(1)
    output_dir = os.path.join(dataset_root_dir, f"synthetic_{num_views}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory set to: '{output_dir}'")

    try:
        with open(split_file_path, 'r') as f:
            split_data = json.load(f)
        with open(transforms_file_path, 'r') as f:
            transforms_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse a JSON file. Details: {e}")
        raise

    # --- 2. Prepare Data and Mappings ---
    
    # Get and sort the test IDs numerically
    test_ids = sorted(split_data.get('test_ids', []))
    if not test_ids:
        raise ValueError(f"Error: No 'test_ids' found in '{split_file_path}'.")
    print(f"Found {len(test_ids)} test IDs in '{os.path.basename(split_file_path)}'.")

    # Get and sort the source image files lexicographically
    try:
        source_images = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])
    except OSError as e:
        print(f"Error reading from input directory '{input_dir}': {e}")
        raise
        
    print(f"Found {len(source_images)} image files in '{input_dir}'.")

    # --- 3. Core Validation ---
    if len(source_images) != len(test_ids):
        raise ValueError(
            f"Error: Mismatch between number of images and test IDs.\n"
            f" - Images found in '{input_dir}': {len(source_images)}\n"
            f" - Test IDs found in '{os.path.basename(split_file_path)}': {len(test_ids)}\n"
            "Please ensure the input directory contains only the images corresponding to the test set."
        )

    # Create an efficient lookup map from colmap_im_id to file_path
    id_to_filepath_map = {frame['colmap_im_id']: frame['file_path'] for frame in transforms_data['frames']}

    # --- 4. Process, Rename, and Save Images ---
    print(f"Output will be saved to '{output_dir}'.")

    # Pair sorted source images with sorted test_ids and process them
    for source_image_name, test_id in zip(source_images, test_ids):
        # Find the target file path from the transforms data
        target_relative_path = id_to_filepath_map.get(test_id)
        
        if not target_relative_path:
            raise KeyError(f"Error: test_id '{test_id}' from split file was not found in '{transforms_file_path}'.")

        # Construct full source path
        source_path = os.path.join(input_dir, source_image_name)
        
        # Get base filename from transforms.json
        target_basename = os.path.basename(target_relative_path)
        
        destination_path = os.path.join(output_dir, target_basename)

        # Copy the file to the new location with the new name
        print(f"  - Mapping '{source_image_name}' (id: {test_id}) -> '{target_basename}'")
        shutil.copy2(source_path, destination_path)

    print("\nSuccessfully processed and saved all images.")


def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Renames synthetic images and places them in the correct 'synthetic_XX' directory "
                    "based on the provided split file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--split_file',
        type=str,
        required=True,
        help="Path to the train_test_split[...].json file."
    )
    parser.add_argument(
        '--transforms_file',
        type=str,
        required=True,
        help="Path to the main transforms.json file."
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Directory containing the original images to be renamed (e.g., the test images)."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Root directory of the dataset. The script will create a 'synthetic_XX' sub-directory here."
    )

    args = parser.parse_args()

    try:
        rename_and_suffix_images(
            args.split_file,
            args.transforms_file,
            args.input_dir,
            args.output_dir
        )
    except (FileNotFoundError, ValueError, KeyError, OSError) as e:
        print(f"\nOperation failed. Reason: {e}")
        # The script will exit with a non-zero status code implicitly
        # due to the unhandled exception, which is good for scripting.

if __name__ == '__main__':
    main()