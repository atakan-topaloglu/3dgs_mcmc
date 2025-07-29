import os
import json
import shutil
import argparse

def rename_and_suffix_images(split_file_path, transforms_file_path, input_dir, output_dir):
    """
    Renames images, adds a suffix, and saves them to an existing directory.

    This function reads a list of 'test_ids' from a split file, finds the
    corresponding 'file_path' for each ID in a transforms file, and copies
    the images from the input directory to an output directory. The new
    filenames will have a '_synthetic' suffix added before the extension.

    Args:
        split_file_path (str): Path to the train_test_split[...].json file.
        transforms_file_path (str): Path to the transforms.json file.
        input_dir (str): Path to the directory containing the original images.
        output_dir (str): Path to the directory where renamed images will be saved.
                          This directory MUST exist and MUST NOT be empty.

    Raises:
        FileNotFoundError: If any of the input files, the input directory, or the
                           output directory do not exist or are not directories.
        ValueError: If the number of images in the input directory does not match
                    the number of 'test_ids', or if the output directory is empty.
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
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Error: The destination directory '{output_dir}' does not exist or is not a directory.")
    if not os.listdir(output_dir):
        raise ValueError(f"Error: The destination directory '{output_dir}' must not be empty.")

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
        
        # Add the '_synthetic' suffix before the file extension
        name, ext = os.path.splitext(target_basename)
        suffixed_basename = f"{name}_synthetic{ext}"
        
        # Construct the full destination path
        destination_path = os.path.join(output_dir, suffixed_basename)

        # Copy the file to the new location with the new name
        print(f"  - Mapping '{source_image_name}' (id: {test_id}) -> '{suffixed_basename}'")
        shutil.copy2(source_path, destination_path)

    print("\nSuccessfully processed and saved all images.")


def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Renames image files based on JSON mappings, adds a '_synthetic' suffix, and saves them to a new directory.",
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
        help="Directory where the renamed images will be saved.\n"
             "This directory must already exist and must not be empty."
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