import os
import subprocess

def camera_align_images(fixed_img_path, moving_img_path, output_dir, parameter_file, num_threads=32):
    """
    Perform camera alignment registration on two images using elastix.
    
    Parameters:
    - fixed_img_path: File containing the red image.
    - moving_img_path: File containing the green or blue image.
    - output_dir: Directory to save the output of elastix.
    - parameter_file: Path to the elastix parameter file.
    """
    
    # Ensure the directories exist
    if not os.path.exists(fixed_img_path) or not os.path.exists(moving_img_path):
        raise ValueError("One or both of the input files do not exist.")
    
                
    # Call elastix
    cmd = [
        "elastix",
        "-f", fixed_img_path,
        "-m", moving_img_path,
        "-out", output_dir,
        "-p", parameter_file,
        "-threads", str(n_threads)
    ]
    
    with open(os.path.join(output_dir_dataset, "elastix.log"), "w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)