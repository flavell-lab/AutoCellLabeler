import os
import subprocess

def register_images(fixed_dir, moving_dir, output_dir, datasets, parameter_file):
    """
    Register images using elastix.
    
    Parameters:
    - fixed_dir: Directory containing the reference images.
    - moving_dir: Directory containing the images to be registered.
    - output_dir: Directory to save the output of elastix.
    - datasets: List of dataset names.
    - parameter_file: Path to the elastix parameter file.
    """
    
    # Ensure the directories exist
    if not os.path.exists(fixed_dir) or not os.path.exists(moving_dir):
        raise ValueError("One or both of the input directories do not exist.")
    
    for dataset in tqdm(datasets):
        fixed_image = os.path.join(fixed_dir, f"{dataset}.nrrd")
        moving_image = os.path.join(moving_dir, f"{dataset}.nrrd")
        
        # Check if both images exist
        if not os.path.exists(fixed_image) or not os.path.exists(moving_image):
            print(f"Skipping {dataset} as one or both images do not exist.")
            continue
        
        # Create a directory for the output of this dataset
        output_dir_dataset = os.path.join(output_dir, dataset)
        os.makedirs(output_dir_dataset, exist_ok=True)
        
        # Call elastix
        cmd = [
            "elastix",
            "-f", fixed_image,
            "-m", moving_image,
            "-out", output_dir_dataset,
            "-p", parameter_file,
            "-threads", "32"
        ]
        
        with open(os.path.join(output_dir_dataset, "elastix.log"), "w") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)