import h5py, nrrd, itertools, os, re, sys, csv
import pandas as pd


def map_roi_to_neuron(file_path, confidence_threshold=2, output_comments=False):
    """
    Maps Regions of Interest (ROIs) to neuron IDs based on a CSV file and a specified confidence threshold.

    This function reads a CSV file containing neuron mapping information, filters rows based on a confidence
    threshold, and returns dictionaries that map ROIs to neuron IDs, along with confidence values. Optionally,
    it can also return comments associated with each ROI.

    Args:
        file_path (str): The path to the input CSV file containing neuron mapping data.
        confidence_threshold (int, optional): The minimum confidence value required to include an ROI. 
                                              Defaults to 2.
        output_comments (bool, optional): Whether to include comments for each ROI in the output. 
                                          Defaults to False.

    Returns:
        tuple: If `output_comments` is True, returns a tuple of three dictionaries:
               - neuron_mapping (dict): Maps ROI IDs to a list of neuron IDs.
               - confidence_mapping (dict): Maps ROI IDs to their confidence values.
               - comments (dict): Maps ROI IDs to their comments.
               
               If `output_comments` is False, returns a tuple of two dictionaries:
               - neuron_mapping (dict): Maps ROI IDs to a list of neuron IDs.
               - confidence_mapping (dict): Maps ROI IDs to their confidence values.

    Raises:
        FileNotFoundError: If the specified file is not found.
    """
    # Ensure that the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' not found.")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Use column positions instead of names
    neuron_class_col = 0
    roi_id_col = 2
    confidence_col = 3
    comments_col = 4

    # Filter out rows with confidence below the threshold
    df = df[df.iloc[:, confidence_col] >= confidence_threshold]
    
    # Initialize the dictionaries
    neuron_mapping = {}
    confidence_mapping = {}
    comments = {}
    
    # Iterate over each row in the dataframe
    for _, row in df.iterrows():
        neuron_id = row.iloc[neuron_class_col]
        rois = str(row.iloc[roi_id_col]).split('/')
        confidence_value = row.iloc[confidence_col]
        
        for roi in rois:
            # Convert roi to integer and handle any conversion errors
            try:
                if '.' in roi and roi.endswith('.0'):
                    roi = int(float(roi))
                else:
                    roi = int(roi)
            except ValueError:
                continue
            
            # Add the neuron ID to the neuron_mapping dictionary
            if roi in neuron_mapping:
                if neuron_id not in neuron_mapping[roi]:
                    neuron_mapping[roi].append(neuron_id)
            else:
                neuron_mapping[roi] = [neuron_id]

            # Add the confidence value to the confidence_mapping dictionary
            confidence_mapping[roi] = confidence_value

            if output_comments:
                comments[roi] = row.iloc[comments_col]
    
    if output_comments:
        return neuron_mapping, confidence_mapping, comments
    return neuron_mapping, confidence_mapping


def extract_neuron_ids_threshold(directory_path, threshold, confidence_threshold=3, exclude_question=True):
    """
    Extracts neuron IDs from CSV files in a directory, counting their occurrences and filtering based on a threshold.

    This function scans a directory for CSV files whose names start with '202', extracts neuron IDs using the 
    `map_roi_to_neuron` function, and counts the occurrences of each neuron ID. It filters neuron IDs based on 
    a specified count threshold and optional conditions (e.g., excluding IDs with '?' or 'alt').

    Args:
        directory_path (str): The path to the directory containing the CSV files.
        threshold (int): The minimum count required for a neuron ID to be included in the output.
        confidence_threshold (int, optional): The minimum confidence value for including an ROI in the mapping.
                                              Defaults to 3.
        exclude_question (bool, optional): Whether to exclude neuron IDs containing a '?' character. 
                                           Defaults to True.

    Returns:
        list: A list of neuron IDs that meet or exceed the specified count threshold.
    """
    # List all files in the directory that have a .csv extension and begin with '202'
    csv_files = [f for f in os.listdir(directory_path) 
                 if f.endswith('.csv') and f.startswith('202')]
    
    # Dictionary to accumulate neuron IDs and their counts
    neuron_ids_count = {}
    
    # Process each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        neuron_mapping = map_roi_to_neuron(file_path, confidence_threshold)
        
        # Count neuron IDs while skipping IDs that contain '?' and 'alt'
        for neuron_ids in neuron_mapping.values():
            for neuron_id in neuron_ids:
                if 'alt' not in neuron_id and (not exclude_question or '?' not in neuron_id):
                    neuron_ids_count[neuron_id] = neuron_ids_count.get(neuron_id, 0) + 1
    
    # Filter neuron IDs based on the threshold and return
    return [neuron_id for neuron_id, count in neuron_ids_count.items() if count >= threshold]
