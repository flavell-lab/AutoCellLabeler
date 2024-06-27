import h5py, nrrd, itertools, os, re, sys, csv
import pandas as pd


def map_roi_to_neuron(file_path, confidence_threshold=2, exclude_rois=[0]):
    # Ensure that the file exists
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return "Error: File not found."
    
    # Use column positions instead of names
    neuron_class_col = 0
    roi_id_col = 2
    confidence_col = 3

    # Filter out rows with confidence below the threshold
    df = df[df.iloc[:, confidence_col] >= confidence_threshold]
    
    # Initialize the dictionaries
    neuron_mapping = {}
    confidence_mapping = {}
    
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
            if roi in exclude_rois:
                continue
            
            # Add the neuron ID to the neuron_mapping dictionary
            if roi in neuron_mapping:
                if neuron_id not in neuron_mapping[roi]:
                    neuron_mapping[roi].append(neuron_id)
            else:
                neuron_mapping[roi] = [neuron_id]

            # Add the confidence value to the confidence_mapping dictionary
            confidence_mapping[roi] = confidence_value
    
    return neuron_mapping, confidence_mapping

def extract_neuron_ids_threshold(directory_path, threshold, confidence_threshold=3, exclude_question=True):
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
