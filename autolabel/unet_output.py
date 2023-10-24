import h5py, nrrd, itertools, os, re, sys, csv


def create_probability_dict(img_roi_path, unet_predictions_path, is_gt=False):
    # Step 1: Loading the Data
    with h5py.File(img_roi_path, 'r') as f:
        img_roi_data = f['roi'][:]
    
    if is_gt:
        with h5py.File(unet_predictions_path, 'r') as f:
            unet_predictions = f['label'][:]
    else:
        with h5py.File(unet_predictions_path, 'r') as f:
            unet_predictions = f['predictions'][:]

    # Step 2: Handling Different Shapes
    unet_predictions = np.transpose(unet_predictions, (1, 2, 3, 0))  # Convert CxDxHxW to DxHxWxC

    # Step 3: Constructing the Probability Dictionary
    probability_dict = {}
    
    unique_rois = np.unique(img_roi_data)
    # Exclude the background (ROI=0)
    unique_rois = unique_rois[unique_rois != 0]
    
    for roi in unique_rois:
        roi_mask = img_roi_data == roi
        
        # Average Probability Strategy
        avg_probs = unet_predictions[roi_mask].mean(axis=0)
        
        # Normalization
        avg_probs /= avg_probs.sum()
        
        # Convert to dictionary format
        channel_probs = avg_probs
        
        probability_dict[roi] = channel_probs

    return probability_dict


def output_label_file(probability_dict, h5_path, nrrd_path, output_csv_path):
    # Load the H5 file to get the name mapping
    with h5py.File(h5_path, 'r') as f:
        label_names = ["NULL"] + [name.decode('utf-8') for name in f['neuron_ids'][:]]
    
    # Load the NRRD file
    data, _ = nrrd.read(nrrd_path)
    
    # Track occurrences of neuron classes by confidence
    neuron_confidence_tracker = {}
    rois_in_prob_dict = set(probability_dict.keys())
    rois_in_nrrd = set(np.unique(data)) - {0}
    
    # Process each ROI label in the probability dictionary to build the neuron_confidence_tracker
    for roi_label, probabilities in probability_dict.items():
        # Neuron Class (argmax of probabilities)
        neuron_class_index = np.argmax(probabilities)
        neuron_class = label_names[neuron_class_index]
        
        # Confidence level for current row
        max_prob = np.max(probabilities)
        if neuron_class == "NULL":
            confidence = 1
        elif max_prob < 0.5:
            confidence = 1
        elif 0.5 <= max_prob < 0.8:
            confidence = 2
        elif 0.8 <= max_prob < 0.95:
            confidence = 3
        elif 0.95 <= max_prob < 0.99:
            confidence = 4
        else:
            confidence = 5
        
        # Update the neuron_confidence_tracker
        if neuron_class not in neuron_confidence_tracker:
            neuron_confidence_tracker[neuron_class] = []
        neuron_confidence_tracker[neuron_class].append(confidence)
    
    # Identify neurons to be removed based on the specified criteria
    neurons_to_remove = [
        neuron for neuron, confidences in neuron_confidence_tracker.items()
        if max(confidences) >= 3 and 1 in confidences
    ]
    
    # Remove the confidence-1 entries for the identified neurons
    for neuron in neurons_to_remove:
        neuron_confidence_tracker[neuron] = [conf for conf in neuron_confidence_tracker[neuron] if conf != 1]
    
    # Process each ROI label in the probability dictionary for the final output
    output_data = []
    for roi_label, probabilities in probability_dict.items():
        # Neuron Class (argmax of probabilities)
        neuron_class_index = np.argmax(probabilities)
        neuron_class = label_names[neuron_class_index]
        
        # Confidence level for current row
        max_prob = np.max(probabilities)
        if neuron_class == "NULL":
            confidence = 1
        elif max_prob < 0.5:
            confidence = 1
        elif 0.5 <= max_prob < 0.8:
            confidence = 2
        elif 0.8 <= max_prob < 0.95:
            confidence = 3
        elif 0.95 <= max_prob < 0.99:
            confidence = 4
        else:
            confidence = 5
        
        # If neuron_class is in neurons_to_remove and has confidence 1, skip this entry
        if neuron_class in neurons_to_remove and confidence == 1:
            continue
        
        # Coordinates (center of mass for the ROI label)
        coordinates = np.argwhere(data == roi_label)
        center_of_mass = tuple(map(int, coordinates.mean(axis=0)))
        
        # Alternatives including the most likely candidate
        sorted_indices = np.argsort(probabilities)[::-1]
        alternatives = [
            (label_names[i], p) for i, p in zip(sorted_indices, probabilities[sorted_indices])
            if p >= 0.01
        ]
        alternatives_str = ', '.join([f"{name}({prob:.2f})" for name, prob in alternatives])
        
        # Append to output data
        output_data.append([neuron_class, ",".join(map(str, center_of_mass)), roi_label, confidence, alternatives_str])
    
    # Write to CSV file
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Neuron Class", "Coordinates", "ROI ID", "Confidence", "Alternatives"])
        writer.writerows(output_data)
    
    # Return required lists
    multiple_occurrences = [neuron for neuron, confidences in neuron_confidence_tracker.items() if len(confidences) > 1]
    missing_rois = list(rois_in_nrrd - rois_in_prob_dict)
    
    return multiple_occurrences, missing_rois, neurons_to_remove

# This is the final version of the function incorporating the adjustments.
# You can use it in the same way as the previous versions of the function.
