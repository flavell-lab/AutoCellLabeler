import h5py, nrrd, itertools, os, re, sys, csv
import numpy as np

def modify_edge_weights(mask, new_value=0.01):
    """
    Modifies the edge pixels of a 3D binary mask by reducing their weights.

    Parameters:
    - mask: A 3D numpy array representing the binary mask.
    - new_value: The new value to assign to edge pixels (default is 0.01).

    Returns:
    - A 3D numpy array with modified edge weights.
    """  
    # Ensure new_value is less than 1
    new_value = min(new_value, 1.0)

    # Convert mask to float for manipulation
    float_mask = mask.astype(float)

    # Find edges by applying a gradient operator, summing the absolute values across each dimension
    gradient = np.sum(np.abs(np.gradient(float_mask)), axis=0)

    # Edge pixels will have a gradient greater than 0
    edges = gradient > 0

    # Reduce the weight of edge pixels
    float_mask[edges] = new_value

    return float_mask

def create_probability_dict(img_roi_path, unet_predictions_path, is_gt=False, roi_edge_value=0.01, contamination_threshold=0.75):
    """
    Creates a probability dictionary mapping each ROI (Region of Interest) to its averaged class probabilities,
    and identifies contaminated ROIs based on the AutoCellLabeler model predictions.

    Parameters
    ----------
    img_roi_path : str
        Path to the HDF5 file containing the ROI data under the 'roi' dataset.
    unet_predictions_path : str
        Path to the HDF5 file containing AutoCellLabeler model predictions.
        If `is_gt` is True, predictions are read from 'label'; otherwise, from 'predictions'.
    is_gt : bool, optional
        If True, the function reads ground truth labels from the 'label' dataset in the HDF5 file.
        Default is False.
    roi_edge_value : float, optional
        Value used in edge weight modification for ROIs.
        Default is 0.01.
    contamination_threshold : float, optional
        Confidence threshold above which pixels are considered for contamination detection.
        Default is 0.75.

    Returns
    -------
    probability_dict : dict
        A dictionary where keys are ROI IDs and values are arrays of averaged class probabilities.
    contaminated_rois : dict
        A dictionary of ROIs that are contaminated, mapping ROI IDs to a list of tuples
        (class_index, pixel_count) for each contaminating class.
    """
    contaminated_rois = {}
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

        per_pixel_predictions = np.argmax(unet_predictions[roi_mask], axis=1)
        per_pixel_confidence = np.max(unet_predictions[roi_mask], axis=1)
        confidence_mask = per_pixel_confidence > contamination_threshold
        per_pixel_predictions = per_pixel_predictions[confidence_mask]
        unique_predictions = np.unique(per_pixel_predictions)
        nonzero_unique_predictions = unique_predictions[unique_predictions != 0]
        frequent_predictions = []
        if len(nonzero_unique_predictions) > 1:
            for prediction in nonzero_unique_predictions:
                n_pred = len(per_pixel_predictions[per_pixel_predictions == prediction])
                if n_pred >= 1:
                    frequent_predictions.append((prediction, n_pred))
            if len(frequent_predictions) > 1:
                contaminated_rois[roi] = frequent_predictions

        modified_mask = modify_edge_weights(roi_mask, roi_edge_value)

        # Calculate weighted average probabilities
        # First, ensure the mask is broadcastable over the predictions' shape
        weights = modified_mask[roi_mask].reshape(-1, 1)
        weighted_sum = (unet_predictions[roi_mask] * weights).sum(axis=0)

        # Normalization
        # Use the sum of weights for normalization instead of the count of True values
        total_weight = weights.sum()
        if total_weight > 0:  # Avoid division by zero
            avg_probs = weighted_sum / total_weight
        else:
            # Handle the case where total_weight is 0 to avoid division by zero
            avg_probs = weighted_sum  # This case should be handled based on your application's needs

        # Ensure the probabilities sum to 1 (normalization)
        avg_probs /= avg_probs.sum()

        # Convert to dictionary format (assuming this part remains the same)
        channel_probs = avg_probs        
        
        probability_dict[roi] = channel_probs

    return probability_dict, contaminated_rois

def reorder_rois_by_max_prob(rois, roi_index):
    """
    Reorders the list of ROIs by moving the ROI at a given index to its correct position
    based on its maximum probability.

    Parameters
    ----------
    rois : list of dict
        List of ROI dictionaries, each containing at least a 'max_prob' key.
    roi_index : int
        Index of the ROI in `rois` that needs to be moved.

    Returns
    -------
    list of dict
        The reordered list of ROIs.

    Raises
    ------
    AssertionError
        If the new position of the ROI would decrease its order in the list, which is not allowed.
    """
    # Extract the ROI that needs to be moved
    roi_to_move = rois.pop(roi_index)
    
    # Find the new position for the ROI based on its max_prob
    new_position = next(
        (index for index, roi in enumerate(rois) if roi['max_prob'] < roi_to_move['max_prob']), 
        len(rois)
    )

    assert new_position >= roi_index, "Cannot increase probability of ROI."
    
    # Insert the ROI at the new position
    rois.insert(new_position, roi_to_move)
    
    return rois


# classes with too few detections in training data to believe the labels
EXCLUDED_CLASSES = ['glia', 'granule', 'RIFL', 'RIFR', 'RIFL', 'RIFR', 'AFDL', 'AFDR', 'RMFL', 'RMFR', 'SIADL', 'SIADR', 'VA01', 'VD01', 'AVG', 'DD01', 'SABVL', 'SABVR', 'SABVL', 'SABVR', 'SIBDL', 'SIBDR', 'ADFL', 'RIGL', 'RIGR', 'RIGL', 'RIGR', 'AVFL', 'DB02']

def output_label_file(probability_dict, contaminated_rois, roi_sizes, h5_path, nrrd_path, output_csv_path, max_distance=8, 
        max_prob_decrease=0.3, min_prob=0.01, exclude_rois=[], lrswap_threshold=0.1, roi_matches=[],
        repeatable_labels=["granule", "glia", "UNKNOWN"], contamination_threshold=10, contamination_frac_threshold=0.2,
        confidence_demote=2, excluded_classes=EXCLUDED_CLASSES):
    """
    Generates an output CSV file containing neuron labels, coordinates, and additional information
    based on the probability dictionary and other parameters.

    Parameters
    ----------
    probability_dict : dict
        Dictionary mapping ROI IDs to arrays of averaged class probabilities.
    contaminated_rois : dict
        Dictionary of ROIs that are contaminated, mapping ROI IDs to a list of tuples
        (class_index, pixel_count) for each contaminating class.
    roi_sizes : dict
        Dictionary mapping ROI IDs to their sizes (number of pixels).
    h5_path : str
        Path to the HDF5 file containing neuron IDs under the 'neuron_ids' dataset.
    nrrd_path : str
        Path to the NRRD file containing ROI data.
    output_csv_path : str
        Path where the output CSV file will be saved.
    max_distance : float, optional
        Maximum distance to consider two ROIs as potentially merged.
        Default is 8.
    max_prob_decrease : float, optional
        Maximum allowable decrease in probability when considering alternative labels.
        Default is 0.3.
    min_prob : float, optional
        Minimum probability threshold to assign a neuron class.
        Default is 0.01.
    exclude_rois : list of int, optional
        List of ROI IDs to exclude from processing.
        Default is empty list.
    lrswap_threshold : float, optional
        Confidence threshold of the unlikelier of the 'L' or 'R' label for this neuron class. If above this level, it will be classified as '?' instead of either 'L' or 'R'.
        Default is 0.1.
    roi_matches : list of int, optional
        List indicating each ROI's matches with the freely-moving dataset.
        Default is empty list.
    repeatable_labels : list of str, optional
        Labels that can be assigned to multiple ROIs.
        Default is ["granule", "glia", "UNKNOWN"].
    contamination_threshold : int, optional
        Threshold for the number of contaminating pixels to consider an ROI contaminated.
        Default is 10.
    contamination_frac_threshold : float, optional
        Fraction of ROI size above which contamination is considered significant.
        Default is 0.2.
    confidence_demote : int, optional
        Confidence level to demote contaminated or alternative labels to.
        Default is 2.
    excluded_classes : list of str, optional
        List of neuron classes to exclude due to insufficient training data.
        Default is EXCLUDED_CLASSES.

    Returns
    -------
    list of dict
        List containing information about each processed ROI, including neuron class,
        coordinates, ROI ID, confidence, alternatives, and notes.

    Notes
    -----
    The function writes an output CSV file with the following columns:
    "Neuron Class", "Coordinates", "ROI ID", "Confidence", "Alternatives", "Notes".
    """
    # Load the H5 file to get the name mapping
    with h5py.File(h5_path, 'r') as f:
        label_names = ["UNKNOWN"] + [name.decode('utf-8') for name in f['neuron_ids'][:]]
    
    # Load the NRRD file
    data, _ = nrrd.read(nrrd_path)
    
    # Track occurrences of neuron labels and their centers of mass
    neuron_tracker = {}

    # Process each ROI label in the probability dictionary to extract needed information
    rois = []
    for roi_id, probabilities in probability_dict.items():
        if roi_id in exclude_rois:
            continue
        # Neuron Class (index and probability of most likely neuron class)
        sorted_indices = np.argsort(probabilities)[::-1]
        most_likely_class_index = sorted_indices[0]
        most_likely_prob = probabilities[most_likely_class_index]

        neuron_class = label_names[most_likely_class_index]
        max_prob = most_likely_prob

        # Coordinates (center of mass for the ROI label)
        coordinates = np.argwhere(data == roi_id)
        center_of_mass = tuple(map(lambda x: int(round(x)) + 1, coordinates.mean(axis=0)))

        # Add to list for further processing
        rois.append({
            "roi_id": roi_id,
            "probabilities": probabilities,
            "sorted_indices": sorted_indices,
            "max_prob": max_prob,
            "center_of_mass": center_of_mass,
            "max_used_label_index": 0,
            "alternatives": sorted_indices,
        })

    # Sort ROIs by network's confidence
    rois.sort(key=lambda x: x['max_prob'], reverse=True)

    # Initialize output data
    output_data = []

    count = 0
    # Process ROIs
    while True:
        if count > len(rois) - 1:
            break
        roi = rois[count]

        roi_id = roi["roi_id"]
        probabilities = roi["probabilities"]
        sorted_indices = roi["sorted_indices"]
        center_of_mass = roi["center_of_mass"]
        max_used_label_index = roi["max_used_label_index"]
        alternatives = roi["alternatives"]

        # Get the most likely class and its probability
        neuron_class_index = sorted_indices[max_used_label_index]
        neuron_class = label_names[neuron_class_index]
        max_prob = probabilities[neuron_class_index]

        # Check for alternative labels
        alternatives_str = ', '.join([f"{label_names[i]}({probabilities[i]:.2f})" for i in alternatives if probabilities[i] >= 0.01])

        # Check if the label has been used
        used = False
        alt = False
        split_roi = None

        if neuron_class in neuron_tracker and neuron_class not in repeatable_labels:
            for other_roi, other_roi_id, other_center_of_mass in neuron_tracker[neuron_class]:
                distance = np.linalg.norm(np.array(center_of_mass) - np.array(other_center_of_mass))
                if distance < max_distance:
                    output_data[other_roi]["notes"] += f"ROI likely merged with {roi_id}. "
                    split_roi = other_roi_id
                    break

            if split_roi is None:
                for other_roi, other_roi_id, other_center_of_mass in neuron_tracker[neuron_class]:
                    if probabilities[neuron_class_index] - probabilities[sorted_indices[max_used_label_index + 1]] > max_prob_decrease: # ALT label
                        max_used_label_index = 0
                        neuron_class_index = sorted_indices[max_used_label_index]
                        neuron_class = label_names[neuron_class_index]
                        max_prob = probabilities[neuron_class_index] # don't edit the maximum probability in the ROI because it will break list sorting
                        neuron_class += "-alt"
                        alt = True
                        break

                if not alt:
                    used = True
        
        if used and roi["max_prob"] > min_prob:
            roi["max_used_label_index"] += 1
            neuron_class_index = sorted_indices[roi["max_used_label_index"]]
            neuron_class = label_names[neuron_class_index]
            old_max_prob = max_prob
            roi["max_prob"] = probabilities[neuron_class_index]
            max_prob = roi["max_prob"]

            # print(f"ROI {roi_id} used {neuron_class} instead of {label_names[sorted_indices[roi['max_used_label_index']-1]]} ({old_max_prob:.2f} vs {max_prob:.2f})")
            rois = reorder_rois_by_max_prob(rois, count)
            continue
                

        if not used and not alt:
            neuron_tracker.setdefault(neuron_class, []).append((count, roi_id, center_of_mass))

        original_neuron_class = label_names[neuron_class_index]

        for idx in sorted_indices:
            if idx == neuron_class_index:
                continue
            if probabilities[idx] < lrswap_threshold:
                continue
            if swap_last_character(label_names[idx]) == original_neuron_class:
                neuron_class = original_neuron_class[:-1] + "?" + ("-alt" if alt else "")
                max_prob += probabilities[idx]
                break

        contaminated = False
        notes = f"ROI likely merged with {split_roi}." if split_roi is not None else ""

        if roi_id in contaminated_rois:
            contamination = contaminated_rois[roi_id]
            max_contam = 0
            max_contam_idx = -1
            for (i, (label, contam)) in enumerate(contamination):
                if contam > max_contam:
                    max_contam = contam
                    max_contam_idx = i
            sum_nonmax_contam = 0
            contam_to_txt = ""
            for (i, (label, contam)) in enumerate(contamination):
                if i != max_contam_idx:
                    exclude = False
                    if "?" in neuron_class:
                        legal_possibilities = [neuron_class, neuron_class[:-1] + "L", neuron_class[:-1] + "R"]
                        if label_names[label] in legal_possibilities:
                            exclude = True
                    if not exclude:
                        sum_nonmax_contam += contam
                contam_to_txt += label_names[label] + ": " + str(contam) + ", " 

            if contam_to_txt != "":
                contam_to_txt = contam_to_txt[:-2]
            roi_size = roi_sizes.get(roi_id, 1)
            if roi_size == 1:
                print("WARNING: ROI size 1 for ROI ", roi_id, " in ", nrrd_path, ". Check for errors.")
            if (sum_nonmax_contam >= contamination_threshold) or (sum_nonmax_contam / roi_size >= contamination_frac_threshold):
                notes += f"ROI possibly contaminated - " + contam_to_txt + ". "
                contaminated = True

        if roi_id > len(roi_matches) or roi_matches[roi_id-1] == 0:
            notes += "ROI not matched to freely-moving dataset. "
        
        if max_prob < min_prob:
            neuron_class = "UNKNOWN"
        
        if neuron_class == "UNKNOWN" or max_prob < 0.1:
            confidence = 0
        elif max_prob < 0.5:
            confidence = 1
        elif max_prob < 0.75:
            confidence = 2
        elif max_prob < 0.95:
            confidence = 3
        elif max_prob < 0.99:
            confidence = 4
        else:
            confidence = 5

        if (alt or contaminated or original_neuron_class in excluded_classes) and confidence > confidence_demote:
            confidence = confidence_demote


        output_data.append({
            "neuron_class": neuron_class,
            "coordinates": ",".join(map(str, center_of_mass)),
            "roi_id": roi_id,
            "confidence": confidence,
            "max_prob": max_prob,
            "alt": alt,
            "contaminated": contaminated,
            "alternatives": alternatives_str,
            "notes": notes
        })
        count += 1

    # Sort output data by neuron class
    # output_data.sort(key=lambda x: (x["neuron_class"].rstrip("-alt"), x["neuron_class"].endswith("-alt")))

    # Write to CSV file
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Neuron Class", "Coordinates", "ROI ID", "Confidence", "Alternatives", "Notes"])
        for data in output_data:
            writer.writerow([data["neuron_class"], data["coordinates"], data["roi_id"], data["confidence"], data["alternatives"], data["notes"]])

    return output_data

def get_roi_size(roi_path: str) -> int:
    """
    Calculates the sizes (number of pixels) of each ROI in the given HDF5 file.

    Parameters
    ----------
    roi_path : str
        Path to the HDF5 file containing ROI data under the 'roi' dataset.

    Returns
    -------
    dict
        Dictionary mapping ROI IDs to their sizes.

    Notes
    -----
    The function assumes that ROI labels are positive integers starting from 1 up to the maximum label.
    """
    roi_sizes = {}
    with h5py.File(roi_path, 'r') as f:
        data = f["roi"][:]
        for roi in range(1, np.max(data) + 1):
            roi_sizes[roi] = np.sum(data == roi)
    return roi_sizes

def swap_last_character(s: str) -> str:
    """
    Swaps the last character of a string from 'L' to 'R' or 'R' to 'L'.

    Parameters
    ----------
    s : str
        The input string.

    Returns
    -------
    str
        The string with its last character swapped if it is 'L' or 'R'; otherwise, returns the original string.
    """
    if not s:
        return s
    
    if s[-1] == 'L':
        return s[:-1] + 'R'
    elif s[-1] == 'R':
        return s[:-1] + 'L'
    else:
        return s
