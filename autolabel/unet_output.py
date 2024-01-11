import h5py, nrrd, itertools, os, re, sys, csv
import numpy as np

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


def reorder_rois_by_max_prob(rois, roi_index):
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



def output_label_file(probability_dict, h5_path, nrrd_path, output_csv_path, max_distance=8, 
        max_prob_decrease=0.3, min_prob=0.01, exclude_rois=[], lrswap_threshold=0.1,
        repeatable_labels=["granule", "glia", "UNKNOWN"]):
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
                    output_data[other_roi]["notes"] += f"ROI likely merged with {roi_id}."
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

        if alt and confidence > 2:
            confidence = 2

        notes = f"ROI likely merged with {split_roi}." if split_roi is not None else ""

        output_data.append({
            "neuron_class": neuron_class,
            "coordinates": ",".join(map(str, center_of_mass)),
            "roi_id": roi_id,
            "confidence": confidence,
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

def swap_last_character(s: str) -> str:
    if not s:
        return s
    
    if s[-1] == 'L':
        return s[:-1] + 'R'
    elif s[-1] == 'R':
        return s[:-1] + 'L'
    else:
        return s
