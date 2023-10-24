import h5py, nrrd, itertools, os, re, sys, csv
import numpy as np
import pandas as pd

def generate_combinations(neuron_id):
    possibilities = ['D', 'V', 'L', 'R']
    uncertain_count = neuron_id.count('?')
    combs = list(itertools.product(possibilities, repeat=uncertain_count))
    return [reduce(lambda x, y: x.replace('?', y, 1), comb, neuron_id) for comb in combs]

def one_hot_encode_neurons(csv_file, nrrd_file, neuron_ids_list, confidence_weight, weight_reduction, id_weight, bkg_weight, min_confidence):
    # Reading the CSV and NRRD files
    df = pd.read_csv(csv_file)
    data, _ = nrrd.read(nrrd_file)
    
    # Mapping Neuron IDs to ROI IDs
    roi_to_neuron, confidence_mapping = map_roi_to_neuron(csv_file, confidence_threshold=min_confidence)

    neuron_to_roi = {}
    for roi, neuron_ids in roi_to_neuron.items():
        for neuron_id in neuron_ids:
            if neuron_id in neuron_to_roi:
                neuron_to_roi[neuron_id].append(roi)
            else:
                neuron_to_roi[neuron_id] = [roi]

    # One-Hot Encoding and Weight Array
    output_shape = (len(neuron_ids_list),) + data.shape
    one_hot_encoded = np.zeros(output_shape, dtype=np.uint8)
    weight_array = np.full(output_shape, bkg_weight, dtype=np.float32)  # Initialize with bkg_weight

    roi_weights = {}

    replaced_ids = {}
    # Iterate over unique neuron IDs in neuron_to_roi
    for neuron_id in neuron_to_roi.keys():
        if '?' in neuron_id:
            valid_combinations = [comb for comb in generate_combinations(neuron_id) if comb in neuron_ids_list]
            for valid_neuron_id in valid_combinations:
                channel_idx = neuron_ids_list.index(valid_neuron_id)
                if valid_neuron_id not in replaced_ids:
                    replaced_ids[valid_neuron_id] = [neuron_id]
                else:
                    replaced_ids[valid_neuron_id].append(neuron_id)
                process_neuron_id(channel_idx, neuron_id, neuron_to_roi, data, confidence_mapping, confidence_weight, neuron_ids_list, weight_reduction, one_hot_encoded, weight_array, roi_weights)
        else:
            if neuron_id in neuron_ids_list:
                channel_idx = neuron_ids_list.index(neuron_id)
                process_neuron_id(channel_idx, neuron_id, neuron_to_roi, data, confidence_mapping, confidence_weight, neuron_ids_list, weight_reduction, one_hot_encoded, weight_array, roi_weights)

    # Loop through ROI weights to set non-ID channels
    for roi, weights in roi_weights.items():
        mask = (data == roi)
        max_weight = max(weights)
        # Step 2: Modify the list comprehension
        non_id_channels = [idx for idx, nid in enumerate(neuron_ids_list) 
                           if (nid not in replaced_ids or all([roi not in neuron_to_roi[id] for id in replaced_ids[nid]])) and 
                           (nid not in neuron_to_roi or roi not in neuron_to_roi[nid])]
        for idx in non_id_channels:
            weight_array[idx][mask] = id_weight * max_weight

    return one_hot_encoded, weight_array

def process_neuron_id(channel_idx, neuron_id, neuron_to_roi, data, confidence_mapping, confidence_weight, 
                    neuron_ids_list, weight_reduction, one_hot_encoded, weight_array, roi_weights):
    for roi in neuron_to_roi[neuron_id]:
        mask = (data == roi)

        # Apply confidence weight
        confidence_level = confidence_mapping[roi]
        weight = confidence_weight[min(int(confidence_level) - 1, len(confidence_weight)-1)]

        # Adjust weight for uncertain labels
        matches = [neuron_id]
        if '?' in neuron_id:
            matches = []
            for comb in generate_combinations(neuron_id):
                if comb in neuron_ids_list:
                    matches.append(comb)
            weight /= (weight_reduction ** (len(matches)-1))

        for match in matches:
            match_idx = neuron_ids_list.index(match)
            one_hot_encoded[match_idx][mask] = 1
            weight_array[match_idx][mask] = weight

        # Store the maximum weight for this ROI
        if roi in roi_weights:
            roi_weights[roi].append(weight)
        else:
            roi_weights[roi] = [weight]

def create_h5_from_nrrd(rgb_path, output_path, crop_roi_input_path, 
                        crop_roi_output_path, crop_size, foreground_weights=[0.01, 0.2, 0.6, 0.9, 1.0], 
                        question_weight_reduction=5, id_weight=0.2, background_weight=5e-4, min_confidence=2,
                        label_file=None, neuron_ids_list_file=None, all_red_path=None):
    
    def compute_crop_slices(center_of_mass, shape, crop_size):
        slices = []
        for dim in range(3):
            start = max(center_of_mass[dim] - crop_size[dim] // 2, 0)
            end = start + crop_size[dim]

            # Adjust cropping if bounds exceed image dimensions
            if end > shape[dim]:
                end = shape[dim]
                start = max(end - crop_size[dim], 0)

            slices.append(slice(start, end))
        return slices

    def pad_with_background(data, pad_width):
        # Pad the background channel with ones
        data_padded_background = np.pad(data[0:1, :, :, :], pad_width, mode='constant', constant_values=1)
        # Pad the other channels with zeros
        data_padded_others = np.pad(data[1:, :, :, :], pad_width, mode='constant', constant_values=0)
        return np.concatenate([data_padded_background, data_padded_others], axis=0)
    
    # Step 1: Reading the NRRD file
    img_rgb, _ = nrrd.read(rgb_path)
    
    # Round median value, set negative values to 0, and convert to uint16
    median_val = np.round(np.median(img_rgb)).astype(np.uint16)
    img_rgb[img_rgb < 0] = 0

    # If all_red_path is provided, read the NRRD file and append it to img_rgb
    if all_red_path is not None:
        img_red, _ = nrrd.read(all_red_path)
        img_rgb = np.concatenate([img_rgb, img_red[..., np.newaxis]], axis=-1)

        img_rgb = img_rgb.astype(np.uint16)

    
    # Permute dimensions for the img_rgb from WxHxDxC to CxDxHxW
    img_rgb = np.transpose(img_rgb, (3, 2, 1, 0))

    img_roi, _ = nrrd.read(crop_roi_input_path)
    # Permute dimensions for the img_roi data from WxHxD to DxHxW
    img_roi = np.transpose(img_roi, (2, 1, 0))

    if label_file is not None and neuron_ids_list_file is not None:
        with h5py.File(neuron_ids_list_file, 'r') as f:
            ids_list = [name.decode('utf-8') for name in f['neuron_ids'][:]]

        one_hot_encoded, weight_data = one_hot_encode_neurons(label_file, crop_roi_input_path, ids_list, foreground_weights, question_weight_reduction, id_weight, background_weight, min_confidence)
        one_hot_encoded = np.transpose(one_hot_encoded, (0, 3, 2, 1))

        weight_data = np.transpose(weight_data, (0, 3, 2, 1))
        background_mask = np.logical_not(one_hot_encoded.sum(axis=0, keepdims=True).astype(bool))
        one_hot_encoded = np.concatenate([background_mask, one_hot_encoded], axis=0)
        weight_data = np.concatenate([np.full(weight_data[0:1,:,:,:].shape, background_weight), weight_data], axis=0)
    
    center_of_mass = np.round(np.array(np.unravel_index(np.argmax(img_roi), img_roi.shape))).astype(int)
    
    # Define crop slices
    slices = compute_crop_slices(center_of_mass, img_rgb.shape[1:], crop_size)
    
    # Extract the parts of the original img_roi that fall outside of our cropping slices
    outside_slices = [
        (slice(None, slices[0].start), slice(None), slice(None)),
        (slice(slices[0].stop, None), slice(None), slice(None)),
        (slice(None), slice(None, slices[1].start), slice(None)),
        (slice(None), slice(slices[1].stop, None), slice(None)),
        (slice(None), slice(None), slice(None, slices[2].start)),
        (slice(None), slice(None), slice(slices[2].stop, None))
    ]
    cropped_out_values = set()
    for s in outside_slices:
        cropped_out_values.update(np.unique(img_roi[s]))
    cropped_out_values.discard(0)  # Remove zero, since we're interested in non-zero values

    # Crop img_rgb and img_roi
    img_rgb = img_rgb[:, slices[0], slices[1], slices[2]]
    img_roi = img_roi[slices[0], slices[1], slices[2]]

    # If one_hot_encoded data is provided, crop and pad it
    if label_file is not None:
        one_hot_encoded = one_hot_encoded[:, slices[0], slices[1], slices[2]]
        pad_width_one_hot = [(0, 0)] + [(max((crop_size[dim] - one_hot_encoded.shape[dim+1]) // 2, 0),
                                        max(crop_size[dim] - one_hot_encoded.shape[dim+1] - 
                                            max((crop_size[dim] - one_hot_encoded.shape[dim+1]) // 2, 0), 0))
                                       for dim in range(3)]
        # Use the custom padding function for one_hot_encoded
        one_hot_encoded = pad_with_background(one_hot_encoded, pad_width_one_hot)

        weight_data = weight_data[:, slices[0], slices[1], slices[2]]
        weight_data = np.pad(weight_data, pad_width_one_hot, mode='constant', constant_values=0)
    else:
        weight_data = None

    pad_width_rgb = [(0, 0)] + [(max((crop_size[dim] - img_rgb.shape[dim+1]) // 2, 0),
                                 max(crop_size[dim] - img_rgb.shape[dim+1] - 
                                     max((crop_size[dim] - img_rgb.shape[dim+1]) // 2, 0), 0))
                                for dim in range(3)]
    img_rgb = np.pad(img_rgb, pad_width_rgb, mode='constant', constant_values=median_val)

    pad_width_roi = [(max((crop_size[dim] - img_roi.shape[dim]) // 2, 0),
                      max(crop_size[dim] - img_roi.shape[dim] - 
                          max((crop_size[dim] - img_roi.shape[dim]) // 2, 0), 0))
                     for dim in range(3)]
    img_roi = np.pad(img_roi, pad_width_roi, mode='constant', constant_values=0)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('raw', data=img_rgb)
        if label_file is not None:
            f.create_dataset('label', data=one_hot_encoded)
            f.create_dataset('weight', data=weight_data)

    with h5py.File(crop_roi_output_path, 'w') as f:
        f.create_dataset('roi', data=img_roi)

    # Return the list of all unique nonzero values in the ROI file that were cropped out
    return list(cropped_out_values)