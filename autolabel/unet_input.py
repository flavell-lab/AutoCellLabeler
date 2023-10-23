def one_hot_encode_neurons(csv_file, nrrd_file, neuron_ids_list):
    # Reading the CSV and NRRD files
    df = pd.read_csv(csv_file)
    data, _ = nrrd.read(nrrd_file)
    
    # Mapping Neuron IDs to ROI IDs
    roi_to_neuron = map_roi_to_neuron(csv_file)

    neuron_to_roi = {}
    for roi, neuron_ids in roi_to_neuron.items():
        for neuron_id in neuron_ids:
            if neuron_id in neuron_to_roi:
                neuron_to_roi[neuron_id].append(roi)
            else:
                neuron_to_roi[neuron_id] = [roi]

    # One-Hot Encoding
    output_shape = (len(neuron_ids_list),) + data.shape
    one_hot_encoded = np.zeros(output_shape, dtype=np.uint8)
    
    for channel_idx, neuron_id in enumerate(neuron_ids_list):
        if neuron_id in neuron_to_roi:
            for roi in neuron_to_roi[neuron_id]:
                one_hot_encoded[channel_idx][data == roi] = 1
    
    return one_hot_encoded



def create_h5_from_nrrd(nrrd_path, one_hot_encoded, output_path, crop_roi_input_path, 
                        crop_roi_output_path, crop_size, foreground_weight=1, 
                        id_weight=0.3, background_weight=0.001):
    
    # Step 1: Reading the NRRD file
    raw_data, _ = nrrd.read(nrrd_path)
    img_roi, _ = nrrd.read(crop_roi_input_path)

    # Round median value, set negative values to 0, and convert to uint16
    median_val = np.round(np.median(raw_data)).astype(np.uint16)
    raw_data[raw_data < 0] = 0
    raw_data = raw_data.astype(np.uint16)
    
    # Permute dimensions for the raw data from WxHxDxC to CxDxHxW
    raw_data = np.transpose(raw_data, (3, 2, 1, 0))
    
    # Permute dimensions for the one-hot encoded data from CxWxHxD to CxDxHxW
    one_hot_encoded = np.transpose(one_hot_encoded, (0, 3, 2, 1))

    # Permute dimensions for the img_roi data from WxHxD to DxHxW
    img_roi = np.transpose(img_roi, (2, 1, 0))
    
    # Compute the background mask and add it as a new channel at the beginning of the one-hot encoded data
    background_mask = np.logical_not(one_hot_encoded.sum(axis=0, keepdims=True).astype(bool))
    one_hot_encoded = np.concatenate([background_mask, one_hot_encoded], axis=0)
    
    # Compute center of mass
    combined_data = one_hot_encoded[1:].sum(axis=0)  # Excluding the background channel
    center_of_mass = np.round(np.array(np.unravel_index(np.argmax(combined_data), combined_data.shape))).astype(int)

    # Define crop slices
    slices = []
    for dim in range(3):
        start = max(center_of_mass[dim] - crop_size[dim] // 2, 0)
        end = start + crop_size[dim]
        
        # Adjust cropping if bounds exceed image dimensions
        if start < 0:
            start = 0
            end = crop_size[dim]
        if end > raw_data.shape[dim+1]:
            end = raw_data.shape[dim+1]
            start = max(end - crop_size[dim], 0)
        
        slices.append(slice(start, end))

    
    # Crop raw_data, one_hot_encoded data, and img_roi
    raw_data = raw_data[:, slices[0], slices[1], slices[2]]
    one_hot_encoded = one_hot_encoded[:, slices[0], slices[1], slices[2]]
    img_roi = img_roi[slices[0], slices[1], slices[2]]
    
    # Pad raw_data, one_hot_encoded data, and img_roi
    pad_width_raw = [(0, 0)] + [(max((crop_size[dim] - raw_data.shape[dim+1]) // 2, 0),
                                 max(crop_size[dim] - raw_data.shape[dim+1] - 
                                     max((crop_size[dim] - raw_data.shape[dim+1]) // 2, 0), 0))
                                for dim in range(3)]
    raw_data = np.pad(raw_data, pad_width_raw, mode='constant', constant_values=median_val)
    
    pad_width_one_hot = [(0, 0)] + [(max((crop_size[dim] - one_hot_encoded.shape[dim+1]) // 2, 0),
                                    max(crop_size[dim] - one_hot_encoded.shape[dim+1] - 
                                        max((crop_size[dim] - one_hot_encoded.shape[dim+1]) // 2, 0), 0))
                                   for dim in range(3)]
    one_hot_encoded = np.pad(one_hot_encoded, pad_width_one_hot, mode='constant', constant_values=0)

    # Pad img_roi
    pad_width_roi = [(max((crop_size[dim] - img_roi.shape[dim]) // 2, 0),
                      max(crop_size[dim] - img_roi.shape[dim] - 
                          max((crop_size[dim] - img_roi.shape[dim]) // 2, 0), 0))
                     for dim in range(3)]
    img_roi = np.pad(img_roi, pad_width_roi, mode='constant', constant_values=0)
    
    # Construct the weight field
    weight_data = np.full_like(one_hot_encoded, background_weight, dtype=np.float32)
    foreground_mask = (one_hot_encoded == 1)
    id_mask = (one_hot_encoded[1:].sum(axis=0) > 0) & ~foreground_mask[1:]  # Excluding the background channel

    weight_data[foreground_mask] = foreground_weight
    weight_data[1:][id_mask] = id_weight  # Excluding the background channel
    weight_data[0,:,:,:] = background_weight
    
    # Write to the HDF5 File
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('raw', data=raw_data)
        f.create_dataset('label', data=one_hot_encoded)
        f.create_dataset('weight', data=weight_data)

    # Write to the HDF5 File for ROI Cropping
    with h5py.File(crop_roi_output_path, 'w') as f:
        f.create_dataset('roi', data=img_roi)

    return "HDF5 file created successfully."