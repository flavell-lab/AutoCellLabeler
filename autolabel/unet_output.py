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