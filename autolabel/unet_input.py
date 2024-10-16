from functools import reduce
import h5py, nrrd, itertools, os, re, sys, csv
import numpy as np
import pandas as pd

from autolabel.roi_map import map_roi_to_neuron

RAW_DTYPE = np.uint16
LABEL_DTYPE = np.uint8
WEIGHT_DTYPE = np.uint16

def generate_combinations(neuron_id):
    """
    Generates all possible combinations for a neuron ID by replacing each '?' with 'D', 'V', 'L', or 'R'.

    This function takes a neuron ID string that may contain one or more '?' characters, which represent 
    uncertain positions. It generates all possible combinations by replacing each '?' with every 
    combination of the characters 'D', 'V', 'L', and 'R'.

    Args:
        neuron_id (str): The neuron ID containing zero or more '?' characters indicating uncertainty.

    Returns:
        list: A list of strings representing all possible combinations of the neuron ID with '?' replaced.
    """
    possibilities = ['D', 'V', 'L', 'R']
    uncertain_count = neuron_id.count('?')
    combs = list(itertools.product(possibilities, repeat=uncertain_count))
    return [reduce(lambda x, y: x.replace('?', y, 1), comb, neuron_id) for comb in combs]

def expand_nrrd_dimension(input_filepath, output_filepath):
    """
    Expands a 3D NRRD image by adding an extra dimension to create a 4D image.

    This function reads a 3D NRRD image from the specified input file, adds an 
    extra dimension to the image, and saves the resulting 4D image to the specified 
    output file. The added dimension is inserted as the first dimension, effectively 
    converting the original image shape from (X, Y, Z) to (X, Y, Z, 1).

    Parameters:
    ----------
    input_filepath : str
        Path to the input NRRD file containing the 3D image.
    output_filepath : str
        Path to the output NRRD file where the 4D image will be saved.

    Raises:
    ------
    ValueError
        If the input image is not a 3D NRRD image.

    Example:
    --------
    expand_nrrd_dimension('input.nrrd', 'output.nrrd')
    """
    # Read the original 3D NRRD image and its header
    data, header = nrrd.read(input_filepath)
    
    # Check if the image is indeed 3D
    if data.ndim != 3:
        raise ValueError("Input image must be a 3D NRRD image.")
    
    # Add an extra dimension
    data_4d = np.expand_dims(data, axis=3)
    
    # Update the header for the new dimension
    header['sizes'] = [1] + list(data.shape)
    
    # Write the new 4D NRRD image to output file
    nrrd.write(output_filepath, data_4d, header)


# Encodes neurons with collapsed (3D) labels and weights.
def encode_neurons(csv_file, nrrd_file, neuron_ids_list, confidence_weight, weight_reduction, id_weight, bkg_weight, unlabeled_weight, min_confidence, num_labels, non_neuron_ids, fm_roi_map = None, unalignedDict = None):
    """
    Generates one-hot encoded labels and a corresponding weight array for neuron segmentation and labeling tasks.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing human labels comprised of neuron ID and ROI mappings.
    nrrd_file : str
        Path to the NRRD file containing ROI data.
    neuron_ids_list : list of str
        List of neuron IDs to be included in the encoding.
    confidence_weight : list of float
        Weights corresponding to different confidence levels in the ROI mappings.
    weight_reduction : float
        Reduction factor for weights associated with uncertain labels (ie: labels with "?" in them). Currently disabled.
    id_weight : float
        Weight assigned to non-target neuron IDs in the weight array.
    bkg_weight : float
        Weight assigned to the background in the weight array.
    min_confidence : int
        Minimum confidence level to consider in the ROI to neuron mapping.
    num_labels : dict
        Dictionary mapping neuron IDs to the number of labels associated with them.
    non_neuron_ids : list of str
        List of IDs that are not neurons (e.g., "granule", "glia").
    fm_roi_map : dict
        Has keys of timepoint which contain roi. The value is the immobilized ROI. The dict is created from a file (of format t x rois) which can be generated using `AutoCellLabeler/notebook/extract_roi_matches.ipynb`. A None value indicates that these datasets are IM and don't need their ROIs converted
    unalignedDict : dict
        Unnecessary, simply for recording & aggregating ROIs that were unable to be aligned to IM

    Returns
    -------
    encoded_labels : numpy.ndarray
        The encoded label data with shape (height, width, depth).
    weight_array : numpy.ndarray
        The corresponding weight array for the labels with the same shape as `encoded_labels`.
    """
    # Read the NRRD file
    data_raw, _ = nrrd.read(nrrd_file) # ROIs

    if fm_roi_map is not None: # Indicates this is FM
        # Match with immobilized - Create "data: immobiled-converted"
        data_raw[data_raw >= len(fm_roi_map)] = 0 # Some of the data is too large, presumably unmatched ROIs
        data = np.vectorize(fm_roi_map.get)(data_raw)
        # data_im_conv[data_im_conv == -1] = 0 # BRIAN -FOR NOW
        data[data == -1] = 185

    all_rois = np.unique(data)


    foreground_mask = (data > 0)

    # Mapping Neuron IDs to ROI IDs
    roi_to_neuron, confidence_mapping = map_roi_to_neuron(csv_file, confidence_threshold=min_confidence)

    # Generate mask images for each ROI
    roi_masks = {roi: (data == roi) for roi in all_rois}
    roi_masks_nonmatch = {roi: np.logical_and(~roi_masks[roi], foreground_mask) for roi in all_rois} 
    

    neuron_to_roi = {}
    for roi, neuron_ids in roi_to_neuron.items():
        if len(neuron_ids) > 1:
            continue # For now we'll skip multiple labeled ROIs, but might want to come back and look at this - TODO

        for neuron_id in neuron_ids:
            if neuron_id in neuron_to_roi: # One neuron can have multiple ROIs
                neuron_to_roi[neuron_id].append(roi)
            else:
                neuron_to_roi[neuron_id] = [roi]
    
    # For this weighting scheme, we don't throw out unlabeled ROIs
    ## It's unclear if this actually helps
    neuron_id = "UNLABELED"
    unlabeled_rois = [roi for roi in all_rois if (roi not in roi_to_neuron.keys() and roi != 0)]
    for roi in unlabeled_rois:
        if neuron_id in neuron_to_roi: # One neuron can have multiple ROIs
            neuron_to_roi[neuron_id].append(roi)
        else:
            neuron_to_roi[neuron_id] = [roi]

    if unlabeled_weight is None:
        unlabeled_weight = int(confidence_weight[-1]/len(unlabeled_rois)) # TODO: Might want to change this

    roi_weights = {}

    # One-Hot Encoding and Weight Array
    encoded_labels = np.zeros(data_raw.shape, dtype=LABEL_DTYPE) # We might need to increase the dtype if we pick up more neurons
    weight_array = np.full(data_raw.shape, bkg_weight, dtype=WEIGHT_DTYPE)  # Initialize with bkg_weight
    max_labels = np.max([num_labels[x] for x in num_labels if x not in non_neuron_ids])
    
    # Iterate over unique neuron IDs in neuron_to_roi
    for neuron_id in neuron_to_roi.keys():
        if neuron_id in neuron_ids_list:
            process_neuron_id(neuron_id, neuron_to_roi, data, confidence_mapping, confidence_weight, id_weight, max_labels, num_labels, neuron_ids_list, weight_reduction, encoded_labels, weight_array, roi_weights, roi_masks, roi_masks_nonmatch, unlabeled_weight, unalignedDict)
    

    # Loop through ROI weights to set non-ID channels
    for roi, weights in roi_weights.items():
        mask = roi_masks[roi]
        max_weight = max(weights)
        weight_array[mask] = np.maximum(weight_array[mask], id_weight * max_weight)

    return encoded_labels, weight_array


def process_neuron_id(neuron_id, neuron_to_roi, data, confidence_mapping, confidence_weight, id_weight, max_labels, num_labels,
                    neuron_ids_list, weight_reduction, encoded_labels, weight_array, roi_weights, roi_masks, roi_masks_nonmatch, unlabeled_weight, unalignedDict = None):
    """
    Updates the one-hot encoded labels and weight arrays for a specific neuron ID.

    Parameters
    ----------
    neuron_id : str
        Neuron ID to process.
    neuron_to_roi : dict
        Mapping from neuron IDs to their associated ROIs, from human labelers.
    data : numpy.ndarray
        ROI data from the NRRD file.
    confidence_mapping : dict
        Mapping from ROIs to confidence levels.
    confidence_weight : list of float
        Weights corresponding to different confidence levels in the ROI mappings.
    id_weight : float
        Weight assigned to non-target neuron IDs in the weight array.
    max_labels : int
        Maximum number of labels associated with any neuron ID.
    num_labels : dict
        Dictionary mapping neuron IDs to the number of labels associated with them.
    neuron_ids_list : list of str
        List of neuron IDs to be included in the encoding.
    weight_reduction : float
        Reduction factor for weights associated with uncertain labels (ie: labels with "?" in them). Currently disabled.
    one_hot_encoded : numpy.ndarray
        The one-hot encoded label data being updated.
    weight_array : numpy.ndarray
        The weight array being updated.
    roi_weights : dict
        Dictionary to store weights for each ROI.
    roi_masks : dict
        Dictionary of masks for each ROI.
    roi_masks_nonmatch : dict
        Dictionary of masks for regions not matching each ROI.
    unlabeled_weight : int
        The weight we want to assign to the unlabeled ROIs. My sense is that a higher value here makes the network more willing to guess but generally less confident.
    unalignedDict : dict
        Unnecessary, simply for recording & aggregating ROIs that were unable to be aligned to IM

    Returns
    -------
    None
        Updates `one_hot_encoded`, `weight_array`, and `roi_weights` in place.
    """
    for roi in neuron_to_roi[neuron_id]:
        try:
            mask = roi_masks[roi]
        except KeyError:
            if unalignedDict is not None:
                unalignedDict[roi] = (unalignedDict.get(roi, (0, None))[0] + 1, neuron_id)
            else:
                print(f"Imobilized ROI {roi} ({neuron_id}) not aligned from fm")
                # Not going to write a check, but if you get here in IM it's probably an issue...
            return
        mask_nonmatch = roi_masks_nonmatch[roi]

        # Apply confidence weight
        if neuron_id == "UNLABELED":
            weight = unlabeled_weight
            scaled_weight = weight * 1 # Maybe scale later
        else:
            confidence_level = confidence_mapping[roi]
            weight = confidence_weight[min(int(confidence_level) - 1, len(confidence_weight)-1)]
            scaled_weight = np.round(weight * (max_labels / num_labels.get(neuron_id, 1))).astype(WEIGHT_DTYPE)

        # Adjust weight for uncertain labels
        matches = [neuron_id]
        for match_ in matches:
            match_idx = neuron_ids_list.index(match_)
            if match_idx == 0:
                ValueError("Same ID as background")
            encoded_labels[mask] = match_idx
            
            weight_array[mask] = scaled_weight


        # Store the maximum weight for this ROI
        if roi in roi_weights:
            roi_weights[roi].append(weight)
        else:
            roi_weights[roi] = [weight]

def create_h5_from_nrrd(rgb_path, output_path, crop_roi_input_path,
                        crop_roi_output_path, crop_size, num_labels, θh_pos_is_ventral, foreground_weights=[10, 50, 600, 900, 1000], 
                        question_weight_reduction=5, id_weight=0.3, background_weight=1, unlabeled_weight=1, min_confidence=2,
                        label_file=None, neuron_ids_list_file=None, all_red_path=None, non_neuron_ids=["granule", "glia"], unalignedDict = None, is_freely_moving = False, freely_moving_map_file = None):
    """
    Creates an HDF5 file from NRRD image data for neuron segmentation and labeling, including optional labels and weights.

    Parameters
    ----------
    rgb_path : str
        Path to the NRRD file containing the RGB image data.
    output_path : str
        Path to the output HDF5 file to be created.
    crop_roi_input_path : str
        Path to the NRRD file containing the ROI data for cropping.
    crop_roi_output_path : str
        Path to the output HDF5 file for the cropped ROI data.
    crop_size : tuple of int
        The size of the crop (depth, height, width).
    num_labels : dict
        Dictionary mapping neuron IDs to the number of labels associated with them.
    θh_pos_is_ventral : bool
        Flag indicating the orientation of the image; if False, the image is rotated 180 degrees about the y-axis.
    foreground_weights : list of float, optional
        Weights for different confidence levels in foreground labeling.
    question_weight_reduction : float, optional
        Reduction factor for weights associated with uncertain labels (ie: labels with "?" in them). Currently disabled.
    id_weight : float, optional
        Weight assigned to non-target neuron IDs in the weight array.
    background_weight : float, optional
        Weight assigned to the background in the weight array.
    min_confidence : int, optional
        Minimum confidence level to consider in the ROI to neuron mapping.
    label_file : str, optional
        Path to the CSV file containing human labels consisting of neuron ID and ROI mappings.
    neuron_ids_list_file : str, optional
        Path to the HDF5 file containing the list of neuron IDs.
    all_red_path : str, optional
        Path to an additional NRRD file containing TagRFP data to be concatenated with the RGB data to create the full 4D image.
    non_neuron_ids : list of str, optional
        List of IDs that are not neurons (e.g., "granule", "glia").

    Returns
    -------
    list of int
        List of unique non-zero values in the ROI file that were cropped out.
    """
    
    def compute_crop_slices(center_of_mass, shape, crop_size):
        """
        Computes the slices for cropping around a center of mass.

        Parameters
        ----------
        center_of_mass : numpy.ndarray
            The center of mass coordinates (z, y, x).
        shape : tuple of int
            The shape of the image data (z, y, x).
        crop_size : tuple of int
            The desired crop size (z, y, x).

        Returns
        -------
        slices : list of slice
            List of slice objects for each dimension.
        """
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
        """
        Pads the data with background values.

        Parameters
        ----------
        data : numpy.ndarray
            The data to be padded.
        pad_width : list of tuple
            The pad width for each dimension.

        Returns
        -------
        numpy.ndarray
            The padded data.
        """
        # Pad the background channel with ones
        # data_padded_background = np.pad(data[0:1, :, :, :], pad_width, mode='constant', constant_values=1)
        # Pad the other channels with zeros
        data_padded_others = np.pad(data, pad_width, mode='constant', constant_values=0)
        return data_padded_others #np.concatenate([data_padded_background, data_padded_others], axis=0)
    
    if not is_freely_moving:
        # Step 1: Reading the NRRD file
        img_rgb, _ = nrrd.read(rgb_path)
        
        # Round median value, set negative values to 0
        median_val = np.round(np.median(img_rgb))
        img_rgb[img_rgb < 0] = 0

        # If all_red_path is provided, read the NRRD file and append it to img_rgb
        if all_red_path is not None:
            img_red, _ = nrrd.read(all_red_path)
            img_rgb = np.concatenate([img_rgb, img_red[..., np.newaxis]], axis=-1)
        # TODO: Why is this done after getting the median? Was there a reason behind that?

    else:
        # Read and convert to uint16
        img_rgb, _ = nrrd.read(all_red_path)

        # Round median value, set negative values to 0
        median_val = np.round(np.median(img_rgb))
        img_rgb[img_rgb < 0] = 0

    img_rgb = img_rgb.astype(RAW_DTYPE)
    
    # Permute dimensions for the img_red from WxHxDxC to CxDxHxW
    img_rgb = np.transpose(img_rgb, (3, 2, 1, 0)) # Brian: ok but isn't it faster to have color last?

    img_roi, _ = nrrd.read(crop_roi_input_path)

    if is_freely_moving:
        # Parse dataset and timepoint from filename 
        dataset, t = os.path.splitext(os.path.split(output_path)[1])[0].split("_")
    
        # Read in the conversion from FM ROI #s to Immobilized ROI #s
        with h5py.File(freely_moving_map_file, 'r') as f:
            try:
                fm_roi_map = dict(enumerate(f[dataset][:,int(t) - 1], 1)) # Need to start at 1 because         JULIA
                fm_roi_map[0] = 0
            except:
                print(f"Dataset {dataset} not present in fm ROI match file")
                return
            
        # Convert what will become roi_crop/dataset.h5 to the imobilized numbering scheme 
        img_roi[img_roi >= len(fm_roi_map)] = 0 # Some of the data is too large, presumably unmatched ROIs
        img_roi = np.vectorize(fm_roi_map.get)(img_roi)
        # img_roi[img_roi == -1] = 0 # The prediction analysis just ignores negative ROIs - # TODO: Check the conversion to UNKNOWN in the weighting scheme 
    else:
        fm_roi_map = None

    # Permute dimensions for the img_roi data from WxHxD to DxHxW
    img_roi = np.transpose(img_roi, (2, 1, 0))

    if label_file is not None and neuron_ids_list_file is not None:
        with h5py.File(neuron_ids_list_file, 'r') as f:
            ids_list = ["BACKGROUND"] + [name.decode('utf-8') for name in f['neuron_ids'][:]] + ['UNLABELED']  # Adding "background" ensures the label 0 is reserved
            # # BRIAN TODO IMPORTANT: For now, we're going to use the -1 encoding since it didn't seem to significantly improve the previous version. Maybe change

        encoded_labels, weight_data = encode_neurons(label_file, crop_roi_input_path, ids_list, foreground_weights, question_weight_reduction, id_weight, background_weight, unlabeled_weight, min_confidence, num_labels, non_neuron_ids, fm_roi_map, unalignedDict)

        # Permute dimensions from WxHxD to DxHxW
        encoded_labels = np.transpose(encoded_labels, (2, 1, 0))
        weight_data = np.transpose(weight_data, (2, 1, 0))
    
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

    # If encoded_labels data is provided, crop and pad it
    if label_file is not None:
        encoded_labels = encoded_labels[slices[0], slices[1], slices[2]]
        pad_width_one_hot = [(max((crop_size[dim] - encoded_labels.shape[dim]) // 2, 0),
                                        max(crop_size[dim] - encoded_labels.shape[dim] - 
                                            max((crop_size[dim] - encoded_labels.shape[dim]) // 2, 0), 0))
                                       for dim in range(3)]
        # Use the custom padding function for encoded_labels
        encoded_labels = pad_with_background(encoded_labels, pad_width_one_hot)

        weight_data = weight_data[slices[0], slices[1], slices[2]]
        weight_data = np.pad(weight_data, pad_width_one_hot, mode='constant', constant_values=background_weight)
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

    # If θh_pos_is_ventral is False, rotate image by 180 degrees about x-axis
    if not θh_pos_is_ventral:
        channels_rgb = [np.rot90(img_rgb[c,:,:,:], 2, (0,1)) for c in range(img_rgb.shape[0])]
        img_rgb = np.stack(channels_rgb, axis=0)
        img_roi = np.rot90(img_roi, 2, (0,1))
        if label_file is not None:
            encoded_labels = np.rot90(encoded_labels, 2, (0, 1))
            weight_data = np.rot90(weight_data, 2, (0,1))

    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('raw', data=img_rgb)
        if label_file is not None:
            f.create_dataset('label', data=encoded_labels)
            f.create_dataset('weight', data=weight_data)

    with h5py.File(crop_roi_output_path, 'w') as f:
        f.create_dataset('roi', data=img_roi)

    # Return the list of all unique nonzero values in the ROI file that were cropped out
    return list(cropped_out_values)
