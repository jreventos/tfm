
import numpy as np
from readers import*
from DataVisualization import*

# Original source : https://github.com/JielongZ/3D-UNet-PyTorch-Implementation/blob/master/image_reconstruct.py

def get_img_patch_idxs(img, overlap_stepsize):
    """
    This function is used to get patch indices of a single image
    The patch indices generated here are used to crop one image into patches

    :param img: the single image
    :param overlap_stepsize: the overlap step size to generate patches
    :return: patch indices
    """
    patch_idxs = []
    depth, height, width = img.shape
    print(depth,height,width)
    patch_depth, patch_height, patch_width = 128, 128, 56 # TODO: define final patches dimensions

    depth_range = list(range(0, depth - patch_depth + 1, overlap_stepsize))
    height_range = list(range(0, height - patch_height + 1, overlap_stepsize))
    width_range = list(range(0, width - patch_width + 1, overlap_stepsize))

    if (depth - patch_depth) % overlap_stepsize != 0:
        depth_range.append(depth - patch_depth)
    if (height - patch_height) % overlap_stepsize != 0:
        height_range.append(height - patch_height)
    if (width - patch_width) % overlap_stepsize != 0:
        width_range.append(width - patch_width)

    for d in depth_range:
        for h in height_range:
            for w in width_range:
                patch_idxs.append((d, h, w))

    return patch_idxs

#array = VtkReader("/Users/jreventos/Desktop/TFM/tfm/patients_data/masks/train/mask_44.vtk")
#patch_idxs = get_img_patch_idxs(array,64)
#print(patch_idxs)
#print(len(patch_idxs))

def volume_patches(data_path, overlap_stepsize):
    """
    Cropping volumetric images into various patches with fixed size.

    :param data_path: volume path
    :param overlap_stepsize: int
    :return: dictionary of the different patches
    """
    patch_dict = dict()
    patch_depth, patch_height, patch_width = 128, 128, 56 # TODO: define final patches dimensions

    array = VtkReader(data_path)
    patch_idxs = get_img_patch_idxs(img=array, overlap_stepsize=overlap_stepsize)

    for j in range(len(patch_idxs)):
        d, h, w = patch_idxs[j]
        cropped_array = array[d: d + patch_depth, h: h + patch_height, w: w + patch_width]
        patch_dict[j+1] = cropped_array

    return patch_dict

#dict = crop_image("/Users/jreventos/Desktop/TFM/tfm/patients_data/masks/train/mask_44.vtk",64)
#print(len(dict))

def recombine_results(data_path,overlap_stepsize):
    """
    This function is used to reconstruct the cropped mask image patches back to the original size of image.
    # TODO: This function is just a draft to check that the reconstruction works!
    """

    patch_depth, patch_height, patch_width = 128, 128, 56 #TODO: define final patches dimensions.

    array = VtkReader(data_path)

    patch_idxs = get_img_patch_idxs(img=array, overlap_stepsize=overlap_stepsize)

    # Create patches:
    cropped_arrays = []
    for j in range(len(patch_idxs)):
        d, h, w = patch_idxs[j]
        # cropped_array = sitk.GetImageFromArray(array[d: d + patch_depth, h: h + patch_height, w: w + patch_width])
        cropped_array = array[d: d + patch_depth, h: h + patch_height, w: w + patch_width]
        cropped_arrays.append(cropped_array)

    # Reconstruct original volume from patches:
    _d, _h, _w = array.shape
    predicted_mask = np.zeros((_d, _h, _w), np.uint8)

    for patch_idx, array_val in zip(patch_idxs,cropped_arrays):
        d, h, w = patch_idx
        pred_arr = array_val
        depth, height, width = pred_arr.shape
        roi = pred_arr[:, :, :]
        predicted_mask[d: d + depth, h: h + height, w: w + width] = roi

    return predicted_mask

# predicted_array = recombine_results("/Users/jreventos/Desktop/TFM/tfm/patients_data/masks/train/mask_44.vtk",64)
# true_array = VtkReader("/Users/jreventos/Desktop/TFM/tfm/patients_data/masks/train/mask_44.vtk")
# print(np.array_equal(predicted_array,true_array))


