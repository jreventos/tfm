
import glob
import re
import numpy as np
from skimage.exposure import rescale_intensity
from tfm.utils.DataVisualization import*
from tfm.utils.readers import*
import matplotlib.pyplot as plt





def standarize(vol,outliers = None):
    """
    Function that rescales data to have a mean (𝜇) of 0 and standard deviation (𝜎) of 1 (unit variance).
    If outliers, computes the 10-th and 99-th percentiles of the volume in a flattened version of the array and
    rescales the volume from the 10-th to 99-th range. It is a way to increase the contrast of the MRI image.

    :param vol: 3D numpy array
    :param outliers: bool
    :return: standarized 3D numpy array
    """

    if outliers:
        p10 = np.percentile(vol, 10)
        p99 = np.percentile(vol, 99)
        vol_rescaled = rescale_intensity(vol, in_range=(p10, p99))
        vol = vol_rescaled

    m = np.mean(vol, axis=(0, 1, 2))
    s = np.std(vol, axis=(0, 1, 2))
    vol_standarized = (vol - m) / s
    return vol_standarized

#array = VtkReader("/Users/jreventos/Desktop/TFM/tfm/patients_data/MRI_volumes/train/MRI_2.vtk")
#stand_array = standarize(array,outliers=True)
#DisplaySlices(stand_array, int(stand_array.max()))

def normalize(vol):
    " Funcion that takes the volume and normalizes its values between [0,1]"

    if (vol.max() - vol.min()) != 0:
        norm_vol = (vol - vol.min()) / (vol.max() - vol.min())
    else:
        norm_vol = (vol - vol.min())
    return norm_vol

# array = VtkReader("/Users/jreventos/Desktop/TFM/tfm/patients_data/MRI_volumes/val/MRI_1.vtk")
# norm_array = normalize(array)
# DisplaySlices(norm_array, int(norm_array.max()))


def center_crop(vol, new_width=None, new_height=None):
    """
    Function that crops the volume from the center with a new width and height. The number of slices does not change.

    :param img: numpy array
    :param new_width: int
    :param new_height: int
    :return: numpy array in the new dimensions
    """

    height = vol.shape[0]
    width = vol.shape[1]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(vol.shape) == 2:
        center_cropped_vol = vol[top:bottom, left:right]
    else:
        center_cropped_vol = vol[top:bottom, left:right, ...]

    return center_cropped_vol


def find_mask_boundaries(path):
    """
    Uses the mask of a single MRI volume to find the boundaries (top and bottom) of the mask in the XYZ axis.

    :param path: mask path
    :return: int lists --> [x_bottom_coord,x_top_coord], [y_bottom_coord, y_top_coord], [z_bottom_coord,z_top_coord]
    """

    bad_mask = ["/Users/jreventos/Desktop/TFM/tfm/patients_data/masks/val/mask_1.vtk",
                "/Users/jreventos/Desktop/TFM/tfm/patients_data/masks/train/mask_3.vtk",
                "/Users/jreventos/Desktop/TFM/tfm/patients_data/masks/train/mask_51.vtk",
                "/Users/jreventos/Desktop/TFM/tfm/patients_data/masks/train/mask_74.vtk",
                "/Users/jreventos/Desktop/TFM/tfm/patients_data/masks/train/mask_75.vtk",
                "/Users/jreventos/Desktop/TFM/tfm/patients_data/masks/train/mask_80.vtk",
                "/Users/jreventos/Desktop/TFM/tfm/patients_data/masks/train/mask_38.vtk"] # the 38 has a problem in the mask
    # Read MRI volume
    if path in bad_mask:
        return [], [], []

    else:
        # print(path)
        mask_array = VtkReader(path)

        # Find coordinates in the 3 axis

        x_idx = [i for i in range(mask_array.shape[0]) if 1 in mask_array[i, :, :]]
        y_idx = [i for i in range(mask_array.shape[1]) if 1 in mask_array[:, i, :]]
        z_idx = [i for i in range(mask_array.shape[-1]) if 1 in mask_array[:,:,i]]
        #a = 0
        #DisplaySlices(mask_array[(min(x_idx) - a):(max(x_idx) + a), (min(y_idx) - a):(max(y_idx) + a), (min(z_idx) - a):(max(z_idx) + a)], int(mask_array.max()))

        return [min(x_idx),max(x_idx)], [min(y_idx),max(y_idx)], [min(z_idx),max(z_idx)]


#x,y,z = find_mask_boundaries("/Users/jreventos/Desktop/TFM/tfm/patients_data/masks/val/mask_5.vtk")
#print(x,y,z)


def overall_mask_boundaries(masks_dir):
    """
     Function that iterates over each mask in the database to find the biggest mask boundaries and take it as reference
     to crop the volumes.

    :param labels_dir: masks directory

    :return: int lists --> [x_bottom_coord,x_top_coord], [y_bottom_coord, y_top_coord], [z_bottom_coord,z_top_coord]
    """
    mask_dirs = []
    for i in next(os.walk(masks_dir))[1]:
        name = os.path.join(masks_dir, i)

        for img_path in glob.glob(name + '/*.vtk'):
            mask_dirs.append(img_path)

    mask_dirs = sorted(mask_dirs, key=lambda x: float(re.findall("(\d+)", x)[0]))

    all_x_idx, all_y_idx, all_z_idx = [],[],[]
    for dir in mask_dirs:
        x_idx, y_idx, z_idx = find_mask_boundaries(dir)

        print(dir, x_idx,y_idx,z_idx)
        all_x_idx.extend(x_idx)
        all_y_idx.extend(y_idx)
        all_z_idx.extend(z_idx)


    all_x_idx = set(all_x_idx)
    all_y_idx = set(all_y_idx)
    all_z_idx = set(all_z_idx)

    print('Boundaries x axis:', min(all_x_idx), max(all_x_idx))
    print('Boundaries y axis:', min(all_y_idx), max(all_y_idx))
    print('Boundaries z axis:', min(all_z_idx), max(all_z_idx))

    return [min(all_x_idx), max(all_x_idx)], [min(all_y_idx), max(all_y_idx)], [min(all_z_idx), max(all_z_idx)]


# x,y,z = overall_mask_boundaries("/Users/jreventos/Desktop/TFM/tfm/patients_data/masks")
#mask_array = VtkReader("/Users/jreventos/Desktop/TFM/tfm/patients_data/masks/train/mask_44.vtk")
#DisplaySlices(mask_array[105:221,121:209,13:56], int(mask_array.max()))

def resize_volume(vol_array,x_idx = (0,221),y_idx = (131,209),z_idx=(0,56),error_percentage= 0.1):
    "Resize volumes from the paramteres extracted with overal_vol_crop_corners"

    # TODO : We didn't define the final MRI volume dimensions yet.

    return vol_array[x_idx[0]:x_idx[1], y_idx[0]:y_idx[1],z_idx[0]:z_idx[1]]


def compute_mean_std(path):
    """
    Mean and standard deviation of the hole MRI dataset.

    :param path: MRI volumes path
    :return: mean, std
    """

    vol_dirs = []
    for i in next(os.walk(path))[1]:
        name = os.path.join(path, i)
        for img_path in glob.glob(name + '/*.vtk'):
            vol_dirs.append(img_path)

    vol_dirs = sorted(vol_dirs, key=lambda x: float(re.findall("(\d+)", x)[0]))

    vol_arrays_list = []
    for dir in vol_dirs:
        vol_array = VtkReader(dir)
        vol_array = vol_array[:,:,0:58] # Not all volumes have the same number of slices, we took the minimum.
        vol_arrays_list.append(vol_array)

    # Compute mean and standard variation
    all_volumes = np.stack(list(map(np.float32, vol_arrays_list)), axis=0)
    mean = np.mean(all_volumes, axis=(0,1,2,3))
    std = np.std(all_volumes, axis=(0,1,2,3))
    print('Mean:',mean)
    print('Std',std)
    return mean, std


#path = '/Users/jreventos/Desktop/TFM/tfm/patients_data/MRI_volumes'
#compute_mean_std(path)
#vol_array = VtkReader("/Users/jreventos/Desktop/TFM/tfm/patients_data/MRI_volumes/val/MRI_19.vtk")

def volume_array_histogram(array):
    """Display a histogram of the values that contains the array.
    If the array is a volume will show the voxel intensities. """

    print(array.max())
    plt.hist(array.flatten(),histtype='step')
    plt.show()

#volume_array_histogram(vol_array)








