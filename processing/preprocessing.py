
import re
import numpy as np
from skimage.exposure import rescale_intensity
from utils.DataVisualization import*
from utils.readers import*
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
from scipy.ndimage import rotate

def data_exploratory_analysis(path):
    ''' Complete data exploratory analysis of the dataset. It computes the following parameters:
    - Signal to noise ratio
    - Contrast Ratio
    - Homogeneity
    - Frequency of Intesnities

    Inputs:
    : path (string): dataset path
    '''

    splits = ['train','val','test']

    total_mri_directories = []
    total_mask_directories = []

    for set in splits:
        path_mri,_,names_mri = next(os.walk(os.path.join(path,os.path.join('MRI_volumes',set))))
        for mri in names_mri:
            path_mri_sample = os.path.join(path_mri,mri)
            total_mri_directories.append(path_mri_sample)

        path_mask, _, names_mask = next(os.walk(os.path.join(path, os.path.join('masks',set))))
        for mask in names_mask:
            path_mask_sample = os.path.join(path_mask,mask)
            total_mask_directories.append(path_mask_sample)

    assert len(total_mask_directories) == len(total_mri_directories)


    MRIs = sorted(total_mri_directories, key=lambda x: float(re.findall("(\d+)", x)[0]))
    MASKS= sorted(total_mask_directories, key=lambda x: float(re.findall("(\d+)", x)[0]))

    SNR_dataset, CR_dataset, HET_dataset = [], [], []
    total_foreground = 0
    total_background= 0
    intensities = np.zeros(1)

    for i in range(len(total_mri_directories)):
        #mri_array = VtkReader(MRIs[i])
        #mask_array = VtkReader(MASKS[i])

        mri_array = np.load(MRIs[i])
        mask_array = np.load(MASKS[i])

        #flatten array
        flat_mri = mri_array.flatten()
        intensities = intensities + flat_mri


        foreground = mri_array[np.where(mask_array == mask_array.max())]
        total_foreground+=len(foreground)
        background = mri_array[np.where(mask_array == mask_array.min())]
        total_background+=len(background)

        mean_LA_pixels = np.mean(foreground)
        mean_background_pixels = np.mean(background)

        std_LA_pixels = np.std(foreground)
        std_background_pixels = np.std(background)

        # Signal To Noise Ratio:
        SNR = abs(mean_LA_pixels-mean_background_pixels)/std_background_pixels
        SNR_dataset.append(SNR)
        # Contrast ratio
        CR = mean_LA_pixels/mean_background_pixels
        CR_dataset.append(CR)
        # Heterogeneity
        HET = std_LA_pixels/abs(mean_LA_pixels-mean_background_pixels)
        HET_dataset.append(HET)

        from matplotlib import pyplot as plt

        if SNR>=4 and 5<CR<=6:
            plt.imshow(rotate(mri_array[118:272, 118:272,25],-90),cmap='gray')
            plt.xticks(color='w')
            plt.yticks(color='w')
            plt.ylabel('SNR>4  5<CR<6 HET= '+str(round(HET,1)))
            plt.show()
            print('SNR>=4 and 5<CR<=6')
            print(MRIs[i])

        elif 3<=SNR<4 and 4<CR<=5:
            plt.imshow(rotate(mri_array[118:272, 118:272,25],-90),cmap='gray')
            plt.xticks(color='w')
            plt.yticks(color='w')
            plt.ylabel('3<SNR<4  4<CR<5 HET= '+str(round(HET,1)))
            plt.show()
            print('3<=SNR<4 and 4<CR<=5')
            print(MRIs[i])

        elif 2<=SNR<3 and 3<CR<=4:
            plt.imshow(rotate(mri_array[118:272, 118:272,25],-90),cmap='gray')
            plt.xticks(color='w')
            plt.yticks(color='w')
            plt.ylabel('2<SNR<3  3<CR<4 HET= '+str(round(HET,1)))
            plt.show()
            print('2<=SNR<3 and 3<CR<=4')
            print(MRIs[i])

        elif 1<=SNR<2 and 2<CR<=3:
            plt.imshow(rotate(mri_array[118:272, 118:272,25],-90),cmap='gray')
            plt.xticks(color='w')
            plt.yticks(color='w')
            plt.ylabel('1<SNR<2  2<CR<3 HET= '+str(round(HET,1)))
            plt.show()
            print('1<=SNR<2 and 2<CR<=3')
            print(MRIs[i])

        elif SNR<1 and 1<CR<=2:
            plt.imshow(rotate(mri_array[118:272, 118:272,25],-90),cmap='gray')
            plt.xticks(color='w')
            plt.yticks(color='w')
            plt.ylabel('SNR<1  1<CR<2 HET= '+str(round(HET,1)))
            plt.show()
            print('SNR<1 and 1<CR<=2')
            print(MRIs[i])


        #print('SNR = ', SNR, 'CR = ', CR, 'HET = ', HET)


    plt.hist(intensities)
    plt.xlabel('Intensity value')
    plt.ylabel('Frequency')
    plt.title('Large Clínic LA LGE-MRIs intensities')
    plt.show()

    plt.bar(['background (N = '+ str(total_background)+')','foreground (N = '+ str(total_foreground)+')'],[total_background,total_foreground])
    plt.title('Background/Foreground = '+str(round(total_background/total_foreground,3)))
    plt.ylabel('# Samples')
    plt.show()

    SNR_bin_1 = [i for i in SNR_dataset if i<1 ]
    SNR_bin_2 = [i for i in SNR_dataset if i < 2 and i>= 1]
    SNR_bin_3 = [i for i in SNR_dataset if i < 3 and i>= 2 ]
    SNR_bin_4 = [i for i in SNR_dataset if i < 4 and i >= 3]
    SNR_bin_5 = [i for i in SNR_dataset if i >= 4]

    SNR_bins = [len(SNR_bin_1),len(SNR_bin_2),len(SNR_bin_3),len(SNR_bin_4),len(SNR_bin_5)]
    bins_names = ['< 1','1-2','2-3','3-4','> 4']
    plt.bar(bins_names,SNR_bins,color=['red','blue','green','purple','orange'])
    plt.title('SNR Large ClínicLA Dataset')
    plt.grid('True')
    plt.show()

    plt.scatter(SNR_dataset,CR_dataset)
    plt.xticks(color='w')
    plt.yticks(color = 'w')
    plt.xlabel('Signal-Noise Ratio')
    plt.ylabel('Contrast Ratio')
    plt.title('Large ClínicLA Dataset')
    plt.show()

    plt.scatter(SNR_dataset,HET_dataset)
    plt.xticks(color='w')
    plt.yticks(color = 'w')
    plt.xlabel('Signal-Noise Ratio')
    plt.ylabel('Heterogeneity')
    plt.title('Large ClínicLA Dataset')
    plt.show()

    plt.scatter(CR_dataset,HET_dataset)
    plt.xticks(color='w')
    plt.yticks(color = 'w')
    plt.xlabel('Contrast Ratio')
    plt.ylabel('Heterogeneity')
    plt.title('Large ClínicLA Dataset')
    plt.show()


def resize_data(data,dimensions):
    '''Resize input data

    Input:
    : data (array): array-like format
    : dimensions (list): dimensions to resize the data

    Return: resized data in array-like format
    '''
    initial_size_x = data.shape[0]
    initial_size_y = data.shape[1]
    initial_size_z = data.shape[2]

    new_size_x = dimensions[0]
    new_size_y = dimensions[1]
    new_size_z = dimensions[2]

    delta_x = initial_size_x / new_size_x
    delta_y = initial_size_y / new_size_y
    delta_z = initial_size_z / new_size_z

    new_data = np.zeros((new_size_x, new_size_y, new_size_z))

    for x, y, z in itertools.product(range(new_size_x),
                                     range(new_size_y),
                                     range(new_size_z)):
        new_data[x][y][z] = data[int(x * delta_x)][int(y * delta_y)][int(z * delta_z)]

    return new_data



def standarize(vol,outliers = None):
    """
    Function that rescales data to have a mean (𝜇) of 0 and standard deviation (𝜎) of 1 (unit variance).
    If outliers, computes the 10-th and 99-th percentiles of the volume in a flattened version of the array and
    rescales the volume from the 10-th to 99-th range. It is a way to increase the contrast of the MRI_volumes image.

    Input:
    :param vol (array): 3D numpy array-like
    :param outliers (bool): True computes outliers

    Return: standarized 3D numpy array
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



def custom_normalize(array,mean,sd):
    """
    Volume normalization by means of mean and sd.

    Inputs:
    :param array (ndarray):  [H,W,slices]
    :param mean (int): overall mean
    :param sd (int): overall standard deviation

    Return: normalized array in each channel
    """
    return (array - mean) / sd




def normalize(vol):
    '''
    Array normalization by its meand and std.

    Input:
    : vol (array): array like format

    Return: normalized input
    '''

    if (vol.max() - vol.min()) != 0:
        norm_vol = (vol - vol.min()) / (vol.max() - vol.min())
    else:
        norm_vol = (vol - vol.min())
    return norm_vol

# array = VtkReader("/Users/jreventos/Desktop/TFM/tfm/patients_data2/MRI_volumes/val/MRI_1.vtk")
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
    Uses the mask of a single MRI_volumes volume to find the boundaries (top and bottom) of the mask in the XYZ axis.

    Input:
    :param path: mask path

    Return: int lists --> [x_bottom_coord,x_top_coord], [y_bottom_coord, y_top_coord], [z_bottom_coord,z_top_coord]
    """

    #mask_array = VtkReader(path)
    mask_array = np.load(path)

    # Find coordinates in the 3 axis
    max_val = mask_array.max()
    x_idx = [i for i in range(mask_array.shape[0]) if max_val in mask_array[i, :, :]]
    y_idx = [i for i in range(mask_array.shape[1]) if max_val in mask_array[:, i, :]]
    z_idx = [i for i in range(mask_array.shape[-1]) if max_val in mask_array[:,:,i]]

    #DisplaySlices(mask_array[(min(x_idx) - a):(max(x_idx) + a), (min(y_idx) - a):(max(y_idx) + a), (min(z_idx) - a):(max(z_idx) + a)], int(mask_array.max()))

    return [min(x_idx),max(x_idx)], [min(y_idx),max(y_idx)], [min(z_idx),max(z_idx)]

def find_mask_boundaries_from_array(mask_array):
    """
    Uses the mask of a single MRI_volumes volume to find the boundaries (top and bottom) of the mask in the XYZ axis.

    Input:
    :param path: mask path

    Return: int lists --> [x_bottom_coord,x_top_coord], [y_bottom_coord, y_top_coord], [z_bottom_coord,z_top_coord]
    """

    # Find coordinates in the 3 axis
    max_val = mask_array.max()
    x_idx = [i for i in range(mask_array.shape[0]) if max_val in mask_array[i, :, :]]
    y_idx = [i for i in range(mask_array.shape[1]) if max_val in mask_array[:, i, :]]
    z_idx = [i for i in range(mask_array.shape[-1]) if max_val in mask_array[:,:,i]]
    #a = 0
    #DisplaySlices(mask_array[(min(x_idx) - a):(max(x_idx) + a), (min(y_idx) - a):(max(y_idx) + a), (min(z_idx) - a):(max(z_idx) + a)], int(mask_array.max()))

    return [min(x_idx),max(x_idx)], [min(y_idx),max(y_idx)], [min(z_idx),max(z_idx)]


def overall_mask_boundaries(dir_mask):
    """
     Function that iterates over each mask in the database to find the biggest mask boundaries and take it as reference
     to crop the volumes.

    Input:
    :param labels_dir: masks directory

    Return:int lists --> [x_bottom_coord,x_top_coord], [y_bottom_coord, y_top_coord], [z_bottom_coord,z_top_coord]
    """

    path_root, _, mask_names = next(os.walk(dir_mask))
    masks= []
    for i in mask_names:
        name = os.path.join(path_root,i)
        masks.append(name)

    mask_dirs = sorted(masks, key=lambda x: float(re.findall("(\d+)", x)[0]))

    all_x_idx, all_y_idx, all_z_idx = [],[],[]
    for dir in mask_dirs:

        x_idx, y_idx, z_idx = find_mask_boundaries(dir,bad_mask=[])
        print(dir,y_idx,z_idx)
        all_x_idx.extend(x_idx)
        all_y_idx.extend(y_idx)
        all_z_idx.extend(z_idx)

    all_x_idx = set(all_x_idx)
    all_y_idx = set(all_y_idx)
    all_z_idx = set(all_z_idx)

    return [min(all_x_idx), max(all_x_idx)], [min(all_y_idx), max(all_y_idx)], [min(all_z_idx), max(all_z_idx)]


def compute_mean_std(path):
    """
    Mean and standard deviation of the hole MRI_volumes ClinicLA_dataset.

    :param path: MRI_volumes volumes path
    :return: mean, std
    """

    vol_dirs = os.listdir(path)
    # for i in next(os.walk(path))[1]:
    #     name = os.path.join(path, i)
    #     for img_path in glob.glob(name + '/*.npy'):
    #         vol_dirs.append(img_path)

    vol_dirs = sorted(vol_dirs, key=lambda x: float(re.findall("(\d+)", x)[0]))
    print(vol_dirs)
    vol_arrays_list = []
    for dir in vol_dirs:
        numpy_dir = path +'/'+dir
        print(numpy_dir)
        vol_array = np.load(numpy_dir)
        print(vol_array)
        #vol_array = resize_volume(vol_array,[360,360,60])
        #vol_array = vol_array[:,:,:] # Not all volumes have the same number of slices, we took the minimum.
        vol_arrays_list.append(vol_array)

    # Compute mean and standard variation
    all_volumes = np.stack(list(map(np.float32, vol_arrays_list)), axis=0)
    mean = np.mean(all_volumes, axis=(0,1,2,3))
    std = np.std(all_volumes, axis=(0,1,2,3))
    print('Mean:',mean)
    print('Std',std)
    return mean, std










