import glob
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

#from Model import config


def create_distance_map(mask_path):

    np_mask = np.load(mask_path)
    np_mask_invert = np_mask == 0
    np_dist = distance_transform_edt(np_mask,sampling=reversed([1.25,1.25,2.5]))
    np_dist_invert = distance_transform_edt(np_mask_invert,sampling=reversed([1.25,1.25,2.5]))
    np_dist = [y - x for x, y in zip(np_dist, np_dist_invert)]
    dist = sitk.GetImageFromArray(np_dist)
    #dist.CopyInformation(mask)
    #sitk.WriteImage(dist, 'distance_mask', True)
    return np_dist

def run_create_distance_map(data_folder):
    for subject_folder in glob.glob(os.path.join(data_folder, "*")):
        create_distance_map(subject_folder)

#data_folder = config["data_folder"]
#a = run_create_distance_map(data_folder)



import distancemap
from distancemap import distance_map_from_binary_matrix



a= create_distance_map("Dataset_arrays/masks/train/mask_2.npy")
a = np.array(a)

np_mask = np.load("Dataset_arrays/masks/train/mask_2.npy")
b= distance_map_from_binary_matrix(np_mask[:,:,20])

for i in a[:,:,20]:
    for j in i:
        print(j)

import matplotlib.pyplot as plt
plt.imshow(a[:,:,20], cmap='gray')
plt.show()

import torch
import torch.nn.functional as F

def contour_loss():
    sobelFilters = np.array([
                         [ [ [-1, -2, -1], [-2, -4, -2], [-1, -2, -1] ],
                                [ [0, 0, 0], [0, 0, 0], [0, 0, 0] ],
                                 [ [1, 2, 1], [2, 4, 2], [1, 2, 1] ] ],
                         [ [ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ],
                                [ [2, 4, 2], [0, 0, 0],  [-2, -4, -2] ],
                                 [ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ] ],
                         [ [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ],
                                [ [-2, 0, 2], [-4, 0, 4], [-2, 0, 2] ],
                                 [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ] ]
                            ])


    contour = K.sum(K.concatenate(
        [K.abs(K.conv3d(y_pred, sobelFilters[0], padding='same', data_format='channels_first')),
         K.abs(K.conv3d(y_pred, sobelFilters[1], padding='same', data_format='channels_first')),
         K.abs(K.conv3d(y_pred, sobelFilters[2], padding='same', data_format='channels_first'))]
        , axis=0), axis=0)
    contour_f = K.batch_flatten(contour)
    y_true_f = K.batch_flatten(K.abs(y_true) - config["drain"])

    finalChamferDistanceSum = K.sum(contour_f * y_true_f, axis=1, keepdims=True)

    return K.mean(finalChamferDistanceSum)

x = torch.randn([1, 2, 360, 360, 60])
sobel = np.array([
    [[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
     [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
     [[1, 2, 1], [2, 4, 2], [1, 2, 1]]],
    [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
     [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
     [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
    [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
     [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
     [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
])
depth = x.size()[4]
channels = x.size()[1]
sobel_kernel = torch.tensor(sobel, dtype=torch.float32)
print(sobel_kernel.shape)
print(x.shape)
malignacy = F.conv3d(x, sobel_kernel[0])
print(malignacy.shape)

