from torch.utils.data import Dataset
import torch
from processing.preprocessing import *
import numpy as np
import re


class BalancedPatchGenerator(Dataset):
    """
        Dataset Pytorch generator for the LGE-MRI data.

        :path (string): path to the Clínic LA dataset (in numpy or vtk formats). The LGE-MRI data names should be 'MRI_#.npy/vtk",
                        and the ground truth "mask_#.npy/vtk. The following structure of folders inside the path is required:
                        - MRI_volumes:
                            - train
                            - val
                            - test
                        - masks
                            - train
                            - val
                            - test
        : patch_shape (list of integers): heigth x width x number of slices (only used for train set)
        : positive_prob (int): positive probability percentage, from 0 to 1 (only used for train set)
        : shuffle_images (bool): True or False
        : mode (string): train or val modes
        : transform (bool): True applies data normalization.
        : large (bool): False uses the small ClínicLA dataset, True the large ClínicLA dataset

    """
    def __init__(self, path, patch_shape, positive_prob=0.5, shuffle_images=True, mode=None,transform=True, large=True):


        self.path = path
        self.patch_shape = patch_shape
        self.positive_prob = positive_prob
        self.shuffle_images = shuffle_images
        self.mode = mode
        self.images_files = self.get_files(self.path, self.mode, 'MRI_volumes',large)
        self.masks_files = self.get_files(self.path, self.mode, 'masks',large)
        self.transform = transform
        self.large = large

        assert self.mode in ['train', 'val', 'test']

        if self.shuffle_images:
            self.images_files = np.random.permutation(self.images_files)

        assert len(self.images_files) == len(self.masks_files), 'Number of images is different from number of ground truth masks.'

        self.num_patients = len(self.images_files)



    @staticmethod
    def get_files(path, mode, type_files,large):
        "Get files directories."

        large_dataset = [os.path.join(os.path.join(os.path.join(path, type_files), mode), file) for file in
                    os.listdir(os.path.join(os.path.join(path, type_files), mode)) if file.split('.')[-1] == 'npy' or file.split('.')[-1] == 'vtk']
        small_dataset = [file for file in large_dataset if 'ge' not in file]

        if large:
            return large_dataset
        else:
            return small_dataset


    def get_patient_identifier(self, idx):
        " Extract sample number"
        image_path = self.images_files[idx]
        path, filename = os.path.split(image_path)
        name, extension = filename.split('.')
        identifier = int(name.split('_')[1])
        return identifier

    def get_image_mask_path(self, patient_id):
        " Get the LGE-MRI and masks path path given a patient id. "
        img_pattern = re.compile('MRI_{}'.format(patient_id))
        image_path = list(filter(img_pattern.search, self.images_files))[0]
        mask_pattern = re.compile('mask_{}'.format(patient_id))
        mask_path = list(filter(mask_pattern.search, self.masks_files))[0]
        return image_path, mask_path

    @staticmethod
    def get_candidate_indexes(tensor, value, patch_size):
        "Extract random candidate indexes in the three dimension of the masks volumes that satisfies the positive probability percentage parameter."
        idxs_list = []
        idxs_condition = np.where(tensor == value)
        position = np.random.randint(len(idxs_condition[0]))
        suggested_initial_points = [idxs_condition[i][position] for i in range(len(tensor.shape))]
        for i in range(len(suggested_initial_points)):
            ini = suggested_initial_points[i]
            total_pixels = tensor.shape[i]
            patch_pixels = patch_size[i]
            if ini + patch_pixels >= total_pixels:
                ini = np.maximum(total_pixels - patch_pixels - 1, 0)
            end = ini + patch_pixels
            idxs_list.append(slice(ini, end, 1))
        return tuple(idxs_list)

    @staticmethod
    def get_indexes(self, tensor, value, patch_size, num_attempts=3):
        " Compute the best candidate indexes for extracting the final volume patch from the volume. Three attemps are used and selects the best candidate."
        i = 0
        best_patch_coverage = 0
        best_idxs = None
        while i < num_attempts:
            idxs = self.get_candidate_indexes(tensor, value, patch_size)
            patch_candidate = tensor[idxs]
            patch_coverage = np.sum(patch_candidate == value) / np.prod(patch_candidate.shape)
            if patch_coverage > best_patch_coverage:
                best_patch_coverage = patch_coverage
                best_idxs = idxs
            i += 1
        return best_idxs

    def __len__(self):
        return self.num_patients

    def __getitem__(self, idx):

        # Get patient identifier and paths:
        image_idx = idx
        patient_id = self.get_patient_identifier(image_idx)
        image_path, mask_path = self.get_image_mask_path(patient_id)

        if image_path.split('.')[-1] == 'vtk':
            # ClínicLA dataset. Centered LA Crop: Values computed with the ClínicLA dataset([119, 260], [148, 260], [13, 59])
            image = VtkReader(image_path)[110:265, 140:265, 10:]
            mask = VtkReader(mask_path)[110:265, 140:265, 10:]

        else:
            # ClínicLA dataset. Centered LA Crop: Values computed with the ClínicLA dataset([119, 260], [148, 260], [13, 59])
            image = np.load(image_path, allow_pickle=True)[110:265, 140:265, 10:]
            mask = np.load(mask_path, allow_pickle=True)[110:265, 140:265, 10:]

            # Challenge competition. Centered crop LA:  [119, 254] [119, 271] [3, 59]
            # image = np.load(image_path,allow_pickle=True)[118:255, 118:272,2:]
            # mask_original = np.load(mask_path,allow_pickle=True)[118:255, 118:272,2:]
            # mask = np.where(mask_original == 255.0, 1, mask_original)

        # Get random patches
        if self.mode == 'train':
            value = 1 if np.random.rand(1)[0] <= self.positive_prob else 0  # Compute a random value between 0 and 1 to
            idxs = self.get_indexes(self, mask, value, self.patch_shape)    # Get best canidadate index

            # Extract patches
            im_patch = image[idxs]
            mask_patch = mask[idxs]
        else:
            im_patch = image
            mask_patch = mask

        if self.transform and self.mode == 'train':
            if self.large:
                im_patch = (1/468.0883)*(im_patch.astype(np.float32) - 158.37267)       # Large ClínicLA Dataset
                #im_patch = (1 / 20.710878) * (im_patch.astype(np.float32) - 13.553477) # Challenge dataset
            else:
                im_patch = (1 / 40.294086) * (im_patch.astype(np.float32) - 21.201036)  # Small ClínicLA dataset

        # Expand dimensions
        im_patch = np.expand_dims(im_patch, axis=0)
        mask_patch = np.expand_dims(mask_patch, axis=0)

        # Transform array to tensor
        im_patch = torch.from_numpy(im_patch.astype(np.float32))
        mask_patch = torch.from_numpy(mask_patch.astype(np.float32))

        return im_patch, mask_patch




