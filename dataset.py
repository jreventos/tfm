from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from utils.readers import VtkReader
from utils.patches import *
import numpy as np
import glob
import os
import re
from preprocessing import *
import random
import matplotlib.pyplot as plt
from utils import DataVisualization
import numpy as np





class BalancedPatchGenerator(Dataset):
    """
    FINAL DATASET CLASS
    """
    def __init__(self, path, patch_shape, positive_prob=0.7, shuffle_images=False, mode=None,
                 transform=None, large=True):

        self.path = path
        self.patch_shape = patch_shape
        self.positive_prob = positive_prob
        self.shuffle_images = shuffle_images
        self.mode = mode
        assert self.mode in ['train', 'val','test']
        self.images_files = self.get_files(self.path, self.mode, 'MRI_volumes',large)
        self.masks_files = self.get_files(self.path, self.mode, 'masks',large)
        self.transform = transform
        self.large = large

        if self.shuffle_images:
            self.images_files = np.random.permutation(self.images_files)

        assert len(self.images_files) == len(self.masks_files), 'Num.images is different from num. masks'

        self.num_patients = len(self.images_files)



        # Crop images
        #print(os.path.join(os.path.join(self.path, 'masks'), self.mode))
        #self.height, self.width, self.depth = overall_mask_boundaries(os.path.join(os.path.join(self.path, 'masks'), self.mode))

    @staticmethod
    def get_files(path, mode, type_files,large):
        large_dataset = [os.path.join(os.path.join(os.path.join(path, type_files), mode), file) for file in
                    os.listdir(os.path.join(os.path.join(path, type_files), mode)) if file.split('.')[-1] == 'npy']

        small_dataset = [file for file in large_dataset if 'ge' not in file]

        if large:
            return large_dataset
        else:
            return small_dataset


    # ATENCIO PER LLEGIR VTK POSAR 'VTK' AL == DE LA LINIA SUPERIOR

    #    @staticmethod
    #    def get_shape(image_location):
    #        image = VtkReader(image_location)
    #        return image.shape

    def get_patient_identifier(self, idx):
        image_path = self.images_files[idx]
        path, filename = os.path.split(image_path)
        name, extension = filename.split('.')
        identifier = int(name.split('_')[1])
        return identifier

    def get_image_mask_path(self, patient_id):
        img_pattern = re.compile('MRI_{}'.format(patient_id))
        image_path = list(filter(img_pattern.search, self.images_files))[0]
        mask_pattern = re.compile('mask_{}'.format(patient_id))
        mask_path = list(filter(mask_pattern.search, self.masks_files))[0]
        return image_path, mask_path

    @staticmethod
    def get_candidate_indexes(tensor, value, patch_size):
        idxs_list = []
        idxs_condition = np.where(tensor == value)
        #print(idxs_condition)
        #print(len(idxs_condition[2]))
        #print(tensor.shape)
        #print(len(tensor.shape))
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

        # get patient identifier:
        image_idx = idx
        patient_id = self.get_patient_identifier(image_idx)
        image_path, mask_path = self.get_image_mask_path(patient_id)


        # read arrays
        # image_original = VtkReader(image_path)
        # image_resized = resize_data(image_original,[360,360,60])
        # mask_original = VtkReader(mask_path)
        # mask_resized = resize_data(mask_original,[360,360,60])
        #
        #
        # image = image_resized[110:250, 110:250,10:]
        # mask = mask_resized[110:250, 110:250, 10:]

        #Values computed with the new dataset_patients([119, 260], [148, 260], [13, 59])
        #image = image_resized[110:265, 140:265,10:]
        #mask = mask_resized[110:265, 140:265, 10:]

        image = np.load(image_path)[110:265, 140:265,10:]
        mask = np.load(mask_path)[110:265, 140:265,10:]

        #without re-size old dataset_patients
        #image = VtkReader(image_path)[103:223, 110:230, 10:59]
        #mask = VtkReader(mask_path)[103:223, 110:230, 10:59]


        # compute a value between 0 and 1 to
        if self.mode == 'train':
            value = 1 if np.random.rand(1)[0] <= self.positive_prob else 0
            idxs = self.get_indexes(self, mask, value, self.patch_shape)

            # get patches
            im_patch = image[idxs]
            mask_patch = mask[idxs]
        else:
            im_patch = image
            mask_patch = mask

        # in case of transform (just for the train mode)
        if self.transform is not None:
            if self.large:
                im_patch = (1/468.0883)*(im_patch.astype(np.float32) - 158.37267)
            else:
                im_patch = (1 / 40.294086) * (im_patch.astype(np.float32) - 21.201036)


        # expand dimensions
        im_patch = np.expand_dims(im_patch, axis=0)
        mask_patch = np.expand_dims(mask_patch, axis=0)

        # tensor
        im_patch = torch.from_numpy(im_patch.astype(np.float32))
        mask_patch = torch.from_numpy(mask_patch.astype(np.float32))


        return im_patch, mask_patch




# if __name__ == "__main__":
#
#
#     path = "C:/Users/Roger/Desktop/JANA/tfm/patients_data"
#
#
#     ddd = BalancedPatchGenerator(path,
#                                  (60, 60, 32),
#                                  positive_prob=0.7,
#                                  shuffle_images=True,
#                                  mode='test',
#                                  transform=None)
#
#
#     loader = DataLoader(ddd, batch_size=1, shuffle=False)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     for i, data in enumerate(loader):
#         x, y_true = data
#         x.to(device)
#         y_true.to(device)
#         print(x.shape)
#         print(y_true.shape)
        # plt.figure("check", (18, 6))
        # plt.subplot(1, 2, 1)
        # plt.title(f"Image {i}")
        # plt.imshow(x[0, 0, :, :, 16], cmap="gray")
        # plt.subplot(1, 2, 2)
        # plt.title(f"label {i}")
        # plt.imshow(y_true[0, 0, :, :, 16])
        # plt.show()
        #
        # DisplaySlices(x[0, 0, :, :, :], int(x.max()))
        # DisplaySlices(y_true[0,0,:,:,:], int(1))
        #
        # print('Volume shape', i, ':', x.shape)
        # print('masks shape', i, ':', y_true.shape)


    # val_dataset = LeftAtriumSegmentationDataset(root_dir, transform=None, image_size=320, mode='val',
    #                                             random_sampling=False, seed=42, dimension="3D")
    #
    # print(val_dataset)
    # loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # print(len(loader))
    # for i, data in enumerate(loader):
    #     x, y_true = data
    #     print(i, x.shape)
    #     print(i, y_true.shape)
