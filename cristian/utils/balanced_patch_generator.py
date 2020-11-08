from utils.readers import VtkReader
import os
import numpy as np
import re
import torch
from torch.utils.data import Dataset

class BalancedPatchGenerator(Dataset):
    def __init__(self, images_path, masks_path, patch_shape, positive_prob = 0.7, shuffle_images = False, is_test=False):
        self.images_path = images_path
        self.masks_path = masks_path
        self.patch_shape = patch_shape
        self.positive_prob = positive_prob
        self.shuffle_images = shuffle_images
        self.images_files = self.get_files(self.images_path)
        self.masks_files = self.get_files(self.masks_path)
        self.is_test = is_test
        
        if self.shuffle_images:
            self.images_files = np.random.permutation(self.images_files)
            
        assert len(self.images_files) == len(self.masks_files), 'Num.images is different from num. masks'
        
        self.num_patients = len(self.images_files)
        
    @staticmethod
    def get_files(path):
        return [os.path.join(path, file) for file in os.listdir(path) if file.split('.')[-1] == 'vtk']
        
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
        while i<num_attempts:
            idxs = self.get_candidate_indexes(tensor, value, patch_size)
            patch_candidate = tensor[idxs]
            patch_coverage = np.sum(patch_candidate == value)/np.prod(patch_candidate.shape)
            if patch_coverage > best_patch_coverage:
                best_patch_coverage = patch_coverage
                best_idxs = idxs
            i+=1
        return best_idxs
        
#    def __len__(self):
#        total_patients = self.num_patients
#        percentage_10 = int(np.minimum(np.maximum(np.ceil(0.1*total_patients), 2), total_patients))  
#        first_10_images = np.random.permutation(self.images_files)[:percentage_10]
#        images_shape = np.array([np.array(self.get_shape(image)) for image in first_10_images])
#        mean_shape = np.mean(images_shape, axis=0).astype(int)
#        total_dimensions = len(mean_shape)
#        total_patches = 1
#        for i in range(total_dimensions):
#            total_patches *= (mean_shape[i] - self.patch_shape[i] + 1)
#        return total_patches * self.num_patients
    def __len__(self):
        return self.num_patients
    
    def __getitem__(self, idx):
        #image_idx = idx % self.num_patients
        image_idx = idx
        patient_id = self.get_patient_identifier(image_idx)
        image_path, mask_path = self.get_image_mask_path(patient_id)
        image = VtkReader(image_path)
        mask = VtkReader(mask_path)
        value = 1 if np.random.rand(1)[0] <= self.positive_prob else 0
        idxs = self.get_indexes(self, mask, value, self.patch_shape)
        im_patch = image[idxs]
        im_patch = np.expand_dims(im_patch, axis=0)
        im_patch = torch.from_numpy(im_patch.astype(np.float32))
        mask_patch = mask[idxs]
        mask_patch = np.expand_dims(mask_patch, axis=0)
        mask_patch = torch.from_numpy(mask_patch.astype(np.float32))
        if self.is_test:
            return value, im_patch, mask_patch 
        else:
            return im_patch, mask_patch 
