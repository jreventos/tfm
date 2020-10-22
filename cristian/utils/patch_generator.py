from utils.readers import VtkReader
import os
import numpy as np
import functools
import collections
import re
from torch.utils.data import Dataset

class GeneratePatches(Dataset):
    def __init__(self, images_path, masks_path, patch_shape, shuffle_images = False, allow_overlap = True):
        self.images_path = images_path
        self.masks_path = masks_path
        self.patch_shape = patch_shape
        self.shuffle_images = shuffle_images
        self.allow_overlap = allow_overlap
        self.images_files = self.get_files(self.images_path)
        self.masks_files = self.get_files(self.masks_path)
        
        if self.shuffle_images:
            self.images_files = np.random.permutation(self.images_files)
            self.masks_files = np.random.permutation(self.masks_files)
            
        
        assert len(self.images_files) == len(self.masks_files), 'Num.images is different from num. masks'
        
        self.patients_information = self.get_patients_information()
        self.num_patients = len(self.patients_information)
    
    @staticmethod
    def get_files(path):
        return [os.path.join(path, file) for file in os.listdir(path) if file.split('.')[-1] == 'vtk']  

    @staticmethod
    # This could be done in a better way if we get the shape without opening the image
    def get_shape_image(file):
        return VtkReader(file).shape
    
    @staticmethod
    def calculate_num_patches(image_shape, patch_shape):
        return functools.reduce(lambda x, y: x*y, (map(lambda x, y: int(np.ceil(x/y)), image_shape, patch_shape)))
        
    def get_patients_information(self):
        patient_info = collections.namedtuple('patient_info', 'patient_id image_location mask_location shape num_patches')
        list_patient_info = []
        for file in self.images_files:
            image_path = file
            patient_id = int(file.split('/')[-1].split('.')[0].split('_')[1])
            pattern = re.compile('mask_{}'.format(patient_id))
            mask_path = list(filter(pattern.search, self.masks_files))[0]
            shape = self.get_shape_image(file)
            num_patches = self.calculate_num_patches(shape, self.patch_shape)
            list_patient_info.append(patient_info(patient_id, image_path, mask_path, shape, num_patches))
            
        return list_patient_info
    
    @staticmethod
    def get_indexes(patients_info, idx):
        patch_id = 0
        image_id = 0
        found = False
        cumulative_patches = 0
        i = 0
        while i<len(patients_info) and not found:
            current_patient = patients_info[i]
            if(cumulative_patches + current_patient.num_patches > idx):
                found = True
            else:
                cumulative_patches += current_patient.num_patches
                i += 1
            
        patch_id = idx - cumulative_patches
        return (i, patch_id)
        
    
    def __len__(self):
        total_len = 0
        for patient in self.patients_information:
            total_len += patient.num_patches
        return total_len
    
    def __getitem__(self, idx):
        image_idx, patch_id = self.get_indexes(self.patients_information, idx) 
        this_patient = self.patients_information[image_idx]
        num_patches = this_patient.num_patches
        num_channels, num_rows, num_cols = this_patient.shape
        
        total_tensors_row = int(np.ceil(num_cols/self.patch_shape[2]))
        total_tensors_col = int(np.ceil(num_rows/self.patch_shape[1]))
        total_tensors_channel = total_tensors_row * total_tensors_col
        
        start_row, start_col = divmod(patch_id%total_tensors_channel, total_tensors_row) 
        
        start_row = min(self.patch_shape[1] * start_row, this_patient.shape[1])
        end_row = min(start_row + self.patch_shape[1], this_patient.shape[1])
        
        if(self.allow_overlap and end_row - start_row < self.patch_shape[1] and end_row == this_patient.shape[1]):
            start_row = end_row - self.patch_shape[1]

        start_col = min(self.patch_shape[2]*start_col, this_patient.shape[2])
        end_col = min(start_col + self.patch_shape[2], this_patient.shape[2])
        
        if(self.allow_overlap and end_col - start_col < self.patch_shape[2] and end_col == this_patient.shape[2]):
            start_col = end_col - self.patch_shape[2]

        start_channel, _ = divmod(patch_id, total_tensors_channel)
        start_channel = min(start_channel * self.patch_shape[0], this_patient.shape[0]) 
        end_channel = min(start_channel + self.patch_shape[0], this_patient.shape[0])
        
        if(self.allow_overlap and end_channel - start_channel < self.patch_shape[0] and end_channel == this_patient.shape[0]):
            end_channel = start_channel - self.patch_shape[0]
        
        image = VtkReader(this_patient.image_location)
        mask = VtkReader(this_patient.mask_location)
        
        image_patch = image[start_channel:end_channel, start_row:end_row, start_col:end_col]
        mask_patch = mask[start_channel:end_channel, start_row:end_row, start_col:end_col]
        
        print('image_id:', this_patient.patient_id, 'patch_id:', patch_id)
        return (image_patch, mask_patch)
