
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from tfm.utils.readers import VtkReader
from tfm.utils.patches import *
import numpy as np
import glob
import os
import re
import collections
import functools
import random
import matplotlib.pyplot as plt
from tfm.utils import DataVisualization

class PatchesDataset(Dataset):

    def __init__(
            self,
            root_dir,
            transform=None,
            patch_function=None,
            patch_dim=None,
            patch_stepsize = None,
            mode=None,
            random_sampling=False):

        self.root_dir = root_dir
        self.patch_function = patch_function
        self.patch_dim = patch_dim
        self.patch_stepsize = patch_stepsize
        self.random_sampling = random_sampling
        self.transform = transform
        self.mode = mode

        assert self.mode in ['train', 'val', 'test']

        # Read images
        print('Reading {} images'.format(self.mode))
        patient_ids = None

        for i in next(os.walk(self.root_dir))[1]:
            name = os.path.join(self.root_dir, i)
            name_set = os.path.join(name, self.mode)
            if "masks" in name:
                mask_dirs = [mask_path for mask_path in
                             sorted(glob.glob(name_set + '/*.vtk'), key=lambda x: float(re.findall("(\d+)", x)[0]))]
                mask_arrays = [VtkReader(mask_path)[103:223,110:230,10:59] for mask_path in mask_dirs]
            else:
                volumes_dirs = [vol_path for vol_path in
                                sorted(glob.glob(name_set + '/*.vtk'), key=lambda x: float(re.findall("(\d+)", x)[0]))]

                patient_ids = [vol_dir.split('/')[-1] for vol_dir in volumes_dirs]
                volumes_arrays = [VtkReader(vol_path)[103:223,110:230,10:59] for vol_path in volumes_dirs]

        volumes = {}
        masks = {}
        for i in range(len(patient_ids)):
            volumes[patient_ids[i]] = volumes_arrays[i]
            masks[patient_ids[i]] = mask_arrays[i]

        # List of volumes and masks
        self.patients_vol_mask = [(volumes[i], masks[i]) for i in patient_ids]

        # Create volume & mask patches

        self.patient_patches = {}
        for i in range(len(self.patients_vol_mask)):
            v, m = self.patients_vol_mask[i]
            v_patches = self.patch_function(v, self.patch_dim, self.patch_stepsize)  # extract MRI volume patches
            m_patches = self.patch_function(m, self.patch_dim, self.patch_stepsize)  # extract mask volume patches
            self.patient_patches[i] = v_patches, m_patches  # save vol & mask patches per patient

        # Create patch index
        self.num_patches = len(v_patches)
        self.patient_patch_idx = [(patient_idx, patch_idx) for patient_idx in range(len(self.patients_vol_mask)) for
                                  patch_idx in range(self.num_patches)]
        print('Num patients:', len(self.patients_vol_mask))
        print('Num patches per patient', self.num_patches)

    def __len__(self):

        return len(self.patient_patch_idx)  # iteration by slice

    def __getitem__(self, idx):
        # Read patient index and slice index
        patient_idx = self.patient_patch_idx[idx][0]
        patch_idx = self.patient_patch_idx[idx][1]

        if self.random_sampling:
            patient_idx = np.random.randint(len(self.patients_vol_mask))  # choose a random patient
            patch_idx = np.random.randint(self.num_patches)  # choose a random patch

        # Extract volume & mask patch
        vol_patch = self.patient_patches[patient_idx][0][patch_idx]
        mask_patch = self.patient_patches[patient_idx][1][patch_idx]

        # Add channel dimension in volumes and masks
        vol_patch = vol_patch[None, :, :, :]
        mask_patch = mask_patch[None, :, :, :]

        # Apply transforms to patches
        if self.transform is not None:
            vol_patch, mask_patch = self.transform((vol_patch, mask_patch))

        # Create volume & mask tensors
        vol_tensor = torch.from_numpy(vol_patch.astype(np.float32))
        mask_tensor = torch.from_numpy(mask_patch.astype(np.float32))

        return vol_tensor, mask_tensor




class PatchesDataset_2(Dataset):

    def __init__(
            self,
            root_dir,
            transform=None,
            patch_dim=None,
            num_patches = None,
            mode=None,
            random_sampling=False,):

        self.root_dir = root_dir
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.random_sampling = random_sampling
        self.transform = transform
        self.patient_ids = None
        self.volume_dirs = None
        self.masks_dirs = None
        self.mode = mode

        assert self.mode in ['train', 'val', 'test']

        # Read images
        print('Reading {} images'.format(self.mode))


        for i in next(os.walk(self.root_dir))[1]:
            name = os.path.join(self.root_dir, i)
            name_set = os.path.join(name, self.mode)
            if "masks" in name:
                self.masks_dirs = [mask_path for mask_path in glob.glob(name_set + '/*.vtk')]
                self.masks_dirs.sort(key=lambda f: int(re.sub('\D', '', f)))
            else:
                self.volumes_dirs = [vol_path for vol_path in glob.glob(name_set + '/*.vtk')]
                self.volumes_dirs.sort(key=lambda f: int(re.sub('\D', '', f)))

                self.patient_ids = [vol_dir.split('/')[-1] for vol_dir in self.volumes_dirs]


        # Create patch index
        self.patient_patch_idx = [(patient_idx, patch_idx) for patient_idx in range(len(self.patient_ids)) for
                                  patch_idx in range(self.num_patches)]


        #print('Vol:',self.volumes_dirs)
        #print('Masks:',self.masks_dirs)
    @staticmethod
    def calculate_patch_idx(vol_shape,patch_dim):
        x_corner, y_corner, z_corner = [random.randint(np.ceil(ps/2), vs-(np.ceil(ps/2))) for vs, ps in zip(vol_shape,patch_dim)]
        d_depth, d_height, d_width = [int(np.ceil(d / 2)) for d in patch_dim]
        return [(x_corner - d_depth,x_corner + d_depth), (y_corner - d_height,y_corner + d_height),
                (z_corner - d_width,z_corner + d_width)]

    @staticmethod
    def calculate_patch(array,patch_idx):
        x_idx, y_idx, z_idx = patch_idx
        return array[x_idx[0]:x_idx[1],y_idx[0]:y_idx[1],z_idx[0]:z_idx[1]]


    def __len__(self):

        return len(self.patient_patch_idx)  # iteration by slice

    def __getitem__(self, idx):
        # Read patient index and slice index
        patient_idx = self.patient_patch_idx[idx][0]
        if self.random_sampling:
            idx_rand = np.random.randint(len(self.patient_ids))  # choose a random patient
            patient_idx = self.patient_patch_idx[idx_rand][0]


        # Read arrays
        volume_array = VtkReader(self.volumes_dirs[patient_idx])[103:223, 110:230, 10:59]
        #print(self.volumes_dirs[patient_idx])
        #DisplaySlices(volume_array, int(volume_array.max()))
        mask_array = VtkReader(self.masks_dirs[patient_idx])[103:223, 110:230, 10:59]
        #print(self.masks_dirs[patient_idx])
        #DisplaySlices(mask_array, int(mask_array.max()))


        # Compute shape
        vol_shape = volume_array.shape

        # Compute patch index
        patch_idx = self.calculate_patch_idx(vol_shape,self.patch_dim)
        #print(patch_idx)
        # Crop patches
        vol_patch = self.calculate_patch(volume_array, patch_idx)
        mask_patch = self.calculate_patch(mask_array, patch_idx)

        # Apply transforms to patches
        if self.transform is not None:
            vol_patch = self.transform(vol_patch.astype(np.float32))
            #mask_patch = self.transform(mask_patch.astype(np.float32))

        # If validation mode no patch is extracted
        # if self.mode == 'val':
        #     vol_patch = volume_array
        #     mask_patch = mask_array

        # Add channel dimension in volumes and masks
        vol_patch = vol_patch[None, :, :, :]
        mask_patch = mask_patch[None, :, :, :]

        # Create volume & mask tensors
        vol_tensor = torch.from_numpy(vol_patch.astype(np.float32))
        mask_tensor = torch.from_numpy(mask_patch.astype(np.float32))

        return vol_tensor, mask_tensor




class LeftAtriumSegmentationDataset(Dataset):

    def __init__(
            self,
            root_dir,
            transform=None,
            image_size = 320,
            mode=None,
            random_sampling = False,
            seed= 42,
            dimension = "3D"):

        self.root_dir = root_dir
        self.dimension = dimension
        assert mode in ['train','val','test']
        assert dimension in ['2D', '3D']

        # Read images
        volumes = {}
        masks = {}

        print('Reading {} images'.format(mode))
        patient_ids = None
        image_slices = None
        for i in next(os.walk(self.root_dir))[1]:
            name = os.path.join(self.root_dir, i)
            name_set = os.path.join(name, mode)
            if "masks" in name:
                mask_dirs = [img_path for img_path in
                             sorted(glob.glob(name_set + '/*.vtk'), key=lambda x: float(re.findall("(\d+)", x)[0]))]
                mask_slices = [VtkReader(img_path)[103:223,110:230,10:59] for img_path in mask_dirs]
            else:
                image_dirs = [img_path for img_path in
                              sorted(glob.glob(name_set + '/*.vtk'), key=lambda x: float(re.findall("(\d+)", x)[0]))]

                patient_ids = [img_dir.split('/')[-1] for img_dir in image_dirs]
                image_slices = [VtkReader(img_path)[103:223,110:230,10:59] for img_path in image_dirs]

        for i in range(len(patient_ids)):
            volumes[patient_ids[i]] = image_slices[i]
            masks[patient_ids[i]] = mask_slices[i]

        self.patients = volumes

        # Preprocessing steps
        print('Preprocessing {} volumes ...'.format(mode))

        # TODO: here we could add some preprocessing steps for each volume.

        # List of volumes and tuples
        self.volumes = [(volumes[i],masks[i]) for i in self.patients]

        # Add channel dimension in volumes and masks
        self.volumes = [(v[None,:,:,:], m[None,:,:,:]) for (v, m) in self.volumes]

        print("done creating {} dataset".format(mode))

        num_slices = [v.shape[-1] for v, m in self.volumes]
        if self.dimension == "2D":
            # Create patient index with associated slice index (p_id,slice_id)
            self.patient_slice_index = list(
                zip(
                    sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                    sum([list(range(x)) for x in num_slices], []),
                )
            )
        else:
            # Create paient index
            self.patient_slice_index = list(range(0,len(self.volumes)))

        print('Total slices:',sum(num_slices))
        print('Slices values:',num_slices)

        self.random_sampling = random_sampling
        self.transform = transform




    def __len__(self):

        return len(self.patient_slice_index) # iteration by slice


    def __getitem__(self, idx):
        # Read patient index and slice index

        if self.dimension == "2D":
            patient = self.patient_slice_index[idx][0]
            slice_n = self.patient_slice_index[idx][1]
        else:
            patient = self.patient_slice_index[idx]


        if self.random_sampling:
            patient = np.random.randint(len(self.volumes)) # choose a random patient
            if self.dimension == '2D':
                slice_n = np.random.choice(range(self.volumes[patient][0].shape[0]))


        v,m = self.volumes[patient]

        if self.dimension == "2D":
            image = v[:,:,:,slice_n]
            mask = m[:, :, :, slice_n]

        else:
            image = v
            mask = m


        if self.transform is not None:
            image, mask = self.transform((image, mask))

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))


        return image_tensor, mask_tensor


class BalancedPatchGenerator(Dataset):
    def __init__(self, path, patch_shape, positive_prob=0.7, shuffle_images=False, mode=None, is_test=False,
                 transform=None):

        self.path = path
        self.patch_shape = patch_shape
        self.positive_prob = positive_prob
        self.shuffle_images = shuffle_images
        self.mode = mode
        self.is_test = is_test
        assert self.mode in ['train', 'val']
        self.images_files = self.get_files(self.path, self.mode, 'MRI_volumes')
        self.masks_files = self.get_files(self.path, self.mode, 'masks')
        self.transform = transform

        if self.shuffle_images:
            self.images_files = np.random.permutation(self.images_files)

        assert len(self.images_files) == len(self.masks_files), 'Num.images is different from num. masks'

        self.num_patients = len(self.images_files)

    @staticmethod
    def get_files(path, mode, type_files):
        return [os.path.join(os.path.join(os.path.join(path, type_files), mode), file) for file in
                os.listdir(os.path.join(os.path.join(path, type_files), mode)) if file.split('.')[-1] == 'vtk']

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
        image = VtkReader(image_path)[103:223, 110:230, 10:59]
        mask = VtkReader(mask_path)[103:223, 110:230, 10:59]

        # compute a value between 0 and 1 to
        value = 1 if np.random.rand(1)[0] <= self.positive_prob else 0
        idxs = self.get_indexes(self, mask, value, self.patch_shape)

        # get patches
        im_patch = image[idxs]
        mask_patch = mask[idxs]

        # in case of transform (just for the train mode)
        if self.transform is not None:
            im_patch = self.transform(im_patch.astype(np.float32))

        # expand dimensions
        im_patch = np.expand_dims(im_patch, axis=0)
        mask_patch = np.expand_dims(mask_patch, axis=0)

        # tensor
        im_patch = torch.from_numpy(im_patch.astype(np.float32))
        mask_patch = torch.from_numpy(mask_patch.astype(np.float32))

        if self.is_test:
            return value, im_patch, mask_patch
        else:
            return im_patch, mask_patch






if __name__ == "__main__":


    path = "/Users/jreventos/Desktop/TFM/tfm/patients_data2"
    # val_dataset = PatchesDataset_2(path,
    #                              transform=None,
    #                              patch_dim=[60, 60, 36],
    #                              num_patches=1,
    #                              mode='val',
    #                              random_sampling=False)
    #
    #
    # loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    ddd = BalancedPatchGenerator(path,
                                 (32, 32, 32),
                                 positive_prob=0.7,
                                 shuffle_images=True,
                                 mode='val',
                                 is_test=False,
                                 transform=None)

    loader = DataLoader(ddd, batch_size=1, shuffle=False)




    for i, data in enumerate(loader):
        x, y_true = data
        print(x.shape)
        print(y_true.shape)
        plt.figure("check", (18, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"Image {i}")
        plt.imshow(x[0, 0, :, :, 16], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title(f"label {i}")
        plt.imshow(y_true[0, 0, :, :, 16])
        plt.show()

        DisplaySlices(x[0, 0, :, :, :], int(x.max()))
        DisplaySlices(y_true[0,0,:,:,:], int(1))

        print('Volume shape', i, ':', x.shape)
        print('Mask shape', i, ':', y_true.shape)


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
