
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from tfm.utils.readers import VtkReader

import numpy as np
import glob
import os
import re


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
                mask_slices = [VtkReader(img_path) for img_path in mask_dirs]
            else:
                image_dirs = [img_path for img_path in
                              sorted(glob.glob(name_set + '/*.vtk'), key=lambda x: float(re.findall("(\d+)", x)[0]))]

                patient_ids = [img_dir.split('/')[-1] for img_dir in image_dirs]
                image_slices = [VtkReader(img_path) for img_path in image_dirs]

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

if __name__ == "__main__":

    root_dir = './patients_data'
    val_dataset = LeftAtriumSegmentationDataset(root_dir,transform=None,image_size=320,mode='val',
                                                random_sampling=False,seed=42,dimension="3D")

    print(val_dataset)
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(len(loader))
    for i, data in enumerate(loader):
        x, y_true = data
        print(i, x.shape)
        print(i, y_true.shape)


transform = transforms.Compose([
        #utils.Resize(opt.image_size),
        transforms.ToTensor(),
        #utils.Normalize(mean=opt.mean, std=opt.std)
        ])


