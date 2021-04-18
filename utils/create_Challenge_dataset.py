import os
from processing.preprocessing import resize_data
import SimpleITK as sitk
import numpy as np

# This are actions and functions that have been developed in order to create the Challenge 18 dataset in our desired format.

def save_nrrd_result(file_path,data,reference_img):
    '''
    save data to a nrrd file
    :param file_path: full path
    :param data: np array
    :param reference_img: refence image
    :return: None
    '''
    image=sitk.GetImageFromArray(data)
    image.SetDirection(reference_img.GetDirection())
    image.SetOrigin(reference_img.GetOrigin())
    image.SetSpacing(reference_img.GetSpacing())
    sitk.WriteImage(image,file_path)


def load_nrrd(full_path_filename,dtype=sitk.sitkUInt8):
    '''
    N*h*W
    :param full_path_filename:
    :return:*H*W
    '''
    if not os.path.exists(full_path_filename):
        raise FileNotFoundError
    image = sitk.ReadImage( full_path_filename )
    image= sitk.Cast( sitk.RescaleIntensity(image),dtype)
    data = sitk.GetArrayFromImage(image) # N*H*W

    return data.transpose(1,2,0),image


# mypath = "C:/Users/Roger/Desktop/JANA/DATASETS/2018 Atrial Segmentation Challenge COMPLETE DATASET/Testing Set"
# patients_image_paths, patients_mask_paths= [], []
# for i in os.listdir(mypath):
#     root = os.path.join(mypath, i)
#
#     image_path = os.path.join(root,"lgemri.nrrd")
#     patients_image_paths.append(image_path)
#
#     mask_path = os.path.join(root, "laendo.nrrd")
#     patients_mask_paths.append(mask_path)
#
#
# print(patients_mask_paths)
# print(patients_image_paths)

def move_LA_challenge_dataset(MRI_list, mask_list, outputfile):
    assert len(MRI_list)==len(mask_list)
    for i in range(len(MRI_list)):

        num = i+1
        image_array,_ = load_nrrd(MRI_list[i])
        image_array_resized = resize_data(image_array, [360, 360, 60])
        np.save('C:/Users/Roger/Desktop/JANA/tfm/LA_segmentation_competition/MRI_volumes/' + outputfile + '/MRI_'+ str(num) , image_array_resized)

        mask_array,_ = load_nrrd(mask_list[i])
        mask_array_resized = resize_data(mask_array, [360, 360, 60])
        np.save('C:/Users/Roger/Desktop/JANA/tfm/LA_segmentation_competition/masks/' + outputfile + '/mask_' + str(num) , mask_array_resized)



