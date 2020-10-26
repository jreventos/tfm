import os

import numpy as np
import vtk
from vtk import (vtkPolyData, vtkPolyDataReader, vtkStructuredPointsReader,
                 vtkStructuredPointsWriter)
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy

import nibabel as nib




def VtkReader(VtkPath):
    """
    Vtk BINARY or ASCII file reader to a numpy array.

    :param VtkPath: Vtk ascii file path
    :return: numpy array
    """

    reader = vtkStructuredPointsReader()
    # reader = vtkPolyDataReader()
    reader.SetFileName(VtkPath)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    # Get data
    data = reader.GetOutput()
    dim = data.GetDimensions()
    num_slices = dim[-1]

    intensities = data.GetPointData().GetArray(0)
    numpy_array = vtk_to_numpy(intensities)

    array_reshape = numpy_array.reshape([320, 320, num_slices], order='F')


    return array_reshape

if __name__ == "__main__":
    old_dir = 'patients_data/masks/train/'
    files = os.listdir(old_dir)
    new_dir = 'CLINIC_DATA/labelsTr/'
    
    for img in files:
        path = old_dir + img
        new_path = new_dir + path.split("/")[-1].split(".vtk")[0] + ".nii.gz"
        data = VtkReader(path)
        new_image = nib.Nifti1Image(data, affine=np.eye(4,4))
        nib.save(new_image,new_path)  
        
