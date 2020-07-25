
import os
import vtk
from vtk import vtkStructuredPointsReader, vtkPolyDataReader, vtkPolyData, vtkStructuredPointsWriter

from vtk.util import numpy_support

def BinaryToAscii(PathBinary):
    """
     Conversion between BINARY and ASCII files in vtk format, it saves the Ascii file into the same path.

    :param PathBinary: path of the files folders to convert
    """

    for id in os.listdir(PathBinary):
        print(id)
        for mask in os.listdir(os.path.join(PathBinary, id)):
            if mask.endswith(".vtk"):
                mask_path = os.path.join(os.path.join(PathBinary, id), mask)
                print(mask_path)

                # Read vtk BINARY mask file
                reader = vtkStructuredPointsReader()
                reader.SetFileName(mask_path)
                reader.ReadAllVectorsOn()
                reader.ReadAllScalarsOn()
                reader.Update()

                # Write vtk ASCII mask file
                #train_path = "/Users/jreventos/Desktop/TFM/Data/Train/labels"

                mask_path_ascii = os.path.join(os.path.join(PathBinary, id), 'ENDO_LA_ascii')
                writer = vtkStructuredPointsWriter()
                writer.SetFileName(mask_path_ascii)
                writer.SetInputConnection(reader.GetOutputPort())
                writer.Write()

def VtkReader(VtkPath):
    """
    Vtk ascii file reader
    :param VtkPath: Vtk ascii file path
    :return: numpy array
    """

    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtkStructuredPointsReader()
    # reader = vtkPolyDataReader()
    reader.SetFileName(VtkPath)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()


    #print(reader.GetHeader())
    # Get data
    data = reader.GetOutput()
    dim = data.GetDimensions()

    #print('DATA:\n', data)

    # print('Dimensions:',dim)

   #print('ARRAY:', data.GetPointData().GetArray(''))
    intensities = data.GetPointData().GetArray(0)
    numpy_array = vtk_to_numpy(intensities)
    #print(len(numpy_array))
    list_numpy = list(numpy_array)
    #print(numpy_array)

    array_reshape = numpy_array.reshape([320, 320, 64], order='F')

    return array_reshape


def DicomReader(PathDicom):
    """
    Reads DICOM files and converts the images information into a Numpy Array

    :param PathDicom: path of the DICOM images
    :return: numpy array
    """

    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(PathDicom)
    reader.Update()

    # Load dimensions using `GetDataExtent`
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1] - _extent[0] + 1, _extent[3] - _extent[2] + 1, _extent[5] - _extent[4] + 1]

    # Load spacing values
    ConstPixelSpacing = reader.GetPixelSpacing()

    # Get the 'vtkImageData' object from the reader
    imageData = reader.GetOutput()
    # Get the 'vtkPointData' object from the 'vtkImageData' object
    pointData = imageData.GetPointData()
    # Ensure that only one array exists within the 'vtkPointData' object
    assert (pointData.GetNumberOfArrays() == 1)
    # Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
    arrayData = pointData.GetArray(0)

    # Convert the `vtkArray` to a NumPy array
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')

    #print(ArrayDicom.shape)
    #print(ArrayDicom)
    return ArrayDicom


#data_path = "/Users/jreventos/Desktop/TFM/Data/Masks"
#DataConversion(data_path)

PathDicom = "/Users/jreventos/Desktop/TFM/Data/DICOM/0000017102"
DicomReader(PathDicom)