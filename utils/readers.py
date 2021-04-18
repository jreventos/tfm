import os
import vtk
from vtk import vtkStructuredPointsReader, vtkPolyDataReader, vtkPolyData, vtkStructuredPointsWriter
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk




def VtkReader(VtkPath):
    """
    Vtk BINARY or ASCII file reader to a numpy array.

    Input:
        :VtkPath (string): Vtk ascii file path

    Return: vtk file in numpy array format
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
    height =dim[0]
    width= dim[1]
    num_slices = dim[-1]

    intensities = data.GetPointData().GetArray(0)
    numpy_array = vtk_to_numpy(intensities)

    array_reshape = numpy_array.reshape([height, width, num_slices], order='F')


    return array_reshape


def DicomReader(PathDicom):
    """
    Reads DICOM files and converts the images information into a numpy array

    Input:
        : PathDicom (string): path of the DICOM images

    Return: numpy array of the DICOM file
    """

    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(PathDicom)
    reader.Update()

    # Load dimensions using `GetDataExtent`
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1] - _extent[0] + 1, _extent[3] - _extent[2] + 1, _extent[5] - _extent[4] + 1]

    # Load spacing values
    ConstPixelSpacing = reader.GetPixelSpacing()

    # Get the vtkImageData object from the reader
    imageData = reader.GetOutput()
    # Get the vtkPointData object from the 'vtkImageData' object
    pointData = imageData.GetPointData()
    # Ensure that only one array exists within the 'vtkPointData' object
    assert (pointData.GetNumberOfArrays() == 1)
    # Get the vtkArray (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
    arrayData = pointData.GetArray(0)

    # Convert the vtkArray to a NumPy array
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')

    #print(ArrayDicom.shape)
    #print(ArrayDicom)

    return ArrayDicom


def generate_vtk_from_numpy(ndarray,filename):
    """
    Write vtk files from numpy 3d volumes,

    Input:
       : ndarray
       : filename (string)
    """
    from tvtk.api import tvtk, write_data

    grid = tvtk.ImageData(spacing=(1.25, 1.25, 2.5), origin=(0, 0, 0),
                          dimensions=ndarray.shape)
    grid.point_data.scalars = ndarray.ravel(order='F')
    grid.point_data.scalars.name = 'scalars'

    # Writes legacy ".vtk" format if filename ends with "vtk", otherwise
    # this will write data using the newer xml-based format.
    write_data(grid, filename)



def BinaryToAscii(PathBinary):
    """
     Conversion of a BINARY vtk format to an ASCII vtk format, it saves the Ascii file into the same path.

    Input:
        :PathBinary (string): path of the files folders to convert
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
                #train_path = "/Users/jreventos/Desktop/TFM/Data/Train/masks"

                mask_path_ascii = os.path.join(os.path.join(PathBinary, id), 'ENDO_LA_ascii')
                writer = vtkStructuredPointsWriter()
                writer.SetFileName(mask_path_ascii)
                writer.SetInputConnection(reader.GetOutputPort())
                writer.Write()