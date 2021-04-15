import os
import vtk
from vtk import vtkStructuredPointsReader, vtkPolyDataReader, vtkPolyData, vtkStructuredPointsWriter
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk





def BinaryToAscii(PathBinary):
    """
     Conversion of a BINARY vtk format to an ASCII vtk format, it saves the Ascii file into the same path.

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
                #train_path = "/Users/jreventos/Desktop/TFM/Data/Train/masks"

                mask_path_ascii = os.path.join(os.path.join(PathBinary, id), 'ENDO_LA_ascii')
                writer = vtkStructuredPointsWriter()
                writer.SetFileName(mask_path_ascii)
                writer.SetInputConnection(reader.GetOutputPort())
                writer.Write()

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
    height =dim[0]
    width= dim[1]
    num_slices = dim[-1]

    intensities = data.GetPointData().GetArray(0)
    numpy_array = vtk_to_numpy(intensities)

    array_reshape = numpy_array.reshape([height, width, num_slices], order='F')


    return array_reshape




# Read VTK FILES: (Another way to read VTK files)
# import SimpleITK as sitk
#
# reader = sitk.ImageFileReader()
# reader.SetImageIO("VTKImageIO")
# #inputImageFileName = '/Users/jreventos/Desktop/TFM/Rtfm/patients_data2/MRI_volumes/train/MRI_2.vtk'
# inputImageFileName = "/Users/jreventos/Desktop/TFM/tfm/patients_data2/masks/train/mask_2.vtk"
# reader.SetFileName(inputImageFileName)
# image = reader.Execute();
# nda = sitk.GetArrayFromImage(image)
# image_vtk = VtkReader(inputImageFileName)
# nda  = nda.transpose(2,1,0)
# print(nda.max())
# print(image_vtk.max())
#
# import collections
# print(collections.Counter(nda.flatten()))
# print(collections.Counter(image_vtk.flatten()))



def DicomReader(PathDicom):
    """
    Reads DICOM files and converts the images information into a numpy array

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
    from tvtk.api import tvtk, write_data

    grid = tvtk.ImageData(spacing=(1.25, 1.25, 2.5), origin=(0, 0, 0),
                          dimensions=ndarray.shape)
    grid.point_data.scalars = ndarray.ravel(order='F')
    grid.point_data.scalars.name = 'scalars'

    # Writes legacy ".vtk" format if filename ends with "vtk", otherwise
    # this will write data using the newer xml-based format.
    write_data(grid, filename)


def vtkImageToNumPy(image, pixelDims):
    pointData = image.GetPointData()
    arrayData = pointData.GetArray(0)
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    ArrayDicom = ArrayDicom.reshape(pixelDims, order='F')

    return ArrayDicom




# # Read the source file.
# reader = vtk.vtkStructuredPointsReader()
# reader.SetFileName("C:/Users/Roger/Desktop/JANA/tfm/patients_data/MRI_volumes/test/MRI_1.vtk")
# reader.Update()  # Needed because of GetScalarRange
# output = reader.GetOutput()
# output_port = reader.GetOutputPort()
# scalar_range = output.GetScalarRange()
# print(scalar_range)
#
# # Create the mapper that corresponds the objects of the vtk file
# # into graphics elements
# mapper = vtk.vtkDataSetMapper()
# mapper.SetInputConnection(output_port)
# mapper.SetScalarRange([0,255])
#
# # Create the Actor
# actor = vtk.vtkActor()
# actor.SetMapper(mapper)
#
# # Create the Renderer
# renderer = vtk.vtkRenderer()
# renderer.AddActor(actor)
# renderer.SetBackground(1, 1, 1) # Set background to white
#
# # Create the RendererWindow
# renderer_window = vtk.vtkRenderWindow()
# renderer_window.AddRenderer(renderer)
#
# # Create the RendererWindowInteractor and display the vtk_file
# interactor = vtk.vtkRenderWindowInteractor()
# interactor.SetRenderWindow(renderer_window)
# interactor.Initialize()
# interactor.Start()