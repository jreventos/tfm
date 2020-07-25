# Read vtk

VtkPath = "/Users/jreventos/Desktop/TFM/ENDO_LA_ascii_2.vtk"

# VTK reader

from vtk import vtkStructuredPointsReader, vtkPolyDataReader, vtkPolyData, vtkStructuredPointsWriter
from vtk.util.numpy_support import vtk_to_numpy
from utils.DataVisualization import*
from utils.DataConversion import*
from tfm.transforms import*


#data= VtkReader(VtkPath)


#PathDicom = "/Users/jreventos/Desktop/TFM/Data/DICOM/0000017102"
#ArrayDicom = DicomReader(PathDicom)


import pandas as pd

def CheckArrays(Array1,Array2):
    # Check if the lists are equal

    result = (Array1==Array2).all()
    if result:
        print('Lists are exactly equal')
    else:
        print('Lists are not equal')



#DisplaySlices(data,ColourMax = 1)


PathDicom = "/Users/jreventos/Desktop/TFM/tfm/Data/DICOM/0000017102"
arraydicom= DicomReader(PathDicom)
print(arraydicom)

DisplaySlices(arraydicom,ColourMax = 200)


