import pandas as pd
import sys
from processing.preprocessing import *
from readers import *
sys.path.append("..")

# This are actions and functions that have been developed in order to create the ClínicLA dataset dataset in our desired format.

def create_csv_names(root_dir_images,root_dir_labels):
    """
    Create a CSV file with all image names and masks
    :param root_dir_images:
    :param root_dir_labels:
    :return:
    """
    img_paths = [img_path for img_path in sorted(glob.glob(root_dir_images + '/*.vtk'))]
    img_paths = sorted(img_paths, key=lambda x: float(re.findall("(\d+)", x)[0])) # sort files according to the digits included in the filename

    labels_paths = [label_path for label_path in sorted(glob.glob(root_dir_labels + '/*.vtk'))]
    labels_paths = sorted(labels_paths, key=lambda x: float(re.findall("(\d+)", x)[0])) # sort files according to the digits included in the filename
    names = []
    if len(img_paths) == len(labels_paths):
        for idx in range(len(img_paths)):
            img_path = img_paths[idx]
            img_name = img_path.split('/')[-1].split('.')[0]

            label_path = labels_paths[idx]
            label_name = label_path.split('/')[-1].split('.')[0]

            names.append([img_path, img_name, label_path,label_name])

        df = pd.DataFrame(names, columns=['IMAGE PATH', 'IMAGE NAME', 'LABEL PATH', 'LABEL NAME'])
        df.to_csv('names.csv', index=False)
    else:
        print('Not the same length')
        pass

#root_dir_labels ='/Users/jreventos/Desktop/TFM/tfm/patients_data2/masks'
#root_dir_images ='/Users/jreventos/Desktop/TFM/tfm/patients_data2/MRI_volumes'
#create_csv_names(root_dir_images,root_dir_labels)
# # Split datasets in training and validation
# df = pd.read_csv('names.csv')
# X_train, X_val, y_train, y_val = train_test_split(df['IMAGE NAME'], df['LABEL NAME'], test_size=0.2, random_state= 42)
#
# train = pd.DataFrame({'IMAGE': X_train, 'LABEL': y_train})
# val = pd.DataFrame({'IMAGE': X_val, 'LABEL': y_val})
#
# train.to_csv('labels_train.csv', index=False, header=True)
# val.to_csv('labels_val.csv', index=False, header=True)
#
# input_train = 'labels_train.csv'
# input_val = 'labels_val.csv'
#
# import os
# import shutil


def move_files(dir,names):

    df = pd.read_csv(names)
    images = df['IMAGE'].values
    labels = df['LABEL'].values

    dir_mri = os.path.join(dir,'MRI_volumes')
    dir_labels = os.path.join(dir,'masks')


    if 'train' in names:
        move_to_images = os.path.join(dir_mri,'train')
        move_to_labels = os.path.join(dir_labels, 'train')


    elif 'val' in names:
        move_to_images = os.path.join(dir_mri, 'val')
        move_to_labels = os.path.join(dir_labels, 'val')
    else:
        print('Error')

    if not os.path.exists(move_to_images):
        os.makedirs(move_to_images)

    if not os.path.exists(move_to_labels):
        os.makedirs(move_to_labels)

    for idx in os.listdir(dir_mri):
        img_name = idx.split('.')[0]
        if img_name in images:
            source = os.path.join(dir_mri, idx)
            shutil.move(source,move_to_images)

    for idx in os.listdir(dir_labels):
        label_name = idx.split('.')[0]
        if label_name in labels:
            source = os.path.join(dir_labels, idx)
            shutil.move(source,move_to_labels)

def move_files_general_electrics(mypath):
    import os
    import glob
    import shutil
    lista = os.listdir(mypath)

    j = 0
    for i in lista[5:]:
        j+=1
        print(i)
        vtk_path = os.path.join(mypath, i)
        os.chdir(vtk_path)
        for ii in glob.glob("*.vtk"):
            print(ii)
            path = os.path.join(vtk_path, ii)
            copy_path = os.path.join('C:/Users/Roger/Desktop/JANA/tfm/patients_data/masks/train',ii)
            print(path)
            print(copy_path)

            shutil.copy(path,copy_path)


def save_resized_numpy(mypath,outfile):
    for i in os.listdir(mypath):
        vtk_path = os.path.join(mypath,i)
        print(vtk_path)
        vtk_original_size = VtkReader(vtk_path)
        vtk_resized = resize_data(vtk_original_size,[360,360,60])
        np.save(outfile+'/'+i[:-4],vtk_resized)
