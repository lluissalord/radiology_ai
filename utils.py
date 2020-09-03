from glob import glob
import os
import shutil

import pydicom


def check_DICOM(dcm):
    """ Check the DICOM file if it is has the feature required """

    if dcm.BodyPartExamined != 'LOWER LIMB':
        print('BodyPartExamined: ', dcm.BodyPartExamined)
        return False
    if dcm.SeriesDescription != 'RODILLA AP':
        print('SeriesDescription: ', dcm.SeriesDescription)
        return False
    return True


def move_file(src_filepath, src_folder, dst_folder):
    """ Move and rename file to the destination folder with folder name """
    # Move file to the destination folder 
    _, filename = os.path.split(src_filepath)
    dst_filepath = os.path.join(dst_folder, filename)
    shutil.copyfile(src_filepath, dst_filepath)

    # Rename the file with the name of the folder (patient ID)
    _, foldername = os.path.split(src_folder)
    final_filepath = os.path.join(dst_folder, foldername)
    shutil.move(dst_filepath, final_filepath)


def organize_folders(src_folder, dst_folder):
    """ Organize folders and files to set all the desired DICOM files into the correct folder """

    # Create the destination folder if not exists
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Look at all the DICOM files in the source folder, check them and move them appropiatly
    folders = glob(os.path.join(src_folder, '*'))
    for folder in folders:
        filepaths = glob(os.path.join(folder, '*'))
        for filepath in filepaths:
            dcm = pydicom.dcmread(filepath)
            if check_DICOM(dcm):
                move_file(filepath, folder, dst_folder)
