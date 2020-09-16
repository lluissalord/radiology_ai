from glob import glob
import os
import shutil
import random

from tqdm import tqdm_notebook

import pandas as pd

import pydicom


def check_DICOM(dcm, debug=False):
    """ Check the DICOM file if it is has the feature required """

    if dcm.BodyPartExamined != 'LOWER LIMB':
        if debug:
            print('BodyPartExamined: ', dcm.BodyPartExamined, 'on Accession Number:', dcm.AccessionNumber)
        return False
    if dcm.SeriesDescription != 'RODILLA AP':
        if debug:
            print('SeriesDescription: ', dcm.SeriesDescription, 'on Accession Number:', dcm.AccessionNumber)
        return False
    return True


def move_file(src_filepath, filename, dst_folder, force_extension=None, copy=True):
    """ Copy file to the destination folder with folder name """

    # Get extension of source file
    _, extension = os.path.splitext(src_filepath)
    if force_extension is not None:
        extension = force_extension

    # Define the destination path
    dst_filepath = os.path.join(dst_folder, filename + extension)

    # Create the destination folder if not exists
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    if copy:
        # Copy file with folder name
        shutil.copyfile(src_filepath, dst_filepath)
    else:
        # Move file with folder name
        shutil.move(src_filepath, dst_filepath)


def organize_folders(src_folder, dst_folder, groups=None, subgroup_length=None, force_extension=None, copy=True, debug=False):
    """ Organize folders and files to set all the desired DICOM files into the correct folder """

    # Clean destination folder
    if os.path.isdir(dst_folder):
        shutil.rmtree(dst_folder)

    # Look at all the DICOM files in the source folder, check them and move them appropiatly
    folders = glob(os.path.join(src_folder, '*'))
    correct_filepaths = []
    correct_folders = []
    for folder in tqdm_notebook(folders, desc='Check folders: '):
        filepaths = glob(os.path.join(folder, '*'))
        for filepath in filepaths:
            dcm = pydicom.dcmread(filepath)
            if check_DICOM(dcm, debug):
                correct_filepaths.append(filepath)
                correct_folders.append(folder)

    folders_dst_folders = get_final_dst_folder(dst_folder, correct_folders, groups, subgroup_length)
    for filepath in tqdm_notebook(correct_filepaths, desc='Move files'):
        path, _ = os.path.split(filepath)
        final_dst_folder = folders_dst_folders[path]

        # Rename the file with the name of the folder (patient ID)
        _, filename = os.path.split(path)

        move_file(filepath, filename, final_dst_folder, force_extension=force_extension, copy=copy)


def expand_list(l, n):
    """ Expand a list `l` to repeat its elements till reaching length of `n` """
    return (l*(n // len(l) + 1))[:n]


def shuffle_group_folders(folders, groups, subgroup_length=None):
    """ Generate dictionaries of the shuffled groups and subgroups related to folders """
    if subgroup_length is not None:
        subgroups = list(range(len(folders) // subgroup_length + 1))[:len(folders)]
        expanded_subgroups = expand_list(subgroups, len(folders))
        random.shuffle(folders)
        folders_subgroup = dict(zip(folders, expanded_subgroups))

        random.shuffle(groups)
        expanded_groups = expand_list(groups, len(subgroups))
        subgroup_group = dict(zip(subgroups, expanded_groups))
        
        return folders_subgroup, subgroup_group
    else:
        random.shuffle(groups)
        random.shuffle(folders)
        expanded_groups = expand_list(groups, len(folders))
        folders_group = dict(zip(folders, expanded_groups))

        return folders_group


def get_final_dst_folder(dst_folder, folders, groups, subgroup_length):
    """ Relates source folders with destination folder paths depending on shuffle groups and subgroup length """
    
    folders_dst_folders = {}
    if groups is not None and subgroup_length is not None:
        folders_subgroup, subgroup_group = shuffle_group_folders(folders, groups, subgroup_length)
        for folder in folders:
            current_subgroup = folders_subgroup[folder]
            current_group = subgroup_group[current_subgroup]
            relative_folder = os.path.join(current_group, str(current_subgroup))
            folders_dst_folders[folder] = os.path.join(dst_folder, relative_folder)
    elif groups is None and subgroup_length is None:
        for folder in folders:
            folders_dst_folders[folder] = dst_folder
    else:
        if groups is None:
            n_groups = len(folders) // subgroup_length + (len(folders) % subgroup_length != 0)
            groups = [str(i) for i in range(n_groups)]
            subgroup_length = None
        folders_group = shuffle_group_folders(folders, groups, subgroup_length)
        for folder in folders:
            current_group = folders_group[folder]
            relative_folder = current_group
            folders_dst_folders[folder] = os.path.join(dst_folder, relative_folder)
    return folders_dst_folders


def generate_template(dst_folder, groups, subgroup_length, excel=True, csv_sep=';'):
    """ Generates template files for each group of DICOM files using the filename as ID """
    
    # Define extension
    if excel:
        extension = '.xls'
    else:
        extension = '.csv'

    # Extract the length of the path from the destination folder
    dst_folder_length = len(os.path.normpath(dst_folder).split(os.sep))

    # Take into account if there are or not subgroups
    prefix = ''
    if subgroup_length is not None:
        prefix += '*/'
        if groups is None:
            subgroup_length = None
        
    if groups is not None:
        prefix += '*/'
     
    # Look for Excel/CSV files in the folders and remove them
    template_files = glob(
        os.path.join(
            dst_folder,
            prefix + '*.xls'
        )
    ) + glob(
        os.path.join(
            dst_folder,
            prefix + '*.csv'
        )
    ) 
    if len(template_files) > 0:
        # Remove template files
        for template_file in template_files:
            os.remove(template_file)

    # Get all the folder and subfolders that contain DICOM files
    folderpaths = glob(
        os.path.join(
            dst_folder,
            prefix
        )
    )

    # Loop on across the folders to generate de template file
    for folderpath in tqdm_notebook(folderpaths, desc='Folders: '):
        # Get all the DICOM files
        filepaths = glob(
            os.path.join(
                folderpath,
                '*.dcm'
            )
        )

        data = {}
        # Extract file IDs from each filepath
        data['ID'] = list(
            map(
                lambda x: os.path.splitext(
                    os.path.normpath(x) \
                        .split(os.sep)[-1]
                )[0],
                filepaths
            )
        )

        # Create DataFrame from the data with the proposed structure
        df = pd.DataFrame(data, columns=['ID', 'Target', 'Confidence', 'Incorrect_image', 'Not_enough_quality'])

        # Split the path on all the folders
        path_split = os.path.normpath(filepaths[0]).split(os.sep)

        if subgroup_length is not None or groups is not None:
            # Set template filename as the name of the folder just after the destination folder
            template_name = path_split[dst_folder_length]
        else:
            # Set template filename as 'labels'
            template_name = 'labels'

        # If there are subgroups then it is added also the subfolder on the filename
        if subgroup_length is not None:
            template_name = template_name + '_' + path_split[-2]

        # Transform to Excel/CSV on the corresponding folder
        if excel:
            df.to_excel(
                os.sep.join(path_split[:-1] + [template_name + extension]),
                index=False,
            )
        else:
            df.to_csv(
                os.sep.join(path_split[:-1] + [template_name + extension]),
                index=False,
                sep=csv_sep,
            )


def concat_templates(src_folder, excel=True, csv_sep=';'):

    # Define extension
    if excel:
        extension = '.xls'
    else:
        extension = '.csv'

    label_paths = glob(
        os.path.join(
            src_folder,
            '*' + extension
        )
    ) + glob(
        os.path.join(
            src_folder,
            '*/*' + extension
        )
    ) + glob(
        os.path.join(
            src_folder,
            '*/*/*' + extension
        )
    )

    dtype = {'ID':'string','Target':'string'}

    df = pd.DataFrame()
    for label_path in tqdm_notebook(label_paths, desc='Label files: '):
        df = pd.concat([
            df,
            pd.read_excel(label_path, dtype=dtype) if excel else pd.read_csv(label_path, sep=csv_sep, dtype=dtype)
        ])

    df = df.reset_index(drop=True)
    return df


def rename_patient(dicom_files):
    """ Modify metadata regarding Patient's Name and Patient's ID to set them as the filename """
    
    dcms = dicom_files.map(pydicom.dcmread)
    for filepath,dcm in tqdm_notebook(zip(dicom_files,dcms), desc='Files: '):
        _, filename = os.path.split(filepath)
        filename, _ = os.path.splitext(filename)
        dcm.PatientName = filename
        dcm.PatientID = filename
        with open(filepath, 'wb') as f:
            dcm.save_as(f)
