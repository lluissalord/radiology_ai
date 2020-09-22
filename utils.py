from glob import glob
import os
import shutil
import random

from tqdm import tqdm_notebook

import numpy as np
import pandas as pd

import pydicom


def default_check_DICOM_dict():
    check_DICOM_dict = {
        'SeriesDescription': ['RODILLA AP', 'RODILLAS AP'],
        'BodyPartExamined': ['LOWER LIMB', 'KNEE']
    }

    return check_DICOM_dict


def check_DICOM(dcm, check_DICOM_dict=None, debug=False):
    """ Check the DICOM file if it is has the feature required """

    if check_DICOM_dict is None:
        check_DICOM_dict = default_check_DICOM_dict()

    check = True
    for key, value in check_DICOM_dict:
        if dcm.get(key) not in value:
            check = False
            if debug:
                print(f'{key}: {dcm.get(key)} on Accession Number: {dcm.AccessionNumber}')
            break
    return check


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


def organize_folders(src_folder, dst_folder, relation_filepath, reset=False, groups=None, subgroup_length=None, new_numeration=True, filename_prefix='IMG_', force_extension=None, copy=True, check_DICOM_dict=None, debug=False):
    """ Organize folders and files to set all the desired DICOM files into the correct folder """

    # In case not reseting the folders, then the current relation is required
    if not reset:
        relation_df = open_name_relation_file(relation_filepath, sep=',')
        if len(relation_df.index) == 0:
            raise ValueError('No reset is set, but there is no relation file or it is empty')

    # Look at all the DICOM files in the source folder, check them and move them appropiatly
    folders = glob(os.path.join(src_folder, '*'))
    correct_filepaths = []
    correct_folders = []
    for folder in tqdm_notebook(folders, desc='Check folders: '):
        # Find all files in the folder
        filepaths = glob(os.path.join(folder, '*'))

        # Open all the files as DICOM and check if they fullfil the condition to be used in the study
        for filepath in filepaths:

            # If is no resetting and the file is already on the relation, then there is no need to check
            if not reset:
                src_path = os.path.split(filepath)[0]
                if src_path in relation_df.index:
                    continue
            
            # Read and check DICOM
            dcm = pydicom.dcmread(filepath)
            if check_DICOM(dcm, check_DICOM_dict, debug):
                correct_filepaths.append(filepath)
                correct_folders.append(folder)

    # Only proceed if there is files to move or resetting folders
    if len(correct_folders) > 0 or reset:

        # Relates source folders with destination folder paths depending on shuffle groups and subgroup length
        folders_dst_folders = get_final_dst_folder(dst_folder, correct_folders, groups, subgroup_length)
        temp_relation_df = pd.DataFrame(folders_dst_folders.values(), index=folders_dst_folders.keys(), columns=['Path'])
        temp_relation_df['Filename'] = filename_prefix + str(-1)
        temp_relation_df.index.rename('Original', inplace=True)

        # Define relation DataFrame depending on reset
        if reset:
            # Clean destination folder
            if os.path.isdir(dst_folder):
                shutil.rmtree(dst_folder)

            relation_df = temp_relation_df
        else:
            relation_df = pd.concat([relation_df, temp_relation_df], axis=0)

        # Get last ID of the current files in case of using numeration
        if new_numeration:
            current_id = get_last_id(relation_df, prefix=filename_prefix)

        # Loop over the files that should be copied/moved
        for filepath in tqdm_notebook(correct_filepaths, desc='Move files'):

            # Get the final destination folder
            src_path, src_filename = os.path.split(filepath)
            _, src_folder = os.path.split(src_path)
            final_dst_folder = relation_df.loc[src_path, 'Path']

            # Set filename depending on numeration or patient ID
            if new_numeration:
                filename = filename_prefix + str(current_id + 1)
                current_id += 1
            else:
                # Rename the file with the name of the folder (patient ID)
                filename = src_folder

            # Add new relation on the DataFrame
            relation_df = add_new_relation(relation_df, src_path, src_filename, filename)

            # Copy/Move the file to the final destination with
            move_file(filepath, filename, final_dst_folder, force_extension=force_extension, copy=copy)

            # Save the relation file
            save_name_relation_file(relation_df, relation_filepath, sep=',')

    return relation_df

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


def open_name_relation_file(filepath, sep=','):
    """ Extract DataFrame from file containing the relation between original and new files """

    # Check if file exists
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns = ['Original', 'New_Path', 'Filename', 'Original_Filename'])
    else:
        df = pd.read_csv(filepath, sep=sep, index_col=0)

    return df


def save_name_relation_file(relation_df, filepath, sep=','):
    """ Save file containing the relation between original and new files """
    relation_df.to_csv(filepath, sep=sep, index=True)


def get_last_id(relation_df, prefix='IMG_'):
    """ Get the ID of the last filename from current relation of files """

    # Extract the maximum ID currently set
    new_id = relation_df['Filename'].str.split(prefix).str[1].astype(int).max()

    # In case of nan then set it to -1
    if new_id is np.nan:
        new_id = -1

    return new_id


def add_new_relation(relation_df, src_path, src_filename, new_filename):
    
    # Check it does not exist a conflictive addition
    # if src_path in relation_df.index and 'Original_Filename' in relation_df.columns and not np.isnan(relation_df.loc[src_path, 'Original_Filename']):
    if src_path in relation_df.index and 'Original_Filename' in relation_df.columns and relation_df['Original_Filename'].notnull()[src_path]:
        if relation_df.loc[src_path, 'Filename'] != new_filename:
            raise ValueError(f'For file "{src_path}"" there is already a relation with "{relation_df.loc[src_path, "Filename"]}" but it is being added for "{new_filename}"')
    else:
        relation_df.loc[src_path, 'Filename'] = new_filename
        relation_df.loc[src_path, 'Original_Filename'] = src_filename

    return relation_df
