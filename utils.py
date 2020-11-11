from glob import glob
import os
import shutil
import random
from pathlib import Path

from tqdm import tqdm_notebook
from tqdm.auto import tqdm
tqdm.pandas()

import numpy as np
import pandas as pd

import pydicom


def filter_fnames(
    fnames,
    metadata_raw_path,
    check_DICOM_dict={
        'Modality': ['CR', 'DR', 'DX'],
    }
):
    """ Filter all the filenames which are fulfill the metadata conditions passed on `check_DICOM_dict` """

    # Load metadata 
    metadata_df = pd.read_csv(metadata_raw_path)

    # Filter by the ones taht fulfill the conditions
    metadata_df = df_check_DICOM(metadata_df, check_DICOM_dict)

    # Create a DataFrame to compare filenames
    check_df = pd.DataFrame(metadata_df.fname.str.split('/').str[-1], index=metadata_df.fname.str.split('/').str[-1])

    # Loop over all the filenames
    filter_fnames = []
    for fname in tqdm(fnames):
        try:
            filename, ext = os.path.splitext(fname.name)
        except AttributeError:
            fname = Path(fname)
            filename, ext = os.path.splitext(fname.name)

        # Take into account the ones which seems to have extension ".PACSXXX" is not an extension
        if ext.startswith('.PACS'):
            filename = fname.name

        # Check if is in the list and add it
        try:
            check_df.loc[filename, :]
            filter_fnames.append(str(fname).replace(os.sep, '/'))
        except KeyError:
            continue
    
    return filter_fnames


def default_check_DICOM_dict():
    """ Get default values for check DICOM dictionary """

    check_DICOM_dict = {
        'SeriesDescription': ['RODILLA AP', 'RODILLAS AP'],
        'BodyPartExamined': ['LOWER LIMB', 'KNEE']
    }

    return check_DICOM_dict


def df_check_DICOM(df, check_DICOM_dict):
    """ Filter DataFrame on the rows that match the filter specified on `check_DICOM_dict` """
    match = True
    for key, value in check_DICOM_dict.items():
        match = (match) & df[key].isin(value)
    
    return df[match]


def sample_df_check_DICOM(df, check_DICOM_dict, max_samples_per_case=5):
    """ Get a random sample of the filtered DataFrame determined by `check_DICOM_dict` appearing all the cases there """

    df_match = df_check_DICOM(df, check_DICOM_dict)
    
    all_keys = list(check_DICOM_dict.keys())

    return get_each_case_samples(df_match, all_keys, max_samples_per_case)


def get_each_case_samples(df, all_keys, max_samples_per_case=5):
    """ Extract all the different cases on `all_keys` and creating a DataFrame with the number set for all the cases """
    cases_df = df[all_keys].drop_duplicates()
    
    concat_list = []
    for i in range(len(cases_df.index)):
        concat_list.append(
            df[
                (
                    df[all_keys] == cases_df.iloc[i][all_keys]
                ).all(axis=1)
            ].sample(max_samples_per_case, replace=True)
        )

    return pd.concat(concat_list)


def check_DICOM(dcm, check_DICOM_dict=None, debug=False):
    """ Check the DICOM file if it is has the feature required """

    if check_DICOM_dict is None:
        check_DICOM_dict = default_check_DICOM_dict()

    check = True
    for key, value in check_DICOM_dict.items():
        if dcm.get(key) not in value:
            check = False
            if debug:
                print(f'{key}: {dcm.get(key)} on Accession Number: {dcm.AccessionNumber}')
            break
    return check


def check_metadata_label(raw_path, metadata_labels, label='ap'):
    """ Check if the path matches with the desired label on metadata_labels """

    if 'Final_pred' in metadata_labels.columns:
        pred_col = 'Final_pred'
    else:
        pred_col = 'Pred'

    try:
        return metadata_labels.loc[raw_path, pred_col] == label
    except KeyError as e:
        # Probably these are cases which have wrong metadata out of 'ap', 'lat', 'two'
        # However, we try if it can be a case of wrong path sep and only take the filename
        filename = os.path.splitext(
            os.path.split(raw_path)[-1]
        )[0]
        row_match = metadata_labels.index.str.endswith(
            filename
        )
        return (metadata_labels.loc[row_match, pred_col] == label).any()


def move_file(src_filepath, filename, dst_folder, force_extension=None, copy=True, return_filepath=True):
    """ Move or copy file to the destination folder with folder name """

    # Define extension
    _, src_extension = os.path.splitext(src_filepath)
    dst_filename, dst_extension = os.path.splitext(filename)
    if force_extension != False and force_extension is not None:
        extension = force_extension
    elif dst_extension != '' or force_extension == False:
        extension = dst_extension
    else:
        extension = src_extension

    # Define filename
    filename = dst_filename + extension

    # Define the destination path
    dst_filepath = os.path.join(dst_folder, filename)

    # Create the destination folder if not exists
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    if copy:
        # Copy file with folder name
        shutil.copyfile(src_filepath, dst_filepath)
    else:
        # Move file with folder name
        shutil.move(src_filepath, dst_filepath)

    if return_filepath:
        return dst_filepath


def move_relation(relation_filepath, copy=True, to_raw=True):
    """ Move or copy files to/from raw destination from/to final destination based on relation file """

    # Open relation file where the move/copy will be based on
    relation_df = open_name_relation_file(relation_filepath, sep=',')

    # Check source files
    check_relation(relation_df, check_path=to_raw, check_raw=not to_raw)

    if to_raw:
        # Loop over all the files to move/copy them to the raw destination
        relation_df.progress_apply(
            lambda x: move_file(
                src_filepath=os.path.join(x.Path, x.Filename + '.dcm'),
                filename=x.Original_Filename,
                dst_folder=os.path.split(x.name)[0],
                force_extension=False,
                copy=copy,
                return_filepath=False
            ),
            axis=1
        )
    else:
        # Loop over all the files to move/copy them to the final destination
        relation_df.progress_apply(
            lambda x: move_file(
                src_filepath=x.name,
                filename=x.Filename,
                dst_folder=x.Path,
                force_extension='.dcm',
                copy=copy,
                return_filepath=False
            ),
            axis=1
        )

    # Check destination files
    check_relation(relation_df, check_path=not to_raw, check_raw=to_raw)


def move_blocks(parent_folder, new_folder, blocks, relation_filepath, template_extension='xlsx'):
    """ Move blocks of files to the new folder (person name) """

    # Create new folder if it does not exist
    new_folder_path = os.path.join(parent_folder, new_folder)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    relation_df = open_name_relation_file(relation_filepath, sep=',')

    if type(blocks) is not list:
        blocks = [blocks]

    for block in blocks:
        block = str(block)

        # Search for the block folder path
        block_path = glob(
            os.path.join(parent_folder, '*', block)
        )[0]

        # Move entire block folder
        dst_folder = os.path.join(new_folder_path, block)
        shutil.move(block_path, dst_folder)

        # Search for the template of this folder and rename it appropiately
        template_path = glob(
            os.path.join(dst_folder, f'*.{template_extension}')
        )[0]
        new_template_filename = f'{new_folder}_{block}.{template_extension}'
        os.rename(
            template_path,
            os.path.join(dst_folder, new_template_filename)
        )

        # Update the relation file with the new folder path
        relation_df = update_block_relation(relation_df, block, new_folder, sep='/')

    # Save relation file
    save_name_relation_file(relation_df, relation_filepath, sep=',')


def organize_folders(src_folder, dst_folder, relation_filepath, reset=False, groups=None, subgroup_length=None, filename_prefix='IMG_', force_extension=None, copy=True, metadata_labels=None, check_DICOM_dict=None, debug=False):
    """ Organize folders and files to set all the desired DICOM files into the correct folder """

    # In case not reseting the folders, then the current relation is required
    if not reset:
        relation_df = open_name_relation_file(relation_filepath, sep=',')
        if len(relation_df.index) == 0:
            raise ValueError('No reset is set, but there is no relation file or it is empty')

    # Look at all the DICOM files in the source folder, check them and move them appropiatly
    folders = glob(os.path.join(src_folder, '*'))
    correct_filepaths = []
    for folder in tqdm_notebook(folders, desc='Check folders: '):
        # Find all files in the folder
        filepaths = glob(os.path.join(folder, '*'))

        # Open all the files as DICOM and check if they fullfil the condition to be used in the study
        for filepath in filepaths:

            # Normalize path to be equal without depending on the OS
            filepath = os.path.normpath(filepath).replace(os.sep, '/')

            # If is no resetting and the file is already on the relation, then there is no need to check
            if not reset:
                src_path = os.path.split(filepath)[0]
                if src_path in relation_df.index:
                    continue
            
            # Check if current file is frontal (ap) image
            if metadata_labels is not None:
                if check_metadata_label(filepath, metadata_labels, label='ap'):
                    correct_filepaths.append(filepath)
            else:
                # Read and check DICOM
                dcm = pydicom.dcmread(filepath)
                if check_DICOM(dcm, check_DICOM_dict, debug):
                    correct_filepaths.append(filepath)

    # Only proceed if there is files to move or resetting folders
    if len(correct_filepaths) > 0 or reset:

        # Relates source filepaths with destination folder paths depending on shuffle groups and subgroup length
        folders_dst = get_final_dst(dst_folder, correct_filepaths, groups, subgroup_length)
        temp_relation_df = pd.DataFrame(folders_dst.values(), index=folders_dst.keys(), columns=['Path'])
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

        # Get last ID of the current files
        current_id = get_last_id(relation_df, prefix=filename_prefix)

        # Loop over the files that should be copied/moved
        for filepath in tqdm_notebook(correct_filepaths, desc='Move files'):

            # Get the final destination folder
            _, src_filename = os.path.split(filepath)
            final_dst = relation_df.loc[filepath, 'Path']

            # Set filename depending on numeration
            filename = filename_prefix + str(current_id + 1)
            current_id += 1

            # Add new relation on the DataFrame
            relation_df = add_new_relation(relation_df, filepath, src_filename, filename)

            # Check raw relation before moving
            check_relation(relation_df.loc[filepath], check_path=False, check_raw=True)

            # Copy/Move the file to the final destination with
            move_file(filepath, filename, final_dst, force_extension=force_extension, copy=copy)

            # Check new path relation before saving
            check_relation(relation_df.loc[filepath], check_path=True, check_raw=False)

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


def get_final_dst(dst_folder, filepaths, groups, subgroup_length):
    """ Relates source filepaths with destination paths depending on shuffle groups and subgroup length """
    
    folders_dst = {}
    if groups is not None and subgroup_length is not None:
        folders_subgroup, subgroup_group = shuffle_group_folders(filepaths, groups, subgroup_length)
        for filepath in filepaths:
            current_subgroup = folders_subgroup[filepath]
            current_group = subgroup_group[current_subgroup]
            relative_folder = os.path.join(current_group, str(current_subgroup))
            folders_dst[filepath] = os.path.join(dst_folder, relative_folder)
    elif groups is None and subgroup_length is None:
        for filepath in filepaths:
            folders_dst[filepath] = dst_folder
    else:
        if groups is None:
            n_groups = len(filepaths) // subgroup_length + (len(filepaths) % subgroup_length != 0)
            groups = [str(i) for i in range(n_groups)]
            subgroup_length = None
        folders_group = shuffle_group_folders(filepaths, groups, subgroup_length)
        for filepath in filepaths:
            current_group = folders_group[filepath]
            relative_folder = current_group
            folders_dst[filepath] = os.path.join(dst_folder, relative_folder)
    return folders_dst


def generate_template(dst_folder, groups, subgroup_length, filename_prefix='IMG_', excel=True, csv_sep=';'):
    """ Generates template files for each group of DICOM files using the filename as ID """
    
    # Define extension
    if excel:
        extension = '.xlsx'
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
        df = pd.DataFrame(data, columns=['ID', 'Target', 'Side', 'Difficulty', 'Incorrect_image', 'Not_enough_quality'])

        # Look for Excel/CSV files in the folders
        template_files = glob(
            os.path.join(
                folderpath + '*.xls'
            )
        ) + glob(
            os.path.join(
                folderpath + '*.xlsx'
            )
        ) + glob(
            os.path.join(
                folderpath + '*.csv'
            )
        )

        # If there are template then we have to check if there is data and if it has to be overwritten or not
        template_file = None
        if len(template_files) == 1:
            template_file = template_files[0]

            # Check if there is data on the template file
            old_df, check_data = check_data_in_template(template_file)

        elif len(template_files) > 1:
            raise ValueError(f'There are more than one template on the following folder, please remove the out-dated one:\n{folderpath}')

        # In case that there is no template files then we are sure that there is no data
        else:
            check_data = False

        # Sort DataFrame by ID if possible
        df = sort_template_file(df, filename_prefix)

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

        # If there is data then we need to request permission to the user to overwrite
        allow_overwrite = True
        check_same_ids = True
        if check_data:
            
            # Check if old DataFrame match with current IDs
            old_df = sort_template_file(old_df, filename_prefix)
            check_same_ids = (df['ID'] == old_df['ID']).all()
            if check_same_ids:
                # Ask user if want to overwrite file with data
                allow_input = input(f'There is data on file the following file:\n{template_file}\n\nDo you want to overwrite it?[y/n] (y default): ')
                if len(allow_input) != 0 and allow_input[0].lower() == 'n':
                    allow_overwrite = False
            else:
                # Rename file with prefix 'old_'
                src_folder, src_filename = os.path.split(template_file)
                new_filename = 'old_' + src_filename
                move_file(template_file, new_filename, src_folder, copy=False)

                print(f'WARNING: The following file is out-dated with different IDs than the ones in the folder:\n{template_file}\n\nHowever, there is data in it, then it is being renamed from {src_filename} to {new_filename}. Please updated the file the new file and remove the old one before continuing')

        # If there is no data or the user allow to remove the file
        if not check_data or (template_file is not None and allow_overwrite):
            # Only remove if file still exists
            if template_file is not None and os.path.exists(template_file):
                os.remove(template_file)

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


def sort_template_file(df, filename_prefix):
    """ Sort DataFrame based on template filename """

    try:
        # Extract ID from the filename and sort by it as numerical sorting
        df['sort'] = df['ID'].str[len(filename_prefix):].astype(int)
        df = df.sort_values('sort')
        df = df.drop('sort', axis=1)
    except ValueError as e:
        print('Not able to sort template by ID because: ', e)
        df = df.sort_values('ID')

    return df.reset_index(drop=True)


def check_data_in_template(template_file, sep=None):
    """ Check if there is data different than null in each template """

    dtype = {'ID':'string','Target':'string'}

    # Extract extension and define loading method depending on it
    _, extension = os.path.splitext(template_file)
    if extension.startswith('.xls'):
        df = pd.read_excel(template_file, dtype=dtype)

    elif extension == '.csv':
        if sep is not None:
            df = pd.read_csv(template_file, sep=sep, dtype=dtype)
        else:
            df = pd.read_csv(template_file, sep=',', dtype=dtype)

        # Check if CSV has been loaded with the correct sep
        if len(df.columns) == 1:
            df = pd.read_csv(template_file, sep=';', dtype=dtype)
            if len(df.columns) == 1:
                raise ValueError('Please define the correct sep for the CSV files already existing as for examples: ', template_file)

    return df, df.drop('ID', axis=1).notnull().any().any()


def concat_templates(src_folder, excel=True, csv_sep=';'):
    """ Concatenate all the template into a DataFrame """

    # Define extension
    if excel:
        extension = '.xls*'
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
        # Check if there is any file which is out-dated
        _, label_filename = os.path.split(label_path)
        if label_filename.startswith('old_'):
            raise ValueError(f'The following file is out-dated, please move the data in this file to the corresponding file and remove the out-dated file.\n\n{label_path}')

        df = pd.concat([
            df,
            pd.read_excel(label_path, dtype=dtype) if excel else pd.read_csv(label_path, sep=csv_sep, dtype=dtype)
        ])

    df = df.reset_index(drop=True)
    return df


def rename_patient(dicom_files):
    """ Modify metadata regarding Patient's Name and Patient's ID to set them as the filename """
    
    dcms = dicom_files.map(pydicom.dcmread)
    for filepath, dcm in tqdm_notebook(zip(dicom_files, dcms), desc='Files: ', total=len(dicom_files)):
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
    """ Add a new relation on the relation DataFrame """

    # Check it does not exist a conflictive addition
    # if src_path in relation_df.index and 'Original_Filename' in relation_df.columns and not np.isnan(relation_df.loc[src_path, 'Original_Filename']):
    if src_path in relation_df.index and 'Original_Filename' in relation_df.columns and relation_df['Original_Filename'].notnull()[src_path]:
        if relation_df.loc[src_path, 'Filename'] != new_filename:
            raise ValueError(f'For file "{src_path}"" there is already a relation with "{relation_df.loc[src_path, "Filename"]}" but it is being added for "{new_filename}"')
    else:
        relation_df.loc[src_path, 'Filename'] = new_filename
        relation_df.loc[src_path, 'Original_Filename'] = src_filename

    return relation_df


def update_block_relation(relation_df, block, new_folder, sep='/'):
    """ Replace the old folder names by the new folder only to the paths where the block appears """

    relation_df.loc[
        relation_df['Path'].str.endswith(block),
        'Path'
    ] = relation_df.loc[
        relation_df['Path'].str.endswith(block),
        'Path'
    ].str.replace(
        f'((?<={sep})(.*)(?={sep}{block}$))',
        new_folder,
    )

    check_relation(relation_df, check_path=True, check_raw=False)

    return relation_df


def check_relation(relation_df, check_path=True, check_raw=True):
    """ Check that the relation on the relation DataFrame is preserved """

    # Check current path
    if check_path:
        if type(relation_df) is pd.DataFrame:
            check = relation_df['Path'].apply(lambda x: os.path.exists(x))
        else:
            check = pd.Series(os.path.exists(relation_df['Path']))

        # Raise error if not true for all the cases
        if not check.all():
            raise ValueError(f'The following cases do not have correct `Path` on relation DataFrame:\n{relation_df.loc[~check]}')

    # Check raw path
    if check_raw:
        if type(relation_df) is pd.DataFrame:
            check = pd.Series(relation_df.index.map(lambda x: os.path.exists(x)), index=relation_df.index, dtype=bool)
        else:
            check = pd.Series(os.path.exists(relation_df.name))

        # Raise error if not true for all the cases
        if not check.all():
            raise ValueError(f'The following cases do not have correct `Path` on relation DataFrame:\n{relation_df.loc[~check]}')

    return True