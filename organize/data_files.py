""" Functions required for organizing files for proper annotation """

from glob import glob
import os
import shutil
import random
from pathlib import Path
from collections.abc import Iterable

from tqdm.auto import tqdm

tqdm.pandas()

import numpy as np
import pandas as pd

import pydicom

from organize.dicom import *
from organize.relation import *
from organize.utils import *


def move_file(
    src_filepath,
    filename,
    dst_folder,
    force_extension=None,
    copy=True,
    return_filepath=True,
):
    """Move or copy file to the destination folder with folder name"""

    # Define extension
    _, src_extension = os.path.splitext(src_filepath)
    dst_filename, dst_extension = os.path.splitext(filename)
    if force_extension != False and force_extension is not None:
        extension = force_extension
    elif dst_extension != "" or force_extension == False:
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
    """Move or copy files to/from raw destination from/to final destination based on relation file"""

    # Open relation file where the move/copy will be based on
    relation_df = open_name_relation_file(relation_filepath, sep=",")

    # Check source files
    check_relation(relation_df, check_path=to_raw, check_raw=not to_raw)

    if to_raw:
        # Loop over all the files to move/copy them to the raw destination
        relation_df.progress_apply(
            lambda x: move_file(
                src_filepath=os.path.join(x.Path, x.Filename + ".dcm"),
                filename=x.Original_Filename,
                dst_folder=os.path.split(x.name)[0],
                force_extension=False,
                copy=copy,
                return_filepath=False,
            ),
            axis=1,
        )
    else:
        # Loop over all the files to move/copy them to the final destination
        relation_df.progress_apply(
            lambda x: move_file(
                src_filepath=x.name,
                filename=x.Filename,
                dst_folder=x.Path,
                force_extension=".dcm",
                copy=copy,
                return_filepath=False,
            ),
            axis=1,
        )

    # Check destination files
    check_relation(relation_df, check_path=not to_raw, check_raw=to_raw)


def move_distribute_blocks(
    parent_folder, new_folders, blocks, relation_filepath, template_extension="xlsx"
):
    """Move and distribute equal number of blocks of files to a list of new folders (person names)"""

    distribution = np.random.permutation(
        np.tile(
            np.random.permutation(new_folders),
            len(blocks) // len(new_folders) + (len(blocks) % len(new_folders) != 0),
        )[: len(blocks)]
    )

    for folder in tqdm(np.unique(new_folders), desc="Distributions"):
        current_blocks = np.array(blocks)[np.where(distribution == folder)]
        move_blocks(
            parent_folder=parent_folder,
            new_folder=folder,
            blocks=current_blocks,
            relation_filepath=relation_filepath,
            template_extension=template_extension,
        )


def move_blocks(
    parent_folder, new_folder, blocks, relation_filepath, template_extension="xlsx"
):
    """Move blocks of files to the new folder (person name)"""

    # Create new folder if it does not exist
    new_folder_path = os.path.join(parent_folder, new_folder)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    relation_df = open_name_relation_file(relation_filepath, sep=",")

    check_relation(relation_df, check_path=True, check_raw=False)
    print("All in place!")

    if not isinstance(blocks, Iterable):
        blocks = [blocks]

    for block in tqdm(blocks, desc="Blocks"):
        block = str(block)

        # Search for the block folder path
        block_path = glob(os.path.join(parent_folder, "*", block))[0]

        # Move entire block folder
        dst_folder = os.path.join(new_folder_path, block)
        shutil.move(block_path, dst_folder)

        # Search for the template of this folder and rename it appropiately
        template_path = glob(os.path.join(dst_folder, f"*.{template_extension}"))[0]
        new_template_filename = f"{new_folder}_{block}.{template_extension}"
        os.rename(template_path, os.path.join(dst_folder, new_template_filename))

        # Update the relation file with the new folder path
        relation_df = update_block_relation(
            relation_df, parent_folder, block, new_folder, sep="/"
        )

    # Save relation file
    save_name_relation_file(relation_df, relation_filepath, sep=",")


def organize_folders(
    src_folder,
    dst_folder,
    relation_filepath,
    reset=False,
    groups=None,
    subgroup_length=None,
    filename_prefix="IMG_",
    force_extension=None,
    copy=True,
    metadata_labels=None,
    label_exact_match=True,
    check_DICOM_dict=None,
    debug=False,
):
    """Organize folders and files to set all the desired DICOM files into the correct folder"""

    # In case not reseting the folders, then the current relation is required
    if not reset:
        relation_df = open_name_relation_file(relation_filepath, sep=",")
        if len(relation_df.index) == 0:
            raise ValueError(
                "No reset is set, but there is no relation file or it is empty"
            )
        if relation_df.index.duplicated("Original").any():
            raise ValueError(
                "There is a duplicated value on relation file, please review it and modify it"
            )

    # Look at all the DICOM files in the source folder, check them and move them appropiatly
    folders = glob(os.path.join(src_folder, "*"))
    correct_filepaths = []
    for folder in tqdm(folders, desc="Check folders: "):
        # Find all files in the folder
        filepaths = glob(os.path.join(folder, "*"))

        # Open all the files as DICOM and check if they fullfil the condition to be used in the study
        for filepath in filepaths:

            # Normalize path to be equal without depending on the OS
            filepath = os.path.normpath(filepath).replace(os.sep, "/")

            # If is no resetting and the file is already on the relation, then there is no need to check
            if not reset:
                if filepath in relation_df.index:
                    continue

            # Check if current file is frontal (ap) image
            if metadata_labels is not None:
                if check_metadata_label(
                    filepath, metadata_labels, label="ap", exact_match=label_exact_match
                ):
                    correct_filepaths.append(filepath)
            else:
                # Read and check DICOM
                dcm = pydicom.dcmread(filepath)
                if check_DICOM(dcm, check_DICOM_dict, debug):
                    correct_filepaths.append(filepath)

    print("Preparing for organizing files...")
    # Only proceed if there is files to move or resetting folders
    if len(correct_filepaths) > 0 or reset:

        # TODO: Make sure that it continues from the last block
        if not reset:
            last_block = relation_df["Path"].str.split("/").str[-1].astype(int).max()
            start_num_subgrups = last_block + 1
        else:
            start_num_subgrups = 0

        # Relates source filepaths with destination folder paths depending on shuffle groups and subgroup length
        folders_dst = get_final_dst(
            dst_folder, correct_filepaths, groups, subgroup_length, start_num_subgrups
        )
        temp_relation_df = pd.DataFrame(
            folders_dst.values(), index=folders_dst.keys(), columns=["Path"]
        )
        temp_relation_df["Filename"] = filename_prefix + str(-1)
        temp_relation_df.index.rename("Original", inplace=True)

        print("Groups and subgroups organized")
        if debug:
            print(temp_relation_df)

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

        print("Current last id is: ", current_id)

        print("Number of correct filepaths: ", len(correct_filepaths))

        if debug:
            print("Current filepaths:\n\n", correct_filepaths)

        # Loop over the files that should be copied/moved
        for filepath in tqdm(correct_filepaths, desc="Move files"):

            # Get the final destination folder
            _, src_filename = os.path.split(filepath)
            final_dst = relation_df.loc[filepath, "Path"]

            # Set filename depending on numeration
            filename = filename_prefix + str(current_id + 1)
            current_id += 1

            # Add new relation on the DataFrame
            relation_df = add_new_relation(
                relation_df, filepath, src_filename, filename
            )

            # Check raw relation before moving
            check_relation(relation_df.loc[filepath], check_path=False, check_raw=True)

            # Copy/Move the file to the final destination with
            move_file(
                filepath,
                filename,
                final_dst,
                force_extension=force_extension,
                copy=copy,
            )

            # Check new path relation before saving
            check_relation(relation_df.loc[filepath], check_path=True, check_raw=False)

            # Save the relation file
            save_name_relation_file(relation_df, relation_filepath, sep=",")

    return relation_df, len(correct_filepaths)


def shuffle_group_folders(
    folders, groups, subgroup_length=None, start_num_subgrups=None
):
    """Generate dictionaries of the shuffled groups and subgroups related to folders"""
    if subgroup_length is not None:
        num_subgroups = len(folders) // subgroup_length + 1
        start_num_subgrups = start_num_subgrups if start_num_subgrups is not None else 0
        subgroups = list(range(start_num_subgrups, start_num_subgrups + num_subgroups))[
            : len(folders)
        ]
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


def get_final_dst(
    dst_folder, filepaths, groups, subgroup_length, start_num_subgrups=None
):
    """Relates source filepaths with destination paths depending on shuffle groups and subgroup length"""

    folders_dst = {}
    if groups is not None and subgroup_length is not None:
        folders_subgroup, subgroup_group = shuffle_group_folders(
            filepaths, groups, subgroup_length, start_num_subgrups
        )
        for filepath in filepaths:
            current_subgroup = folders_subgroup[filepath]
            current_group = subgroup_group[current_subgroup]
            relative_folder = os.path.join(current_group, str(current_subgroup))
            folders_dst[filepath] = os.path.join(dst_folder, relative_folder).replace(
                os.sep, "/"
            )
    elif groups is None and subgroup_length is None:
        for filepath in filepaths:
            folders_dst[filepath] = dst_folder
    else:
        if groups is None:
            n_groups = len(filepaths) // subgroup_length + (
                len(filepaths) % subgroup_length != 0
            )
            groups = [str(i) for i in range(n_groups)]
            subgroup_length = None
        folders_group = shuffle_group_folders(
            filepaths, groups, subgroup_length, start_num_subgrups
        )
        for filepath in filepaths:
            current_group = folders_group[filepath]
            relative_folder = current_group
            folders_dst[filepath] = os.path.join(dst_folder, relative_folder).replace(
                os.sep, "/"
            )
    return folders_dst
