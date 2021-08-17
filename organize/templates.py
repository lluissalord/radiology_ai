import os
import pandas as pd
from glob import glob
from pathlib import Path
import re
from organize.data_files import move_file

from tqdm.auto import tqdm

tqdm.pandas()


def generate_template(
    dst_folder,
    groups,
    subgroup_length,
    filename_prefix="IMG_",
    excel=True,
    csv_sep=";",
    able_overwrite=False,
):
    """Generates template files for each group of DICOM files using the filename as ID"""

    # Define extension
    if excel:
        extension = ".xlsx"
    else:
        extension = ".csv"

    # Extract the length of the path from the destination folder
    dst_folder_length = len(os.path.normpath(dst_folder).split(os.sep))

    # Take into account if there are or not subgroups
    prefix = ""
    if subgroup_length is not None:
        prefix += "*/"
        if groups is None:
            subgroup_length = None

    if groups is not None:
        prefix += "*/"

    # Get all the folder and subfolders that contain DICOM files
    folderpaths = glob(os.path.join(dst_folder, prefix))

    # Loop on across the folders to generate de template file
    for folderpath in tqdm(folderpaths, desc="Folders: "):
        # Get all the DICOM files
        filepaths = glob(os.path.join(folderpath, "*.dcm"))

        data = {}
        # Extract file IDs from each filepath
        data["ID"] = list(
            map(
                lambda x: os.path.splitext(os.path.normpath(x).split(os.sep)[-1])[0],
                filepaths,
            )
        )

        # Create DataFrame from the data with the proposed structure
        df = pd.DataFrame(
            data,
            columns=[
                "ID",
                "Target",
                "Difficulty",
                "Incorrect_image",
                "Not_enough_quality",
            ],
        )

        # Look for Excel/CSV files in the folders
        template_files = (
            glob(os.path.join(folderpath + "*.xls"))
            + glob(os.path.join(folderpath + "*.xlsx"))
            + glob(os.path.join(folderpath + "*.csv"))
        )

        # If there are template then we have to check if there is data and if it has to be overwritten or not
        template_file = None
        if len(template_files) == 1:
            template_file = template_files[0]

            # Check if there is data on the template file
            old_df, check_data = check_data_in_template(template_file)

        elif len(template_files) > 1:
            raise ValueError(
                f"There are more than one template on the following folder, please remove the out-dated one:\n{folderpath}"
            )

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
            template_name = "labels"

        # If there are subgroups then it is added also the subfolder on the filename
        if subgroup_length is not None:
            template_name = template_name + "_" + path_split[-2]

        # If there is data then we need to request permission to the user to overwrite
        allow_overwrite = False
        check_same_ids = True
        if check_data:

            # Check if old DataFrame match with current IDs
            old_df = sort_template_file(old_df, filename_prefix)
            check_same_ids = (df["ID"] == old_df["ID"]).all()
            if check_same_ids:
                if able_overwrite:
                    # Ask user if want to overwrite file with data
                    allow_input = input(
                        f"There is data on file the following file:\n{template_file}\n\nDo you want to overwrite it?[y/n] (n default): "
                    )
                    if len(allow_input) != 0 and allow_input[0].lower() == "y":
                        allow_overwrite = True
            else:
                # Rename file with prefix 'old_'
                src_folder, src_filename = os.path.split(template_file)
                new_filename = "old_" + src_filename
                move_file(template_file, new_filename, src_folder, copy=False)

                print(
                    f"WARNING: The following file is out-dated with different IDs than the ones in the folder:\n{template_file}\n\nHowever, there is data in it, then it is being renamed from {src_filename} to {new_filename}. Please updated the file the new file and remove the old one before continuing"
                )

        # If there is no data or the user allow to remove the file
        if not check_data or (
            template_file is not None and allow_overwrite and able_overwrite
        ):
            # Only remove if file still exists
            if template_file is not None and os.path.exists(template_file):
                os.remove(template_file)

            # Transform to Excel/CSV on the corresponding folder
            if excel:
                df.to_excel(
                    os.sep.join(path_split[:-1] + [template_name + extension]),
                    index=False,
                    engine="openpyxl"
                )
            else:
                df.to_csv(
                    os.sep.join(path_split[:-1] + [template_name + extension]),
                    index=False,
                    sep=csv_sep,
                )


def modify_template(
    dst_folder, modify_func, groups, subgroup_length, excel=True, csv_sep=";"
):
    """Modify all the templates on dst_folder applying `modify_func` on each DataFrame"""

    # Define extension
    if excel:
        extension = ".xlsx"
    else:
        extension = ".csv"

    # Extract the length of the path from the destination folder
    dst_folder_length = len(os.path.normpath(dst_folder).split(os.sep))

    # Take into account if there are or not subgroups
    prefix = ""
    if subgroup_length is not None:
        prefix += "*/"
        if groups is None:
            subgroup_length = None

    if groups is not None:
        prefix += "*/"

    # Get all the folder and subfolders that contain DICOM files
    folderpaths = glob(os.path.join(dst_folder, prefix))

    # Loop on across the folders to generate de template file
    for folderpath in tqdm(folderpaths, desc="Folders: "):

        # Look for Excel/CSV files in the folders
        template_files = (
            glob(os.path.join(folderpath + "*.xls"))
            + glob(os.path.join(folderpath + "*.xlsx"))
            + glob(os.path.join(folderpath + "*.csv"))
        )

        for template_file in template_files:
            # Read each template
            df = read_template(template_file, sep=csv_sep)

            # Do specified modification
            df = modify_func(df)

            # Transform to Excel/CSV on the corresponding folder
            if excel:
                df.to_excel(
                    template_file,
                    index=False,
                    engine="openpyxl"
                )
            else:
                df.to_csv(
                    template_file,
                    index=False,
                    sep=csv_sep,
                )


def sort_template_file(df, filename_prefix):
    """Sort DataFrame based on template filename"""

    try:
        # Extract ID from the filename and sort by it as numerical sorting
        df = df[df["ID"].notnull()]
        df["sort"] = df["ID"].str[len(filename_prefix) :].astype(int)
        df = df.sort_values("sort")
        df = df.drop("sort", axis=1)
    except ValueError as e:
        print("Not able to sort template by ID because: ", e)
        df = df.sort_values("ID")

    return df.reset_index(drop=True)


def check_data_in_template(template_file, sep=None):
    """Check if there is data different than null in each template"""

    dtype = {"ID": "string", "Target": "string"}
    df = read_template(template_file, sep=sep, dtype=dtype)

    return df, df.drop("ID", axis=1).notnull().any().any()


def read_template(template_file, sep=None, dtype=None):
    """Read template idenpendently of the extension"""

    # Extract extension and define loading method depending on it
    _, extension = os.path.splitext(template_file)
    if extension.startswith(".xls"):
        df = pd.read_excel(template_file, dtype=dtype)

    elif extension == ".csv":
        if sep is not None:
            df = pd.read_csv(template_file, sep=sep, dtype=dtype)
        else:
            df = pd.read_csv(template_file, sep=",", dtype=dtype)

        # Check if CSV has been loaded with the correct sep
        if len(df.columns) == 1:
            df = pd.read_csv(template_file, sep=";", dtype=dtype)
            if len(df.columns) == 1:
                raise ValueError(
                    "Please define the correct sep for the CSV files already existing as for examples: ",
                    template_file,
                )

    return df


def concat_templates(src_folder, excel=True, csv_sep=";"):
    """Concatenate all the template into a DataFrame"""

    # Define extension
    if excel:
        extension = ".xls*"
    else:
        extension = ".csv"

    template_paths = (
        glob(os.path.join(src_folder, "*" + extension))
        + glob(os.path.join(src_folder, "*/*" + extension))
        + glob(os.path.join(src_folder, "*/*/*" + extension))
    )

    dtype = {
        "ID": "string",
        "Target": "string",
        "Incorrect_image": "string",
        "Not_enough_quality": "string",
    }

    df = pd.DataFrame()
    for template_path in tqdm(template_paths, desc="Template files: "):
        # Check if there is any file which is out-dated
        _, template_filename = os.path.split(template_path)
        if template_filename.startswith("old_"):
            raise ValueError(
                f"The following file is out-dated, please move the data in this file to the corresponding file and remove the out-dated file.\n\n{template_path}"
            )

        if excel:
            current_df = pd.read_excel(template_path, dtype=dtype, engine="openpyxl")
        else:
            current_df = pd.read_csv(template_path, sep=csv_sep, dtype=dtype)

        current_df["Reviewers"] = get_reviewer(template_filename)
        current_df["Blocks"] = get_block(template_filename)

        df = pd.concat(
            [
                df,
                current_df,
            ]
        )

    df = df.reset_index(drop=True)

    df = normalize_difficulty_values(df)

    df = transform_to_ID_level(df)

    return df


def get_reviewer(template_filename):
    no_digits = re.search("\D+", template_filename).group(0)
    return no_digits.strip("_")


def get_block(template_filename):
    block = re.search("\d+", template_filename).group(0)
    return block


def normalize_difficulty_values(df):
    df["Difficulty"] = (
        df["Difficulty"]
        .str.lower()
        .map(
            {
                "baja": "0-baja",
                "bajo": "0-baja",
                "media": "1-media",
                "medio": "1-media",
                "alta": "2-alta",
                "alto": "2-alta",
                "dudosa": "3-dudosa",
                "dudoso": "3-dudosa",
            }
        )
    )

    return df


def transform_to_ID_level(df):
    # df = df[df.Target.notnull()] # All data is required to take into account additional reviewers
    for col in ["Difficulty", "Incorrect_image", "Not_enough_quality"]:
        df.loc[df[col].isnull(), col] = ""

    df = df.rename({"Target": "Targets"}, axis=1)
    df = df.groupby("ID").agg(
        {
            "Difficulty": "max",
            "Incorrect_image": "max",
            "Not_enough_quality": "max",
            "Reviewers": lambda revs: [rev for rev in revs],
            "Blocks": lambda blocks: [block for block in blocks],
            "Targets": lambda targets: [target for target in targets],
        }
    )

    df["Target"] = df["Targets"].apply(decide_final_target)

    return df


def decide_final_target(targets):
    counter = {}
    targets_s = pd.Series(targets)
    if targets_s.isnull().all():
        return targets[0]

    for target in targets_s[targets_s.notnull()]:
        if target in counter:
            counter[target] += 1
        else:
            counter[target] = 1

    counter = list(sorted(counter.items(), key=lambda item: item[1], reverse=True))

    # Tie case
    if len(counter) > 1 and counter[0][1] == counter[1][1]:
        return "Unclear fracture"
    else:
        return counter[0][0]
