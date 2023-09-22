import os
import pandas as pd
import numpy as np

from organize.utils import check_generic_path


def open_name_relation_file(filepath, sep=","):
    """Extract DataFrame from file containing the relation between original and new files"""

    # Check if file exists
    if not os.path.exists(filepath):
        df = pd.DataFrame(
            columns=["Original", "New_Path", "Filename", "Original_Filename"]
        )
    else:
        df = pd.read_csv(filepath, sep=sep, index_col=0)
        df.columns = [column if column != "Path" else "Dataset_Path" for column in df.columns]

    return df


def check_inconsistent_relation_file(relation_df):
    if len(relation_df.index) == 0:
        raise ValueError(
            "No reset is set, but there is no relation file or it is empty"
        )
    if relation_df.index.duplicated("Original").any():
        raise ValueError(
            "There is a duplicated value on relation file, please review it and modify it"
        )


def save_name_relation_file(relation_df, filepath, sep=","):
    """Save file containing the relation between original and new files"""
    relation_df[relation_df["Original_Filename"].notnull()].to_csv(
        filepath, sep=sep, index=True
    )


def get_last_id(relation_df, prefix="IMG_"):
    """Get the ID of the last filename from current relation of files"""

    # Extract the maximum ID currently set
    new_id = relation_df["Filename"].str.split(prefix).str[1].astype(int).max()

    # In case of nan then set it to -1
    if new_id is np.nan:
        new_id = -1

    return new_id


def add_new_relation(
    relation_df, src_path, src_filename, new_filename, path=None, check_conflict=False
):
    """Add a new relation on the relation DataFrame"""

    # Check it does not exist a conflictive addition
    # if src_path in relation_df.index and 'Original_Filename' in relation_df.columns and not np.isnan(relation_df.loc[src_path, 'Original_Filename']):
    if check_conflict and (
        src_path in relation_df.index
        and "Original_Filename" in relation_df.columns
        and relation_df["Original_Filename"].notnull()[src_path]
    ):
        if relation_df.loc[src_path, "Filename"] != new_filename:
            raise ValueError(
                f'For file "{src_path}"" there is already a relation with "{relation_df.loc[src_path, "Filename"]}" but it is being added for "{new_filename}"'
            )
    else:
        relation_df.loc[src_path, "Filename"] = new_filename
        relation_df.loc[src_path, "Original_Filename"] = src_filename
        if path:
            relation_df.loc[src_path, "Dataset_Path"] = path

    return relation_df


def update_block_relation(relation_df, parent_folder, block, new_folder, sep="/"):
    """Replace the old folder names by the new folder only to the paths where the block appears"""

    relation_df.loc[relation_df[""].str.endswith(block), ""] = relation_df.loc[
        relation_df[""].str.endswith(block), ""
    ].str.replace(
        f"((?<={os.path.split(parent_folder)[-1]}{sep})(.*)(?={sep}{block}$))",
        new_folder,
    )

    check_relation(relation_df, check_path=True, check_raw=False)

    return relation_df


def check_relation(relation_df, check_path=True, check_raw=True):
    """Check that the relation on the relation DataFrame is preserved"""

    # Check current path
    if check_path:
        print("Checking relation file are in the correct path...")
        if type(relation_df) is pd.DataFrame:
            check = relation_df[""].apply(check_generic_path)
        else:
            check = pd.Series(check_generic_path(relation_df[""]))

        # Raise error if not true for all the cases
        if not check.all():
            raise ValueError(
                f"The following cases do not have correct `Path` on relation DataFrame:\n{relation_df.loc[~check]}"
            )

    # Check raw path
    if check_raw:
        print("Checking relation file are in the raw path...")
        if type(relation_df) is pd.DataFrame:
            check = pd.Series(
                relation_df.index.map(lambda x: os.path.exists(x)),
                index=relation_df.index,
                dtype=bool,
            )
        else:
            check = pd.Series(os.path.exists(relation_df.name))

        # Raise error if not true for all the cases
        if not check.all():
            raise ValueError(
                f"The following cases do not have correct `Original` on relation DataFrame:\n{relation_df.loc[~check]}"
            )

    print("All in place!")
    return True
