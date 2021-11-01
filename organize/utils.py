import os
from glob import glob
import shutil


def expand_list(l, n):
    """Expand a list `l` to repeat its elements till reaching length of `n`"""
    return (l * (n // len(l) + 1))[:n]


def check_generic_path(path):
    """Check if path exists even if folder name modified adding suffix"""

    if os.path.exists(path):
        return True

    else:
        candidates = (
            glob(path + " *")
            + glob(path + "-*")
            + glob(path + ".*")
            + glob(path + "_*")
        )
        if len(candidates) == 1:
            return True
        elif len(candidates) > 1:
            print(f"For path {path} there are duplicates paths")
            return True
        else:
            return False

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