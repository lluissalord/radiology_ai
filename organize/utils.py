import os
from glob import glob


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
