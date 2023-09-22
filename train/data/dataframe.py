from copy import deepcopy
from typing import Tuple
import pandas as pd
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from train.utils import *
from organize.relation import open_name_relation_file


def load_and_merge_all_data(run_params: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_df = pd.read_excel(
        os.path.join(run_params["PATH_PREFIX"], "all.xlsx"),
        dtype={"ID": "string", "Target": "string"},
        engine="openpyxl",
    )
    all_df = all_df.set_index("ID")
    labeled_mask = (all_df["Target"].notnull()) & (
        ~all_df["Target"].str.contains("Unknown")
    )
    label_df = all_df[labeled_mask]

    # Load DataFrame of relation between Original Filename and ID (IMG_XXX)
    relation_df = open_name_relation_file(
        os.path.join(run_params["PATH_PREFIX"], "relation.csv")
    )
    relation_df["Original_Filename"] = relation_df["Original_Filename"].apply(
        lambda path: Path(path).name
    )
    relation_df = relation_df.set_index("Filename")

    # Read all the metadata
    metadata_save_path = os.path.join(run_params["PATH_PREFIX"], "metadata_raw.csv")
    metadata_df = pd.read_csv(metadata_save_path)
    metadata_df["Original_Filename"] = metadata_df.fname.apply(
        lambda path: Path(path).name
    )
    metadata_df = metadata_df.set_index("Original_Filename")

    # Load DataFrame containing labels of OOS classifier ('ap', 'other')
    metadata_labels_path = os.path.join(
        run_params["PATH_PREFIX"], "metadata_labels.csv"
    )
    metadata_labels = pd.read_csv(metadata_labels_path)
    metadata_labels["Original_Filename"] = metadata_labels["Path"].apply(
        lambda path: Path(path).stem
    )
    metadata_labels = metadata_labels.set_index("Original_Filename")
    metadata_df = metadata_df.merge(metadata_labels, left_index=True, right_index=True)

    # Merge data to be able to load directly from preprocessed PNG file
    label_df = label_df.merge(relation_df, left_index=True, right_index=True)

    label_df = label_df.merge(
        metadata_df, left_on="Original_Filename", right_index=True, how="left"
    )

    label_df.index.name = "ID"
    if "ID" in label_df.columns:
        label_df = label_df.drop("ID", axis=1)
    label_df = label_df.reset_index(drop=False)

    unlabel_df = metadata_df[~metadata_df.index.isin(label_df["Original_Filename"])]

    # Define which column to use as the prediction
    if "Final_pred" in unlabel_df.columns:
        pred_col = "Final_pred"
    else:
        pred_col = "Pred"

    # Conditions for AP radiographies on unlabel data
    unlabel_df = unlabel_df[(unlabel_df[pred_col] == "ap")]
    unlabel_df = unlabel_df.reset_index(drop=False)

    return label_df, unlabel_df


def add_png_path(df, run_params):
    df["Raw_preprocess"] = df["Original_Filename"].apply(
        lambda filename: os.path.join(
            run_params["RAW_PREPROCESS_FOLDER"], filename + ".png"
        )
    )
    return df


def filter_centers_mask(label_df):
    # Filter metadata to only sent images fulfiling condition
    return (
        (
            (
                label_df.InstitutionName.str.lower()
                .str.contains("coslada")
                .astype(bool)
                | label_df.InstitutionName.str.lower()
                .str.contains("cugat")
                .astype(bool)
            )
            & (label_df.InstitutionName.notnull())
        )
        | (
            label_df.AccessionNumber.astype("str").str.startswith("885")
            # | label_df.AccessionNumber.astype('str').str.startswith('4104')
        )
        | (label_df["Target"] != "0")
    )


def robust_split_data(df, test_size, target_col, seed=None):
    """Split stratified data, in case of failing due to minor class too low, move it to test"""

    filter_mask = pd.Series(
        [
            True,
        ]
        * len(df),
        index=df.index,
    )
    done = False
    while not done:
        # Try to split stratify if error due to not enough minor class, then it goes to test
        try:
            train_df, test_df = train_test_split(
                df[filter_mask],
                test_size=test_size,
                shuffle=True,
                stratify=df.loc[filter_mask, target_col],
                random_state=seed,
            )
            done = True
        except ValueError as e:
            if str(e).startswith("The least populated class"):
                minor_class = df.loc[filter_mask, target_col].value_counts().index[-1]
                filter_mask = (filter_mask) & (df[target_col] != minor_class)
            else:
                print("Test size is too low to use stratified, then split shuffling")
                train_df, test_df = train_test_split(
                    df[filter_mask],
                    test_size=test_size,
                    shuffle=True,
                    random_state=seed,
                )
                done = True

    # Add minor classes which have not been initially included due to the error on train_test_split

    test_df = shuffle(pd.concat([test_df, df[~filter_mask]], axis=0), random_state=seed)

    return train_df, test_df


def imbalance_robust_split_data(
    df, positive_df, test_size, positive_test_size, target_col, seed=None
):
    """Split between train and test according with the proportion of specified positives"""

    # First split positive examples
    pos_train_df, pos_test_df = robust_split_data(
        positive_df, positive_test_size, target_col, seed=seed
    )

    # Identify as negative examples the ones from `df` which are not in `positive_df`
    negative_df = df.merge(
        positive_df,
        left_index=True,
        right_index=True,
        how="left",
        indicator=True,
        suffixes=("", "_"),
    )
    negative_df = negative_df.loc[negative_df["_merge"] == "left_only", df.columns]

    # Split negative examples
    neg_test_size = (len(df) * test_size - len(pos_test_df)) / (
        len(df) - len(pos_train_df)
    )
    neg_train_df, neg_test_df = train_test_split(
        negative_df, test_size=neg_test_size, shuffle=True, random_state=seed
    )

    # Join positive with negative examples and shuffle them
    train_df = shuffle(pd.concat([pos_train_df, neg_train_df]), random_state=seed)

    test_df = shuffle(pd.concat([pos_test_df, neg_test_df]), random_state=seed)

    assert len(pos_train_df) + len(neg_train_df) == len(train_df)
    assert len(pos_test_df) + len(neg_test_df) == len(test_df)
    assert len(train_df) + len(test_df) == len(df)

    return train_df, test_df


def get_ratio(df, target_col="Target"):
    targets = (df[target_col] != "0").sum()
    non_targets = (df[target_col] == "0").sum()

    ratio = targets / non_targets

    return ratio


def rebalance_equal_to_target_df(df, target_df, target_col="Target", seed=42):
    dataset_ratio = get_ratio(df, target_col=target_col)
    target_dataset_ratio = get_ratio(target_df, target_col=target_col)

    non_target_oversampling_ratio = dataset_ratio / target_dataset_ratio

    if non_target_oversampling_ratio > 1:
        rebalanced_df = shuffle(
            pd.concat(
                [
                    df[df[target_col] == "0"],
                    df[df[target_col] == "0"].sample(
                        frac=non_target_oversampling_ratio - 1, random_state=seed
                    ),
                    df[df[target_col] != "0"],
                ]
            ),
            random_state=seed,
        ).reset_index(drop=True)
    else:
        rebalanced_df = shuffle(
            pd.concat(
                [
                    df[df[target_col] == "0"].sample(
                        frac=non_target_oversampling_ratio, random_state=seed
                    ),
                    df[df[target_col] != "0"],
                ]
            ),
            random_state=seed,
        ).reset_index(drop=True)

    return rebalanced_df


def split_datasets(label_df, run_params):
    """Split between train, valid and test according with the proportion of specified positives"""
    if run_params["TEST_SIZE"] != 0:
        if run_params["POSITIVES_ON_TRAIN"]:
            positive_test_size = (1 - run_params["POSITIVES_ON_TRAIN"]) * (
                run_params["TEST_SIZE"]
                / (run_params["VALID_SIZE"] + run_params["TEST_SIZE"])
            )
            train_df, test_df = imbalance_robust_split_data(
                label_df,
                label_df[label_df["Target"] != "0"],
                test_size=run_params["TEST_SIZE"],
                positive_test_size=positive_test_size,
                target_col="Target",
                seed=run_params["DATA_SEED"],
            )
        else:
            train_df, test_df = robust_split_data(
                label_df,
                run_params["TEST_SIZE"],
                "Target",
                seed=run_params["DATA_SEED"],
            )
    else:
        test_df = pd.DataFrame([], columns=label_df.columns)
        train_df = label_df

    if run_params["VALID_SIZE"] != 0:
        if run_params["POSITIVES_ON_TRAIN"]:
            positive_test_size = (1 - run_params["POSITIVES_ON_TRAIN"]) * (
                run_params["TEST_SIZE"]
                / (run_params["VALID_SIZE"] + run_params["TEST_SIZE"])
            )
            positive_valid_size = (
                1 - run_params["POSITIVES_ON_TRAIN"] - positive_test_size
            ) / (1 - positive_test_size)
            train_df, val_df = imbalance_robust_split_data(
                train_df,
                train_df[train_df["Target"] != "0"],
                test_size=run_params["VALID_SIZE"] / (1 - run_params["TEST_SIZE"]),
                positive_test_size=positive_valid_size,
                target_col="Target",
                seed=run_params["DATA_SEED"],
            )
        else:
            train_df, val_df = robust_split_data(
                train_df,
                run_params["VALID_SIZE"],
                "Target",
                seed=run_params["DATA_SEED"],
            )
    else:
        val_df = pd.DataFrame([], columns=label_df.columns)

    train_df["Dataset"] = "train"
    val_df["Dataset"] = "valid"
    test_df["Dataset"] = "test"

    label_df = pd.concat(
        [
            train_df,
            val_df,
            test_df,
        ]
    ).reset_index(drop=True)

    return label_df


def rebalance_datasets(label_df, run_params):
    """Modify positive-negative proportion on each dataset to meet specification of positives.
    Test with same proportion as the initial dataset, Dev same as train and Valid as initial dataset.
    """

    if run_params["POSITIVES_ON_TRAIN"]:
        if run_params["TEST_SIZE"] != 0:
            # Test set should mantain balance equal to the original data
            test_df = rebalance_equal_to_target_df(
                label_df[label_df["Dataset"] == "test"],
                label_df,
                target_col="Target",
                seed=run_params["DATA_SEED"],
            )
        else:
            test_df = pd.DataFrame([], columns=label_df.columns)

        if run_params["VALID_SIZE"] != 0:
            # Dev set is use to track overfitting so it better to be proportional to Training set
            dev_df = rebalance_equal_to_target_df(
                label_df[label_df["Dataset"] == "valid"],
                label_df[label_df["Dataset"] == "train"],
                target_col="Target",
                seed=run_params["DATA_SEED"],
            )

            # Validation set is used to represent the real proportion
            val_df = rebalance_equal_to_target_df(
                label_df[label_df["Dataset"] == "valid"],
                label_df,
                target_col="Target",
                seed=run_params["DATA_SEED"],
            )
        else:
            dev_df = pd.DataFrame([], columns=label_df.columns)
            val_df = pd.DataFrame([], columns=label_df.columns)
    else:
        dev_df = pd.DataFrame([], columns=label_df.columns)
        val_df = pd.DataFrame([], columns=label_df.columns)
        test_df = pd.DataFrame([], columns=label_df.columns)

    dev_df["Dataset"] = "dev"
    val_df["Dataset"] = "valid"
    test_df["Dataset"] = "test"

    label_df = pd.concat(
        [
            label_df[label_df["Dataset"] == "train"],
            dev_df,
            val_df,
            test_df,
        ]
    ).reset_index(drop=True)

    return label_df


def split_KFolds(label_df, run_params):
    """Generate K-Folds defined on `Fold` column, depending on `K_FOLDS` param.
    `Dataset`column is specified depending on `K` param.
    Current implementation is equivalent to K-Folds but using robust split."""

    modified_run_params = run_params.copy()

    valid_size = 1 / run_params["K_FOLDS"]
    folds = []
    for k in range(run_params["K_FOLDS"] - 1):
        modified_run_params["VALID_SIZE"] = valid_size / (1 - k / run_params["K_FOLDS"])
        modified_run_params["POSITIVES_ON_TRAIN"] = (
            1 - modified_run_params["VALID_SIZE"] - modified_run_params["TEST_SIZE"]
        )
        if not k:
            tmp_label_df = split_datasets(label_df, modified_run_params)
            test_df = tmp_label_df[tmp_label_df["Dataset"] == "test"].copy()
            if len(test_df.index):
                test_df["Fold"] = -1
                folds.append(test_df.copy())
            modified_run_params["TEST_SIZE"] = 0
        else:
            tmp_label_df = split_datasets(train_df, modified_run_params)

        train_df = tmp_label_df[tmp_label_df["Dataset"] == "train"].copy()
        val_df = tmp_label_df[tmp_label_df["Dataset"] == "valid"].copy()
        val_df["Dataset"] = "valid" if k == modified_run_params["K"] else "train"
        val_df["Fold"] = k
        folds.append(val_df.copy())

    train_df["Dataset"] = "valid" if k + 1 == modified_run_params["K"] else "train"
    train_df["Fold"] = k + 1
    folds.append(train_df.copy())

    label_df = pd.concat(folds).reset_index(drop=True)

    return label_df


def split_train_dev_valid_test_data(label_df, run_params):
    if run_params["K_FOLDS"] and run_params["K"] is not None:
        label_df = split_KFolds(label_df, run_params)
    else:
        label_df = split_datasets(label_df, run_params)

    label_df = rebalance_datasets(label_df, run_params)

    return label_df


def generate_dfs(run_params, filter_centers: bool = False, debug: bool = True):
    # TODO: Add case of IN_COLAB to generate DataFrame
    label_df, unlabel_df = load_and_merge_all_data(run_params)

    if run_params["BINARY_CLASSIFICATION"]:
        label_df["Target"] = (label_df["Target"] != "0").astype(int).astype("string")
    else:
        label_df = label_df[~label_df["Target"].str.isalpha()]
        unlabel_df = pd.concat(
            [
                unlabel_df,
                label_df.loc[label_df["Target"].str.isalpha(), unlabel_df.columns],
            ],
            axis=0,
        )

    # Add PNG path to be load directly from preprocessed image
    label_df = add_png_path(label_df, run_params)
    unlabel_df = add_png_path(unlabel_df, run_params)

    # Filter data to only the ones from desired centers
    if filter_centers:
        filtered_label_mask = filter_centers_mask(label_df)
        print(f"Data from centers: {filtered_label_mask.sum()}")
        print(
            f"Filtered data moved to unlabel_df: {len(label_df.loc[~filtered_label_mask, unlabel_df.columns])}"
        )
        unlabel_df = pd.concat(
            [
                unlabel_df,
                deepcopy(label_df.loc[~filtered_label_mask, unlabel_df.columns]),
            ],
            axis=0,
        )
        label_df = label_df[filtered_label_mask]

    label_df = split_train_dev_valid_test_data(label_df, run_params)

    if debug:
        print(f"Currently {len(label_df.index)} data have been labelled")
        print(f"Remaining {len(unlabel_df.index)} data to be labelled")
        print("\nSplit of labelled data is:")
        display(label_df["Dataset"].value_counts())

    sort_dataset = {"train": 0, "dev": 1, "valid": 2, "test": 3}
    label_df = label_df.sort_values(
        "Dataset", key=lambda x: x.map(sort_dataset)
    ).reset_index(drop=True)

    return label_df, unlabel_df
