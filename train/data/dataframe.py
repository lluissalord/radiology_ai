import pandas as pd
import os
from pathlib import Path

from organize.templates import concat_templates

from train.utils import *


def add_png_path(df, run_params):
    # Load DataFrame of relation between Original Filename and ID (IMG_XXX)
    relation_df = pd.read_csv(os.path.join(run_params["PATH_PREFIX"], "relation.csv"))
    relation_df = relation_df.set_index("Filename")

    # Merge data to be able to load directly from preprocessed PNG file
    final_df = df.set_index("ID").merge(relation_df, left_index=True, right_index=True)
    final_df["ID"] = final_df.index.values
    final_df = final_df.reset_index(drop=True)
    final_df["Raw_preprocess"] = final_df["Original_Filename"].apply(
        lambda filename: os.path.join(
            run_params["RAW_PREPROCESS_FOLDER"], filename + ".png"
        )
    )

    return final_df


def filter_centers_data(df, run_params):
    # Read all the sources
    metadata_save_path = os.path.join(run_params["PATH_PREFIX"], "metadata_raw.csv")
    metadata_df = pd.read_csv(metadata_save_path)

    # Filter metadata to only sent images fulfiling condition
    filter_metadata_df = metadata_df[
        (
            metadata_df.InstitutionName.str.lower().str.contains("coslada").astype(bool)
            | metadata_df.InstitutionName.str.lower().str.contains("cugat").astype(bool)
        )
        & (metadata_df.InstitutionName.notnull())
        | (
            metadata_df.AccessionNumber.astype("str").str.startswith("885")
            # | metadata_df.AccessionNumber.astype('str').str.startswith('4104')
        )
    ]

    # Create DataFrame only with the Filename
    filter_df = pd.DataFrame(
        index=filter_metadata_df.fname.apply(lambda x: Path(x).name)
    )
    filter_df["check_condition"] = True

    # Filter data to only the ones from desired centers
    final_df = df.merge(
        filter_df, left_on="Original_Filename", right_index=True, how="left"
    )
    final_df = final_df[
        (final_df["check_condition"] == True) | (final_df["Target"] != "0")
    ]

    return final_df


def split_by_labelled_data(df, run_params):
    # Load DataFrame containing labels of OOS classifier ('ap', 'other')
    metadata_labels_path = os.path.join(
        run_params["PATH_PREFIX"], "metadata_labels.csv"
    )
    metadata_labels = pd.read_csv(metadata_labels_path)
    metadata_labels["Original_Filename"] = metadata_labels["Path"].apply(
        lambda path: Path(path).stem
    )
    metadata_labels = metadata_labels.set_index("Original_Filename")

    # Merge all the data we have with the labelling in order to split correctly according to OOS classifier
    unlabel_all_df = metadata_labels.merge(
        df.set_index("Original_Filename"), how="left", left_index=True, right_index=True
    )
    unlabel_all_df = unlabel_all_df[unlabel_all_df.Target.isnull()]
    unlabel_all_df["Original_Filename"] = unlabel_all_df.index.values
    unlabel_all_df["Raw_preprocess"] = unlabel_all_df["Original_Filename"].apply(
        lambda filename: os.path.join(
            run_params["RAW_PREPROCESS_FOLDER"], filename + ".png"
        )
    )

    # Define which column to use as the prediction
    if "Final_pred" in unlabel_all_df.columns:
        pred_col = "Final_pred"
    else:
        pred_col = "Pred"

    # Conditions for AP radiographies on unlabel data
    ap_match = (unlabel_all_df[pred_col] == "ap") & (
        unlabel_all_df.Incorrect_image.isnull()
    )

    # Split between label_df (labelled data), `unlabel_df` (containing only AP) and `unlabel_not_ap_df` (with the rest of unlabel data)
    label_df = df[df["Target"].notnull()].reset_index(drop=True)
    unlabel_df = unlabel_all_df[ap_match].reset_index(drop=True)
    unlabel_not_ap_df = unlabel_all_df[~ap_match].reset_index(drop=True)

    return label_df, unlabel_df, unlabel_not_ap_df


def split_train_dev_valid_test_data(label_df, run_params):
    # Split between train, valid and test according with the proportion of specified positives
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

    if run_params["POSITIVES_ON_TRAIN"]:
        if run_params["TEST_SIZE"] != 0:
            # Test set should mantain balance equal to the original data
            test_df = rebalance_equal_to_target_df(
                test_df, label_df, target_col="Target", seed=run_params["DATA_SEED"]
            )

        if run_params["VALID_SIZE"] != 0:
            # Dev set is use to track overfitting so it better to be proportional to Training set
            dev_df = rebalance_equal_to_target_df(
                val_df, train_df, target_col="Target", seed=run_params["DATA_SEED"]
            )

            # Validation set is used to represent the real proportion
            val_df = rebalance_equal_to_target_df(
                val_df, label_df, target_col="Target", seed=run_params["DATA_SEED"]
            )
        else:
            dev_df = pd.DataFrame([], columns=label_df.columns)
    else:
        dev_df = pd.DataFrame([], columns=label_df.columns)

    train_df["Dataset"] = "train"
    dev_df["Dataset"] = "dev"
    val_df["Dataset"] = "valid"
    test_df["Dataset"] = "test"

    label_df = pd.concat(
        [
            train_df,
            dev_df,
            val_df,
            test_df,
        ]
    ).reset_index(drop=True)

    return label_df


def generate_dfs(run_params, debug=True):
    # TODO: Add case of IN_COLAB to generate DataFrame
    df = pd.read_excel(
        os.path.join(run_params["PATH_PREFIX"], "all.xlsx"),
        dtype={"ID": "string", "Target": "string"},
        engine="openpyxl",
    )

    # Add PNG path to be load directly from preprocessed image
    final_df = add_png_path(df, run_params)

    # Filter data to only the ones from desired centers
    final_df = filter_centers_data(final_df, run_params)

    # Split data between labelled, unlabelled and no-AP unlabelled data
    label_df, unlabel_df, unlabel_not_ap_df = split_by_labelled_data(
        final_df, run_params
    )

    if debug:
        print(f"Currently {len(label_df.index)} data have been labelled")
        print(f"Remaining {len(unlabel_df.index)} data to be labelled")
        print(f"Discarded {len(unlabel_not_ap_df.index)} data")

    label_df = split_train_dev_valid_test_data(label_df, run_params)

    if debug:
        print("\nSplit of labelled data is:")
        display(label_df["Dataset"].value_counts())

    sort_dataset = {"train": 0, "dev": 1, "valid": 2, "test": 3}
    label_df = label_df.sort_values(
        "Dataset", key=lambda x: x.map(sort_dataset)
    ).reset_index(drop=True)

    if run_params["BINARY_CLASSIFICATION"]:
        label_df["Target"] = (label_df["Target"] != "0").astype(int).astype("string")
        # train_df['Target'] = (train_df['Target'] != '0').astype(int).astype('string')
        # val_df['Target'] = (val_df['Target'] != '0').astype(int).astype('string')
        # test_df['Target'] = (test_df['Target'] != '0').astype(int).astype('string')

    return label_df, unlabel_df, final_df
