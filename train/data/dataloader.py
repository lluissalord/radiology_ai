""" Data utilities """

import numpy as np

from fastai.data.core import TfmdDL
from fastai.data.block import *
from fastai.data.transforms import *
from fastai.vision import models
from fastai.vision.learner import *
from fastai.vision.augment import *
from fastai.vision.core import PILImageBW, PILImage
from fastai.vision.data import *

from preprocessing.transforms import PILPNGFloat, PILPNGFloatBW
from train.utils import TestColSplitter


class AllLabelsInBatchDL(TfmdDL):
    """DataLoader which allows to have a minimum of samples of all the labels in each batch"""

    def __init__(self, dataset=None, min_samples=1, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        if self.bs < len(self.vocab):
            print(
                "AllLabelsInBatchDL working as simple DL because batch size is less than number of labels"
            )
            self.min_samples = 0
        else:
            self.min_samples = min_samples

    def get_idxs(self):
        if self.n == 0:
            return []
        idxs = super().get_idxs()
        if not self.shuffle:
            return idxs

        # Transform to numpy array to replace efficiently
        idxs = np.array(idxs)

        # Generate random indexes which will be substituted by the labels
        n_batches = self.n // self.bs
        idxs_subs = [
            np.random.choice(self.bs, len(self.vocab) * self.min_samples, replace=False)
            + i * self.bs
            for i in range(n_batches)
        ]

        # Iterate along batches and substitute selected indexes with label indexes
        for batch_idxs_subs in idxs_subs:
            label_idxs = []
            for label in self.vocab:
                # Extract indexes of current label and randomly choose `min_samples`
                label_idx = list(
                    self.items[self.col_reader[1](self.items) == label].index
                )
                label_idx = list(
                    np.random.choice(label_idx, size=self.min_samples, replace=True)
                )

                label_idxs = label_idxs + label_idx

            # Shuffle label indexes and replace them
            np.random.shuffle(label_idxs)
            idxs[batch_idxs_subs] = label_idxs

        return idxs


def define_ds_params(run_params, item_tfms, label_tfms):
    base_ds_params = {
        # 'get_x': ColReader('Original_Filename', pref=RAW_PREPROCESS_FOLDER+'/', suff='.png'),
        "get_x": ColReader("Raw_preprocess"),
        # 'get_x': ColReader('Original'),
        "item_tfms": item_tfms,
    }

    # Specific parameters for Label Dataset
    label_ds_params = base_ds_params.copy()
    if run_params["SELF_SUPERVISED"]:
        label_ds_params["blocks"] = (ImageBlock(cls=PILPNGFloat), CategoryBlock)
    else:
        label_ds_params["blocks"] = (ImageBlock(cls=PILPNGFloatBW), CategoryBlock)
    # label_ds_params['blocks'] = (ImageBlock(cls=PILDicom_scaled), MultiCategoryBlock)

    label_ds_params["get_y"] = ColReader("Target")
    label_ds_params["splitter"] = TestColSplitter(col="Dataset")
    label_ds_params["batch_tfms"] = label_tfms

    # Specific parameters for Unlabel Dataset
    unlabel_ds_params = None
    if run_params["SSL"]:
        unlabel_ds_params = base_ds_params.copy()
        if run_params["SELF_SUPERVISED"]:
            unlabel_ds_params["blocks"] = ImageBlock(cls=PILPNGFloat)
        else:
            unlabel_ds_params["blocks"] = ImageBlock(cls=PILPNGFloatBW)
        # unlabel_ds_params['blocks'] = (ImageBlock(cls=PILDicom_scaled))

        unlabel_ds_params["splitter"] = RandomSplitter(0)

    return label_ds_params, unlabel_ds_params


def define_dls_params(run_params):
    dls_params = {
        "bs": run_params["BATCH_SIZE"],
        "num_workers": 0,
        "shuffle_train": True,
        "drop_last": True,
    }

    unlabel_dls_params = None
    if run_params["SSL"]:
        unlabel_dls_params = dls_params.copy()
        if (
            run_params["SSL"] == run_params["SSL_FIX_MATCH"]
            and run_params["LAMBDA_U"] != 0
        ):
            unlabel_dls_params["bs"] = run_params["BATCH_SIZE"] * run_params["MU"]

    return dls_params, unlabel_dls_params


def get_label_dl(run_params, dls_params, label_ds_params, label_df):
    label_db = DataBlock(**label_ds_params)
    label_dl = label_db.dataloaders(label_df, **dls_params)

    if run_params["ALL_LABELS_IN_BATCH"]:
        # Create DataLoader which allows to have a minimum of samples of all the labels in each batch
        new_dl = DataBlock(**label_ds_params).dataloaders(
            label_df,
            **dls_params,
            dl_type=AllLabelsInBatchDL,
            min_samples=run_params["MIN_SAMPLES_PER_LABEL"]
        )
        label_dl.train = new_dl.train

    elif run_params["WEIGTHED_SAMPLER"]:
        # Calculate sample weights to balance the DataLoader
        from collections import Counter

        count = Counter(label_dl.items["Target"])
        class_weights = {}
        for c in count:
            class_weights[c] = 1 / count[c]
        wgts = (
            label_dl.items["Target"]
            .map(class_weights)
            .values[: len(label_df[label_df["Dataset"] == "train"])]
        )

        # Create weigthed dataloader
        weighted_dl = DataBlock(**label_ds_params).dataloaders(
            label_df, **dls_params, dl_type=WeightedDL, wgts=wgts
        )
        label_dl.train = weighted_dl.train

    return label_dl


def get_unlabel_dls(
    run_params, unlabel_tfms, unlabel_dls_params, unlabel_ds_params, unlabel_df
):
    unlabel_dls = [
        DataBlock(**unlabel_ds_params, batch_tfms=batch_tfms).dataloaders(
            unlabel_df, **unlabel_dls_params
        )
        for batch_tfms in unlabel_tfms
    ]

    return unlabel_dls
