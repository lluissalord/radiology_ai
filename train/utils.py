""" Miscellaneous utility functions """

import random
import pandas as pd
import numpy as np
import os

import torch

from fastai.callback.core import Callback


class DebuggingCallback(Callback):
    order = 101

    def __init__(self, start_epoch=0):
        self.start_epoch = start_epoch

    def before_batch(self):
        if self.start_epoch > self.epoch:
            return

        if self.training:
            print("TRAINING")
        else:
            print("EVALUATING")
        try:
            print("self.learn.xb (mean): ", self.learn.xb.mean())
        except AttributeError:
            print("self.learn.xb (mean): ", self.learn.xb[0].mean())
        # print('self.learn.xb: ', self.learn.xb)
        print("self.learn.yb: ", self.learn.yb)

    def after_pred(self):
        if self.start_epoch > self.epoch:
            return

        print("self.pred: ", self.pred)
        with torch.no_grad():
            try:
                print(
                    "Lx_criterion: ",
                    self.loss_func.Lx_criterion(self.pred, *self.yb, reduction="mean"),
                )
                print(
                    "Lu_criterion: ",
                    self.loss_func.Lu_criterion(self.pred, *self.yb, reduction="mean"),
                )
            except:
                pass

            print("loss: ", self.loss_func(self.pred, *self.yb))
        print("\n", "-" * 40, "\n")


def TestColSplitter(col="Dataset"):
    "Split `items` (supposed to be a dataframe) by value in `col`"

    from fastai.data.transforms import IndexSplitter
    from fastai.basics import mask2idxs

    def _inner(o):
        assert isinstance(
            o, pd.DataFrame
        ), "ColSplitter only works when your items are a pandas DataFrame"
        all_vals = o.iloc[:, col] if isinstance(col, int) else o[col]
        unique_vals = all_vals.unique()
        idxs = []
        for unique_value in unique_vals:
            idxs.append(all_vals == unique_value)

        return tuple([IndexSplitter(mask2idxs(idx))(o)[1] for idx in idxs])

    return _inner


def seed_everything(use_seed=0):
    seed = use_seed if use_seed else random.randint(1, 1000000)
    print(f"Using seed: {seed}")

    # python RNG
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # pytorch RNGs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if use_seed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # numpy RNG
    np.random.seed(seed)

    return seed
