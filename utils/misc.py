""" Miscellaneous utility functions """

import random
import pandas as pd
import numpy as np
import os

import torch

from fastai.callback.core import Callback

class GradientClipping(Callback):
    "Gradient clipping during training."
    def __init__(self, clip:float = 0.):
        super().__init__()
        self.clip = clip

    def after_backward(self):
        "Clip the gradient before the optimizer step."
        if self.clip: torch.nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)


def categorical_to_one_hot(x, n_out):
    """ Transform categorical tensor to one hot encoding """
    zeros = torch.zeros(len(x), n_out)
    if torch.cuda.is_available():
        zeros = zeros.cuda()

    return zeros.scatter_(1, x.view(-1,1).long(), 1)


def TestColSplitter(col='Dataset'):
    "Split `items` (supposed to be a dataframe) by value in `col`"

    from fastai.data.transforms import IndexSplitter
    from fastai.basics import mask2idxs

    def _inner(o):
        assert isinstance(o, pd.DataFrame), "ColSplitter only works when your items are a pandas DataFrame"
        train_idx = (o.iloc[:,col] if isinstance(col, int) else o[col]) == 'train'
        valid_idx = (o.iloc[:,col] if isinstance(col, int) else o[col]) == 'valid'
        test_idx = (o.iloc[:,col] if isinstance(col, int) else o[col]) == 'test'
        return IndexSplitter(mask2idxs(train_idx))(o)[1], IndexSplitter(mask2idxs(valid_idx))(o)[1], IndexSplitter(mask2idxs(test_idx))(o)[1]
    return _inner


def seed_everything(use_seed=0):
    seed = use_seed if use_seed else random.randint(1,1000000)
    print(f"Using seed: {seed}")

    # python RNG
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # pytorch RNGs
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if use_seed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # numpy RNG
    np.random.seed(seed)

    return seed