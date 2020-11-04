import torch
import pandas as pd

from fastai.data.transforms import IndexSplitter
from fastai.basics import mask2idxs

def TestColSplitter(col='Dataset'):
    "Split `items` (supposed to be a dataframe) by value in `col`"
    def _inner(o):
        assert isinstance(o, pd.DataFrame), "ColSplitter only works when your items are a pandas DataFrame"
        train_idx = (o.iloc[:,col] if isinstance(col, int) else o[col]) == 'train'
        valid_idx = (o.iloc[:,col] if isinstance(col, int) else o[col]) == 'valid'
        test_idx = (o.iloc[:,col] if isinstance(col, int) else o[col]) == 'test'
        return IndexSplitter(mask2idxs(train_idx))(o)[1], IndexSplitter(mask2idxs(valid_idx))(o)[1], IndexSplitter(mask2idxs(test_idx))(o)[1]
    return _inner

def interleave(x):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([-1, s[0]] + s[1:]), 1, 0), [-1] + s[1:])

def de_interleave(x):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([s[0], -1] + s[1:]), 1, 0), [-1] + s[1:])