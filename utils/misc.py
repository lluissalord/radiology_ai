""" Miscellaneous utility functions """

import random
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

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


class DebuggingCallback(Callback):
    order=101

    def __init__(self, start_epoch=0):
        self.start_epoch = start_epoch

    def before_batch(self):
        if self.start_epoch > self.epoch: return

        if self.training:
            print('TRAINING')
        else:
            print('EVALUATING')
        try:
            print('self.learn.xb (mean): ', self.learn.xb.mean())
        except AttributeError:
            print('self.learn.xb (mean): ', self.learn.xb[0].mean())
        # print('self.learn.xb: ', self.learn.xb)
        print('self.learn.yb: ', self.learn.yb)
    
    def after_pred(self):
        if self.start_epoch > self.epoch: return

        print('self.pred: ', self.pred)
        with torch.no_grad():
            try:
                print('Lx_criterion: ', self.loss_func.Lx_criterion(self.pred, *self.yb, reduction='mean'))
                print('Lu_criterion: ', self.loss_func.Lu_criterion(self.pred, *self.yb, reduction='mean'))
            except:
                pass

            print('loss: ', self.loss_func(self.pred, *self.yb))
        print('\n','-'*40,'\n')


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


def robust_split_data(df, test_size, target_col, seed=None):
    """ Split stratified data, in case of failing due to minor class too low, move it to test """

    filter_mask = pd.Series([True,] * len(df), index=df.index)
    done = False
    while not done:
        # Try to split stratify if error due to not enough minor class, then it goes to test
        try:
            train_df, test_df = train_test_split(
                df[filter_mask],
                test_size=test_size,
                shuffle=True,
                stratify=df.loc[filter_mask, target_col],
                random_state=seed
            )
            done = True
        except ValueError as e:
            if str(e).startswith('The least populated class'):
                minor_class = df.loc[filter_mask, target_col].value_counts().index[-1]
                filter_mask = (filter_mask) & (df[target_col] != minor_class)
            else:
                print('Test size is too low to use stratified, then split shuffling')
                train_df, test_df = train_test_split(
                    df[filter_mask],
                    test_size=test_size,
                    shuffle=True,
                    random_state=seed
                )
                done = True

    # Add minor classes which have not been initially included due to the error on train_test_split
    test_df = pd.concat(
        [
            test_df,
            df[~filter_mask]
        ],
        axis=0
    ).sample(frac=1)

    return train_df, test_df


def imbalance_robust_split_data(df, positive_df, test_size, positive_test_size, target_col, seed=None):
    """ Split between train and test according with the proportion of specified positives """

    # First split positive examples
    pos_train_df, pos_test_df = robust_split_data(positive_df, positive_test_size, target_col, seed=seed)

    # Identify as negative examples the ones from `df` which are not in `positive_df`
    negative_df = df.merge(positive_df, left_index=True, right_index=True, how='left', indicator=True, suffixes=('', '_'))
    negative_df = negative_df[negative_df['_merge'] == 'left_only'][list(df.columns)]

    # Split negative examples
    neg_test_size = (len(df) * test_size - len(pos_test_df)) / (len(df) - len(pos_train_df))
    neg_train_df, neg_test_df = train_test_split(
                    negative_df,
                    test_size=neg_test_size,
                    shuffle=True,
                    random_state=seed
                )

    # Join positive with negative examples and shuffle them
    train_df = pd.concat(
        [
            pos_train_df,
            neg_train_df
        ]
    ).sample(frac=1)

    test_df = pd.concat(
        [
            pos_test_df,
            neg_test_df
        ]
    ).sample(frac=1)

    return train_df, test_df