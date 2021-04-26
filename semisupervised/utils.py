""" Utility functions for Semi-Supervised Learning """

import torch
import pandas as pd
import numpy as np


def interleave(x, bs):
    """ Interleave to mix batch data """
    s = list(x.shape)
    idx = np.argmax((np.array(s) != 1)) + 1
    ret = torch.reshape(torch.transpose(x.reshape([-1, bs] + s[idx:]), 1, 0), s)
    return ret


def de_interleave(x, bs):
    """ Deinterleave to undo mix of batch data """
    s = list(x.shape)
    idx = np.argmax((np.array(s) != 1)) + 1
    ret = torch.reshape(torch.transpose(x.reshape([bs, -1] + s[idx:]), 1, 0), s)
    return ret
