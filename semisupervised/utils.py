""" Utility functions for Semi-Supervised Learning """

import torch
import pandas as pd

def interleave(x, bs):
    """ Interleave to mix batch data """
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([-1, bs] + s[1:]), 1, 0), [-1] + s[1:])

def de_interleave(x, bs):
    """ Deinterleave to undo mix of batch data """
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([bs, -1] + s[1:]), 1, 0), [-1] + s[1:])