import torch
import pandas as pd

# TODO: Review if bt is required
def interleave(x):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([-1, s[0]] + s[1:]), 1, 0), [-1] + s[1:])

def de_interleave(x):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([s[0], -1] + s[1:]), 1, 0), [-1] + s[1:])