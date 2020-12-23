import random

import torch
import torchvision.transforms.functional as TF

from skimage.transform import resize as sk_resize
from PIL import Image
from pydicom import dcmread

from preprocessing.dicom import dcm_scale


def init_bins(fnames, n_samples=None, isDCM=True):
    """ Initialize bins to equally distribute the histogram of the dataset """

    # Select randomly n_samples
    if n_samples is not None:
        fnames_sample = fnames.copy()
        random.shuffle(fnames_sample)
        fnames_sample = fnames_sample[:n_samples]
    else:
        fnames_sample = fnames
    
    if isDCM:
        # Extract the current smallest size
        try:
            # Extract DCMs
            dcms = fnames_sample.map(dcmread)

            # Get the current smallest size
            resize = min(dcms.attrgot('scaled_px').map(lambda x: x.size()))
        except AttributeError:
            import pydicom
            from numpy import inf
            dcms = []
            resize = []

            # Extract different size and get the samllest one
            for fname in fnames_sample:
                dcm = fname.dcmread()
                resize.append(dcm.scaled_px.size())
                dcms.append(dcm)
            resize = min(resize)

        # Extract bins from scaled and resized samples
        samples = []
        for dcm in dcms:
            try:
                samples.append(torch.from_numpy(sk_resize(dcm_scale(dcm), resize)))
            except AttributeError:
                continue
    else:
        samples = []
        for fn in fnames_sample:
            image = Image.open(fn)
            samples.append(TF.to_tensor(image))
    
    # Calculate the frequency bins from these samples
    samples = torch.stack(tuple(samples))
    bins = samples.freqhist_bins()

    return bins