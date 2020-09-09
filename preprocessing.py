from torch.utils.data import Dataset, DataLoader
# from torchvision.utils import save_image
from fastai.vision.augment import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.medical.imaging import *
from skimage.transform import resize as sk_resize

import torchvision.transforms as tfms

import random

class DCMPreprocessDataset(Dataset):
    """ Dataset class for DCM preprocessing """

    def __init__(self, fnames, resize=None, padding_to_square=True, bins=None):
        self.fnames = fnames
        self.resize = resize
        self.padding_to_square = padding_to_square
        self.bins = bins

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self,  idx):
        """ Get sample after reading from DCM file, scale it and applying the transformations """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dcm = self.fnames[idx].dcmread()
        sample = dcm.scaled_px
        if self.bins is not None:
            sample = sample.hist_scaled(brks=self.bins)

        if self.padding_to_square or self.resize is not None:
            transform = [tfms.ToPILImage()]
            # sample = PILImage.create(sample)
            # transform = []

            if self.padding_to_square:
                # Prepare padding transformation to make square image
                x, y = sample.size()
                pad_to = max(x, y)
                extra_width = 0 if (pad_to - y) % 2 else 1
                extra_height = 0 if (pad_to - x) % 2 else 1
                padding = ((pad_to - y) // 2, (pad_to - x) // 2, (pad_to - y) // 2 + extra_width, (pad_to - x) // 2 + extra_height)

                transform += [tfms.Pad(padding=padding, fill=0, padding_mode='constant')]
                # transform += [CropPad(size=pad_to)] # Pending release of issue https://github.com/pytorch/vision/pull/2515

            if self.resize is not None:
                # transform += [Resize(size=self.resize)]
                transform += [tfms.Resize(size=self.resize)]

            sample = tfms.Compose(transform)(sample)
            # for tfms in transform:
            #     sample = tfms(sample)

        return sample

    def get_fname(self, idx):
        """ Get the filenames """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.fnames[idx].name


    def init_bins(self, n_samples=None):
        """ Initialize bins to equally distribute the histogram of the dataset """

        # Select randomly n_samples
        if n_samples is None:
            fnames_sample = self.fnames.copy()
            random.shuffle(fnames_sample)
            fnames_sample = fnames_sample[:n_samples]
        else:
            fnames_sample = self.fnames
        
        # Extract DCMs
        dcms = fnames_sample.map(dcmread)

        # Resize all images to the same size as the smallest one
        resize = min(dcms.attrgot('scaled_px').map(lambda x: x.size()))
        
        # Extract bins from scaled and resized samples
        samples = torch.stack(tuple([torch.from_numpy(sk_resize(dcm.scaled_px, resize)) for dcm in dcms]))
        self.bins = samples.freqhist_bins()

        return self.bins

    def save(self, dst_folder, extension='png'):
        for idx, data in enumerate(self):
            # Create the destination folder if not exists
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)

            filename, _ = os.path.splitext(self.get_fname(idx))
            data.save(f'{dst_folder}/{filename}.{extension}', format=extension, compress_level=0 if extension.lower()=='png' else None)
            # save_image(data, f'{dst_folder}/{filename}.{extension}')