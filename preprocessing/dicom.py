""" Preprocessings used for DICOM treatments """

from tqdm import tqdm
import gc
import os
import shutil

import pydicom
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as tfms
from fastcore.transform import Transform
from fastai.data.all import Path
from fastai.medical.imaging import PILDicom

from preprocessing.misc import *


class PILDicom_scaled(PILDicom):
    """ Generalization of PILDicom class making sure it is properly scaled taking into account DICOM metadata """

    @classmethod
    def create(cls, fn:(Path,str,bytes), mode=None)->None:
        "Open a `DICOM file` from path `fn` or bytes `fn` and load it as a `PIL Image`"
        if isinstance(fn,bytes): im = Image.fromarray(pydicom.dcmread(pydicom.filebase.DicomBytesIO(fn)).pixel_array)
        if isinstance(fn,(Path,str)): im = Image.fromarray(255 * dcmread_scale(fn).numpy())
        im.load()
        im = im._new(im.im)
        return cls(im.convert(mode) if mode else im)


class HistScaled_Dicom(Transform):
    """ Transformation of Histogram Scaling compatible with DataLoaders, allowing Histogram Scaling on the fly """

    def __init__(self, bins=None):
        super().__init__()
        self.bins = bins

    def encodes(self, sample:PILDicom_scaled):
        return Image.fromarray(
            (
                sample._tensor_cls(sample) / 255.
            )
            .hist_scaled(brks=self.bins).numpy() * 255
        )


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

        dcm = self.get_dcm(idx)
        
        # Treat sample even if cannot transform DCM because of missing PresentationLUTShape (so it is not a X-ray image)
        sample = dcm_scale(dcm)
    
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

    def get_dcm(self, idx):
        """ Get the DCM file """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.fnames[idx].dcmread()

    def get_dcm_by_name(self, img_name):
        if img_name[-4:] != '.dcm':
            img_name = img_name + '.dcm'

        return self.fnames.map(lambda x: x if x.name == img_name else None).filter(None)[0].dcmread()

    def init_bins(self, n_samples=None):
        """ Initialize bins to equally distribute the histogram of the dataset """

        self.bins = init_bins(self.fnames, n_samples)
        return self.bins

    def save(self, dst_folder, extension='png', overwrite=True, keep=False, clean_folder=False):
        """ Saves all preprocessed files converting them into image files """
        
        # Do not make sense overwrite=True with keep=True
        if overwrite and keep:
            raise ValueError('Incoherent setting of overwrite=True with keep=True')

        # Create the destination folder if not exists
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        elif clean_folder:
            shutil.rmtree(dst_folder)
            os.makedirs(dst_folder)
            
        for idx in tqdm(range(len(self)), desc='Saving Images: '):
            # Define filename and filepath
            filename, ext = os.path.splitext(self.get_fname(idx))

            # Take into account the ones which seems to have extension ".PACSXXX" is not an extension
            if ext.startswith('.PACS'):
                filename = self.get_fname(idx)

            filepath = f'{dst_folder}/{filename}.{extension}'

            # In case the file already exists and it has to be keep, then continue with the loop
            if os.path.exists(filepath) and keep:
                continue
            
            data = self[idx]

            # Add numeration if files already exist and do not want to overwrite
            i = 1
            while not overwrite and os.path.exists(filepath):
                filepath = f'{dst_folder}/{filename}_{i}.{extension}'
                i += 1

            data.save(filepath, format=extension, compress_level=0 if extension.lower()=='png' else None)
            # save_image(data, f'{dst_folder}/{filename}.{extension}')

            del data
            gc.collect()