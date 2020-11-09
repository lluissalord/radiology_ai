from torch.utils.data import Dataset, DataLoader
# from torchvision.utils import save_image
from fastai.vision.augment import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.medical.imaging import *
from skimage.transform import resize as sk_resize

import torchvision.transforms as tfms

from PIL import Image

import random
import shutil
from tqdm import tqdm


def dcm_scale(dcm):
    """ Transform from raw pixel data to scaled one and inversing (if the case) """
    
    if dcm.PresentationLUTShape == 'INVERSE':
        img = (dcm.scaled_px.max() - dcm.scaled_px)
    else:
        img = dcm.scaled_px
    if dcm.pixel_array.dtype.name == 'uint16':
        return img / (2**12 - 1)
    else:
        return img

def dcmread_scale(fn):
    """ Transform from path of raw pixel data to scaled one and inversing (if the case) """

    dcm = dcmread(fn)
    return dcm_scale(dcm)

def init_bins(fnames, n_samples=None):
        """ Initialize bins to equally distribute the histogram of the dataset """

        # Select randomly n_samples
        if n_samples is not None:
            fnames_sample = fnames.copy()
            random.shuffle(fnames_sample)
            fnames_sample = fnames_sample[:n_samples]
        else:
            fnames_sample = fnames
        
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
        samples = torch.stack(tuple(samples))
        
        # Calculate the frequency bins from these samples
        bins = samples.freqhist_bins()

        return bins

class PILDicom_scaled(PILDicom):
    @classmethod
    def create(cls, fn:(Path,str,bytes), mode=None)->None:
        "Open a `DICOM file` from path `fn` or bytes `fn` and load it as a `PIL Image`"
        if isinstance(fn,bytes): im = Image.fromarray(pydicom.dcmread(pydicom.filebase.DicomBytesIO(fn)).pixel_array)
        if isinstance(fn,(Path,str)): im = Image.fromarray(255 * dcmread_scale(fn).numpy())
        im.load()
        im = im._new(im.im)
        return cls(im.convert(mode) if mode else im)

class HistScaled(Transform):
    def __init__(self, bins):
        super().__init__()
        self.bins = bins
    def encodes(self, sample:PILDicom_scaled):
        return Image.fromarray(
            (
                sample._tensor_cls(sample) / 255
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
        try:
            sample = dcm_scale(dcm)
        except AttributeError:
            if dcm.pixel_array.dtype.name == 'uint16':
                sample = dcm.scaled_px / (2**12 - 1)
            else:
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
            filename, _ = os.path.splitext(self.get_fname(idx))
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