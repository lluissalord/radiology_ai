from torch.utils.data import Dataset, DataLoader
# from torchvision.utils import save_image
from fastai.vision.augment import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.medical.imaging import *
from skimage.transform import resize as sk_resize

import torchvision.transforms as tfms
import torchvision.transforms.functional as TF

from PIL import Image

import cv2

import gc
import random
import shutil
from tqdm import tqdm


def dcm_scale(dcm):
    """ Transform from raw pixel data to scaled one and inversing (if the case) """
    
    if dcm.PhotometricInterpretation == 'MONOCHROME1':
        return (dcm.scaled_px.max() - dcm.scaled_px) / (2**dcm.BitsStored - 1)
    else:
        return dcm.scaled_px / (2**dcm.BitsStored - 1)

def dcmread_scale(fn):
    """ Transform from path of raw pixel data to scaled one and inversing (if the case) """

    dcm = dcmread(fn)
    return dcm_scale(dcm)

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

# TODO: Review how to handle HistScaled for transforms on PNG
class HistScaled_all(Transform):
    def __init__(self, bins=None):
        super().__init__()
        self.bins = bins
    def encodes(self, sample:PILImage):
        return Image.fromarray(
            (
                sample._tensor_cls(sample) / 255.
            )
            .hist_scaled(brks=self.bins).numpy() * 255
        )


class CLAHE_Transform(Transform):
    
    def __init__(self, PIL_cls, clipLimit=2.0, tileGridSize=(8,8), grayscale=True, np_input=False, np_output=False):
        super().__init__()
        self.grayscale = grayscale
        self.np_input = np_input
        self.np_output = np_output

        self.PIL_cls = PIL_cls

        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    
    def encodes(self, sample:(PILImage,np.ndarray)):
        if self.np_input:
            img = sample
        else:
            if self.grayscale:
                img = np.array(sample.convert('L'))
            else:
                img = cv2.cvtColor(np.array(sample.convert('RGB')), cv2.COLOR_RGB2BGR)
            
        clahe_out = self.clahe.apply(img)
        
        if self.np_output:
            return clahe_out
        else:
            return self.PIL_cls.create(clahe_out)


class KneeLocalizer(Transform):
    """ Based on code from https://github.com/MIPT-Oulu/KneeLocalizer
    
    ```
    @inproceedings{tiulpin2017novel,
        title={A novel method for automatic localization of joint area on knee plain radiographs},
        author={Tiulpin, Aleksei and Thevenot, Jerome and Rahtu, Esa and Saarakkala, Simo},
        booktitle={Scandinavian Conference on Image Analysis},
        pages={290--301},
        year={2017},
        organization={Springer}
    }
    ```
    """
    def __init__(self, svm_model_path, PIL_cls, resize=None, np_input=False, np_output=False, debug=False):
        super().__init__()
        self.win_size = (64, 64)
        self.win_stride = (64, 64)
        self.block_size = (16, 16)
        self.block_stride = (8, 8)
        self.cell_size = (8, 8)
        self.padding = (0, 0)
        self.nbins = 9
        self.scales = [1.25, 1.5, 2, 2.5]

        self.svm_w, self.svm_b = np.load(svm_model_path, encoding='bytes', allow_pickle=True)
        
        self.resize = resize

        self.PIL_cls = PIL_cls

        self.debug = debug

        self.np_input = np_input
        self.np_output = np_output

    def encodes(self, x:(PILImage,np.ndarray)):

        if self.np_input:
            img = x
        else:
            img = ToTensor()(x).numpy()

        if len(img.shape) > 2:
          img = np.squeeze(img, 0)
        R, C = img.shape[-2:]

        # We will store the coordinates of the top left and
        # the bottom right corners of the bounding box
        hog = cv2.HOGDescriptor(self.win_size,
                                self.block_size,
                                self.block_stride,
                                self.cell_size,
                                self.nbins)

        # displacements = range(-C // 4, 1 * C // 4 + 1, C // 8)
        x_prop = self.get_joint_x_proposals(img)
        y_prop = self.get_joint_y_proposals(img)
        best_score = -np.inf

        for y_coord in y_prop:
            for x_coord in x_prop:
                for scale in self.scales:
                    # Check if fits on image
                    if x_coord - R / scale / 2 >= 0 and x_coord + R / scale / 2 <= C and y_coord - R / scale / 2 >= 0 and y_coord + R / scale / 2 <= R :
                        # Candidate ROI
                        roi = np.array([x_coord - R / scale / 2,
                                        y_coord - R / scale / 2,
                                        R / scale, R / scale], dtype=np.int)
                        x1, y1 = roi[0], roi[1]
                        x2, y2 = roi[0] + roi[2], roi[1] + roi[3]
                        patch = cv2.resize(img[y1:y2, x1:x2], self.win_size)

                        hog_descr = hog.compute(patch, self.win_stride, self.padding)
                        score = np.inner(self.svm_w, hog_descr.ravel()) + self.svm_b

                        if score > best_score:
                            best_score = score
                            roi_R = ((x1, y1), (x2,y2))

                            if self.debug:
                                print()
                                plt.imshow(patch, cmap=plt.cm.bone)
                                plt.title(f'{score:.2f}{(x1, y1), (x2,y2)}')
                                plt.show()

        if self.resize:
            img = cv2.resize(
                img[roi_R[0][1]:roi_R[1][1], roi_R[0][0]:roi_R[1][0]],
                dsize=(self.resize, self.resize)
            )
        else:
            img = img[roi_R[0][1]:roi_R[1][1], roi_R[0][0]:roi_R[1][0]]

        if self.np_output:
            return img
        else:
            value = self.PIL_cls.create(img)
            # value = Image.fromarray(
            #     img
            # )
            return value


    def smooth_line(self, line, av_points):
        smooth = np.convolve(
            line,
            np.ones((av_points, )) / av_points
        )[(av_points - 1):-(av_points - 1)]

        return smooth

    def get_joint_y_proposals(self, img, av_points=2, av_derv_points=11, margin=0.25, step=10):
        """Return Y-coordinates of the joint approximate locations."""

        R, C = img.shape[-2:]

        # Sum the middle if the leg is along the X-axis
        segm_line = np.sum(img[int(R * margin):int(R * (1 - margin)),
                            int(C / 3):int(C - C / 3)], axis=1)
    
        # Smooth the segmentation line
        segm_line = self.smooth_line(segm_line, av_points)

        # Find the absolute of the derivative smoothed
        derv_segm_line = np.abs(
            self.smooth_line(
                np.diff(
                    segm_line
                ),
                av_derv_points
            )
        )

        # Get top tau % of the peaks
        peaks = np.argsort(derv_segm_line)[::-1][:int(0.1 * R * (1 - 2 * margin))]
        
        return peaks[::step] + int(R * margin)

    
    def get_joint_x_proposals(self, img, av_points=2, av_derv_points=11, margin=0.25, step=10):
        """Return X-coordinates of the bone approximate locations"""

        R, C = img.shape[-2:]

        # Sum the middle if the leg is along the Y-axis
        segm_line = np.sum(img[int(R / 3):int(R - R / 3),
                               int(C * margin):int(C * (1 - margin))], axis=0)
    
        # Smooth the segmentation line
        segm_line = self.smooth_line(segm_line, av_points)

        # Get top tau % of the peaks
        peaks = np.argsort(segm_line)[::-1][:int(0.1 * C * (1 - 2 * margin))]
        # print(peaks)
        return peaks[::step] + int(C * margin)


class XRayPreprocess(Transform):

    def __init__(self, PIL_cls, cut_min=5., cut_max=99., only_non_zero=True, scale=True, np_input=False, np_output=False):
        self.cut_min = cut_min
        self.cut_max = cut_max
        self.only_non_zero = only_non_zero
        self.scale = scale

        self.PIL_cls = PIL_cls

        self.np_input = np_input
        self.np_output = np_output

    def encodes(self, x:(PILImage,np.ndarray)):
        """Preprocess the X-ray image using histogram clipping and global contrast normalization.
        Parameters
        ----------
        cut_min: int
            Lowest percentile which is used to cut the image histogram.
        cut_max: int
            Highest percentile.
        """
        if self.np_input:
            img = x
        else:
            img = ToTensor()(x).numpy()
        
        if len(img.shape) > 2:
          img = np.squeeze(img, 0)

        if self.only_non_zero:
            percentile_img = img[img != 0]

        lim1, lim2 = np.percentile(
            percentile_img if self.only_non_zero and len(percentile_img) > 0 else img,
            [self.cut_min, self.cut_max]
        )

        img[img < lim1] = lim1
        img[img > lim2] = lim2

        img -= int(lim1)

        if self.scale:
            img = img.astype(np.float32)
            if lim2:
                img /= lim2
            img *= 255

        if self.np_output:
            return img.astype('uint8')
        else:
            value = self.PIL_cls.create(img.astype('uint8'))
            # value = Image.fromarray(
            #     img.astype('uint8')
            # )
            return value


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