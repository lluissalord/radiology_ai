""" Fastai Transformations to be used on training """

import numpy as np

from fastai.vision.core import *
from fastcore.transform import Transform

from PIL import Image

import cv2


class HistScaled(Transform):
    """ Transformation of Histogram Scaling compatible with DataLoaders, allowing Histogram Scaling on the fly """

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
    """ Implement CLAHE transformation for Adaptative Histogram Equalization """
    
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
    """ Transformation which finds out where is the knee and cut out the rest of the image
    Based on code from https://github.com/MIPT-Oulu/KneeLocalizer
    
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
    """ Preprocess the X-ray image using histogram clipping and global contrast normalization. """

    def __init__(self, PIL_cls, cut_min=5., cut_max=99., only_non_zero=True, scale=True, np_input=False, np_output=False):
        self.cut_min = cut_min
        self.cut_max = cut_max
        self.only_non_zero = only_non_zero
        self.scale = scale

        self.PIL_cls = PIL_cls

        self.np_input = np_input
        self.np_output = np_output

    def encodes(self, x:(PILImage,np.ndarray)):

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
