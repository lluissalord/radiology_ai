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


class BackgroundPreprocess(Transform):
    """ Background cleaning preprocess. Consist on the following steps:
    1. OTSU binary thresholding from a Blur image
    2. Apply morphology to remove isolated extraneous noise on mask
    3. Dilate mask to be confidently not removing any interesting part
    4. Find contours of the intersting part, which will be the centered biggest zone
    5. Apply Hull convex to be more confident to not remove any interesting part
    6. Bitwise input image with mask to only mantain the masked zone"""

    def __init__(self, PIL_cls, np_input=False, np_output=False, premask_blur_kernel=5, morph_kernel=15, dilate_kernel=5, blur_kernel=7, center_width_scale=1/3, inpaint_radius=3, debug=False):
        super().__init__()
        self.PIL_cls = PIL_cls

        self.premask_blur_kernel = premask_blur_kernel 
        self.morph_kernel = morph_kernel
        self.dilate_kernel = dilate_kernel
        self.blur_kernel = blur_kernel

        self.center_width_scale = center_width_scale

        self.inpaint_radius = inpaint_radius

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

        # threshold input image as mask
        mask = cv2.threshold(
            cv2.GaussianBlur(img, (self.premask_blur_kernel, self.premask_blur_kernel), 0)
            , 0, 255, cv2.THRESH_OTSU)[1]
        # mask_black = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY_INV)[1]

        result = copy(img[:])
        if self.debug:
            boxes = copy(img[:])

        # apply morphology to remove isolated extraneous noise
        kernel = np.ones((self.morph_kernel,self.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Dilate mask to be confidently not removing any interesting part
        kernel = np.ones((self.dilate_kernel,self.dilate_kernel), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=5)

        mask_inpaint = np.ones_like(mask) * 255

        # Find contours of the intersting part, which will be the biggest zone 
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        biggest_c = []
        biggest_area = 0
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            ar = w / float(h)
            area = w * h / (float(mask.shape[0]) * mask.shape[1])
            center_width = x + w // 2
            left_limit = mask.shape[1] // 2 - int(self.center_width_scale * mask.shape[1] / 2)
            right_limit = mask.shape[1] // 2 + int(self.center_width_scale * mask.shape[1] / 2)
            if area > biggest_area and center_width >= left_limit and center_width <= right_limit:
                biggest_c = c
                biggest_area = area

        # Only modify image if there is an area that matched, otherwise do not modify
        if len(biggest_c) != 0:
            # cv2.drawContours(mask_inpaint, [biggest_c], -1, (0,0,0), -1)

            hull = cv2.convexHull(biggest_c)
            cv2.drawContours(mask_inpaint, [hull], -1, (0,0,0), -1)
            
            if self.debug:
                # x,y,w,h = cv2.boundingRect(biggest_c)
                x,y,w,h = cv2.boundingRect(hull)
                cv2.rectangle(boxes, (x,y),(x+w,y+h), (0,255,0), 1)

            # Blur mask to soft edges
            mask_inpaint = cv2.GaussianBlur(mask_inpaint, (self.blur_kernel, self.blur_kernel), 0)

            # Only masked only background which is not already black
            mask_inpaint = mask_inpaint# - mask_black

            # Inpaint background to get a soft background
            # result = cv2.inpaint(result, mask_inpaint, self.inpaint_radius, cv2.INPAINT_TELEA)
            result = cv2.bitwise_and(result, result, mask = 255 - mask_inpaint)
        
        if self.debug:
            print()
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15, 15))
            ax1.imshow(img, cmap=plt.cm.bone)
            ax2.imshow(result, cmap=plt.cm.bone)
            ax3.imshow(boxes, cmap=plt.cm.bone)
            ax4.imshow(mask_inpaint)
            fig.suptitle(f'kernels (P,M,D,B) ({self.premask_blur_kernel},{self.morph_kernel},{self.dilate_kernel},{self.blur_kernel}) - CWS {self.center_width_scale:.2f} - Inpaint {self.inpaint_radius}')
            fig.show()

        if self.np_output:
            return result
        else:
            value = self.PIL_cls.create(result)
            # value = Image.fromarray(
            #     img
            # )
            return value