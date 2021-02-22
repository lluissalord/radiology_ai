""" Fastai Transformations to be used on training """

import numpy as np

from fastai.vision.core import *
from fastcore.transform import Transform

from PIL import Image
import matplotlib.pyplot as plt

import cv2

import torch
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
        
        if not self.grayscale:
            clahe_out = cv2.cvtColor(clahe_out, cv2.COLOR_BGR2RGB)

        if self.np_output:
            return clahe_out
        else:
            return self.PIL_cls.create(clahe_out)


def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)


def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h)


def check_a_in_b(a, b):
    return a[2]*a[3] != b[2]*b[3] and a[0] >= b[0] and a[1] >= b[1] and a[0]+a[2] <= b[0]+b[2] and a[1]+a[3] <= b[1]+b[3]


def check_intersection(a, b):
    inter_box = intersection(a,b)
    return inter_box != () and inter_box[2] != 0 and inter_box[3] != 0


def add_in_dict_list(d, key, value):
    if key in d:
        d[key].append(value)
    else:
        d[key] = [value]


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
        self.scales = [1.05, 1.25, 1.5, 2]
        # self.scales = [0.9, 1, 1.1]

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

        # At least score should be better than this
        best_score = -np.inf

        # Approximate the scales to not include left/right or top/bottom black stripes
        # First check if there is top/bottom black stripes, otherwise check it for left/right ones
        mask_black = cv2.threshold(img, 7, 255, cv2.THRESH_BINARY)[1]
        
        # apply morphology to remove isolated extraneous noise
        kernel = np.ones((5,5), np.uint8)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)

        # Find contours of the intersting part, which will be the biggest zone 
        cnts = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        biggest_area = 0
        total_area = (float(mask_black.shape[0]) * mask_black.shape[1])
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            area = w * h / total_area
            if area > biggest_area:
                biggest_area = area
                roi = np.array([x,y,w,h], dtype=np.int)
                if self.debug:
                    cv2.rectangle(mask_black, (x,y),(x+w,y+h), (0,255,0), 2)
        if biggest_area != 0:
            x1, y1 = roi[0], roi[1]
            x2, y2 = roi[0] + roi[2], roi[1] + roi[3]
            roi_img = img[y1:y2, x1:x2]
        else:
            roi_img = img[:]

        # Check for external white zones and remove them
        # Generate a mask where background is identified
        mask_white = cv2.threshold(roi_img, 50, 255, cv2.THRESH_BINARY_INV)[1]
        # mask_white = 255 - mask_white

        # apply morphology to remove isolated extraneous noise
        kernel = np.ones((5,5), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

        # Find the minimum contour (or two combined) that contain the majority of background and ensure the remaining is only external noise
        # Check that with two conditions:
        # 1. Area of bounding box represents at least 50% of the image
        # 2. The remaining image, after painting this bounding box as black, should be almost nothing
        cnts = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        smallest_area = np.inf
        total_area = (float(mask_white.shape[0]) * mask_white.shape[1])
        found = False
        c_candidates = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            area = w * h / total_area
            if area > 0.05:
                
                # Add candidates to check union of contours
                c_candidates.append(c)

                # Area conditions
                if smallest_area <= area or area < 0.5:
                    continue

                # Check that when filling current proposal then almost nothing is left
                check_mask = mask_white[:].copy()
                cv2.drawContours(check_mask, [c], -1, (0,0,0), -1)
                # apply morphology to remove isolated extraneous noise
                kernel = np.ones((5,5), np.uint8)
                check_mask = cv2.morphologyEx(check_mask, cv2.MORPH_OPEN, kernel)
                check_mask = cv2.morphologyEx(check_mask, cv2.MORPH_CLOSE, kernel)
                if check_mask.sum() / (total_area * 255) < 0.01:
                    found = True
                    smallest_area = area
                    roi = np.array([x,y,w,h], dtype=np.int)
                
                if self.debug:
                    cv2.rectangle(mask_white, (x,y),(x+w,y+h), (0,255,0), 2)

        # Check candidates for combined contours
        if len(c_candidates) > 1:
            for i, c1 in enumerate(c_candidates):
                bbox_1 = cv2.boundingRect(c1)
                for j, c2 in enumerate(c_candidates):

                    # Do not try on itself and already checked
                    if i >= j:
                        continue
                
                    bbox_2 = cv2.boundingRect(c2)

                    # Cannot be on contours which overlap
                    if check_intersection(bbox_1, bbox_2):
                        continue

                    # Area of the union should fullfil area conditions
                    uni_box = union(bbox_1, bbox_2)
                    area = uni_box[2] * uni_box[3] / total_area
                    if smallest_area <= area or area < 0.5:
                        continue

                    # Check that when filling current proposal then almost nothing is left
                    check_mask = mask_white[:].copy()
                    cv2.drawContours(check_mask, [c1, c2], -1, (0,0,0), -1)
                    # apply morphology to remove isolated extraneous noise
                    kernel = np.ones((5,5), np.uint8)
                    check_mask = cv2.morphologyEx(check_mask, cv2.MORPH_OPEN, kernel)
                    check_mask = cv2.morphologyEx(check_mask, cv2.MORPH_CLOSE, kernel)
                    if check_mask.sum() / (total_area * 255) < 0.01:
                        found = True
                        smallest_area = area
                        roi = np.array(uni_box, dtype=np.int)

                    if self.debug:
                        x,y,w,h = tuple(uni_box)
                        cv2.rectangle(mask_white, (x,y),(x+w,y+h), (0,255,0), 2)

        # Set current ROI
        if found:
            x1, y1 = roi[0], roi[1]
            x2, y2 = roi[0] + roi[2], roi[1] + roi[3]
            roi_img = roi_img[y1:y2, x1:x2]
        else:
            roi_img = roi_img[:]

        
        R, C = roi_img.shape[-2:]
        min_R_C = min(R,C)

        # We will store the coordinates of the top left and
        # the bottom right corners of the bounding box
        hog = cv2.HOGDescriptor(self.win_size,
                                self.block_size,
                                self.block_stride,
                                self.cell_size,
                                self.nbins)

        # displacements = range(-C // 4, 1 * C // 4 + 1, C // 8)
        x_prop = self.get_joint_x_proposals(roi_img)
        y_prop = self.get_joint_y_proposals(roi_img)

        if self.debug:
            debug_info = []

        if y_prop != []:
            # Loop across proposals and scales
            for y_coord in y_prop:
                for x_coord in x_prop:
                    for scale in self.scales:
                        # Check if fits on image
                        if x_coord - min_R_C / scale / 2 >= 0 and x_coord + min_R_C / scale / 2 <= C and y_coord - min_R_C / scale / 2 >= 0 and y_coord + min_R_C / scale / 2 <= R :

                            # Candidate ROI
                            roi = np.array([x_coord - min_R_C / scale / 2,
                                            y_coord - min_R_C / scale / 2,
                                            min_R_C / scale, min_R_C / scale], dtype=np.int)
                            x1, y1 = roi[0], roi[1]
                            x2, y2 = roi[0] + roi[2], roi[1] + roi[3]
                            patch = cv2.resize(roi_img[y1:y2, x1:x2], self.win_size)

                            # Calculate score from SVM model
                            hog_descr = hog.compute(patch, self.win_stride, self.padding)
                            score = np.inner(self.svm_w, hog_descr.ravel()) + self.svm_b

                            if self.debug:
                                debug_info.append(
                                    {
                                        'patch': patch,
                                        'score': score,
                                        'scale': scale,
                                        'coords': ((x1, y1), (x2,y2)),
                                    }
                                )
                            
                            # Check and save best score
                            if score > best_score:
                                best_score = score
                                roi_R = ((x1, y1), (x2,y2))
        else:
            roi_img = img[:]
            # self.PIL_cls.create(roi_img).save('sources/extra_images/failing.png', format='png', compress_level=0)
            debug_info = []

        # if self.debug or y_prop == []:
        if self.debug:
            print()
            cols = 4
            rows = len(debug_info) // cols + (len(debug_info) % cols != 0) + 1
            fig, axs = plt.subplots(
                rows,
                cols, 
                figsize=(15,15)
            )
            ax_orig = axs[0] if rows == 1 else axs[0,0]
            ax_orig.imshow(img, cmap=plt.cm.bone)
            ax_orig.set_title(f'Original')
            ax_roi = axs[1] if rows == 1 else axs[0,1]
            ax_roi.imshow(roi_img, cmap=plt.cm.bone)
            ax_roi.set_title(f'ROI from Original')
            ax_mask_black = axs[2] if rows == 1 else axs[0,2]
            ax_mask_black.imshow(mask_black, cmap=plt.cm.bone)
            ax_mask_black.set_title(f'Mask')
            ax_mask_white = axs[3] if rows == 1 else axs[0,3]
            ax_mask_white.imshow(mask_white, cmap=plt.cm.bone)
            ax_mask_white.set_title(f'Mask')
            if len(debug_info) > 0:
                debug_info = sorted(debug_info, key=lambda x: x["score"], reverse=True)
                i = 0
                for row_axs in axs[1:]:
                    for ax in row_axs:
                        if i >= len(debug_info):
                            break
                        info = debug_info[i]
                        ax.imshow(info["patch"], cmap=plt.cm.bone)
                        ax.set_title(f'{info["score"]:.2f}-{info["scale"]:.2f}')
                        i += 1
            fig.tight_layout()
            fig.show()
            plt.show()

        if best_score == -np.inf:
            img = cv2.resize(
                roi_img,
                dsize=(self.resize, self.resize)
            )
        elif self.resize:
            img = cv2.resize(
                roi_img[roi_R[0][1]:roi_R[1][1], roi_R[0][0]:roi_R[1][0]],
                dsize=(self.resize, self.resize)
            )
        else:
            img = roi_img[roi_R[0][1]:roi_R[1][1], roi_R[0][0]:roi_R[1][0]]

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

        # Find the derivative smoothed
        derv_segm_line = self.smooth_line(
                np.diff(
                    segm_line
                ),
                av_derv_points
            )

        # Find the absolute of the second derivative smoothed
        derv_derv_segm_line = np.abs(
            self.smooth_line(
                np.diff(
                    derv_segm_line
                ),
                av_derv_points
            )
        )

        # Filter for peaks with highest second derivate as it has to be a quick change of gradients
        derv_peaks = np.argsort(derv_derv_segm_line)[::-1][:int(0.1 * R * (1 - 2 * margin))]

        if len(derv_peaks) != 0:
            max_cond = min(max(derv_peaks) + R // 20, len(derv_segm_line))
            min_cond = max(min(derv_peaks) - R // 20, 0)

            # Get top tau % of the filtered peaks
            peaks = np.argsort(np.abs(derv_segm_line[min_cond:max_cond]))[::-1][:int(0.1 * R * (1 - 2 * margin))] + min_cond

            # Filter to only use peaks which are separated at least by a 5%
            final_peaks = []
            while len(peaks) != 0:
                peak = peaks[0]
                final_peaks.append(peak)
                peaks = peaks[(peaks > peak + R // 25) | (peaks < peak - R // 25)]

            # return list(peaks[::step] + int(R * margin)) + [R // 2]
            return list(np.array(final_peaks) + int(R * margin)) + [R // 2]

        # Sometimes if there has been some issues on previous steps it happens that there are no enough derv_peaks
        # then it is best to not propose any
        else:
            return []

    
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
        
        # Filter to only use peaks which are separated at least by a 5%
        final_peaks = []
        while len(peaks) != 0:
            peak = peaks[0]
            final_peaks.append(peak)
            peaks = peaks[(peaks > peak + C // 20) | (peaks < peak - C // 20)]

        # return list(peaks[::step] + int(C * margin)) + [C // 2]
        return list(np.array(final_peaks) + int(C * margin)) + [C // 2]


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
    1. Adaptativ binary thresholding and extract Otsu threshold value for later use on contour conditions
    3. Find main contours of the background parts to be removed and create a mask with them
    4. Look for contours inside other contours on mask and paint them with the color of the parent contour
    5. Repeat previous step with inverse resulting mask (taking into account also previous parent contours)
    6. Bitwise input image with mask to remove background"""

    def __init__(self, PIL_cls, np_input=False, np_output=False, morph_kernel=15, inpaint_morph_kernel=15, thresh_limit_multiplier=0.6, min_thresh=120, min_density=0.2, min_area=0.02, max_area=0.5, debug=False):
        super().__init__()
        self.PIL_cls = PIL_cls

        self.morph_kernel = morph_kernel
        self.inpaint_morph_kernel = inpaint_morph_kernel

        # self.center_width_scale = center_width_scale # 1/3

        self.thresh_limit_multiplier = thresh_limit_multiplier
        self.min_thresh = min_thresh
        self.min_density = min_density

        self.min_area = min_area
        self.max_area = max_area

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
        # R, C = img.shape[-2:]

        # threshold input image as mask with adaptative threshold
        mask = cv2.adaptiveThreshold(
            img,
            255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # Extract which threshold would be used by Otsu to be used as condition later
        ret, _ = cv2.threshold(
            img,
            0, 255, cv2.THRESH_OTSU)

        # Copy images to be able to compare on debugging 
        if self.debug:
            result = img[:].copy()
            boxes = img[:].copy()
        else:
            result = img

        # # apply morphology to remove isolated extraneous noise
        kernel = np.ones((self.morph_kernel,self.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Create mask where to paint on the parts that will be removed
        mask_inpaint = np.ones_like(mask) * 255

        # Find main contours of the background parts which will be removed
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        biggest_cs = []
        total_area = (float(mask.shape[0]) * mask.shape[1])
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            ar = w / float(h)
            # area = w * h / total_area
            area = cv2.contourArea(c)/total_area

            # center_width = x + w // 2
            # left_limit = mask.shape[1] // 2 - int(self.center_width_scale * mask.shape[1] / 2)
            # right_limit = mask.shape[1] // 2 + int(self.center_width_scale * mask.shape[1] / 2)
            # if center_width >= left_limit and center_width <= right_limit:

            # Check are is between limits
            if area > self.min_area and area < self.max_area:
                
                temp = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(temp,[c],0,255,-1)
                match = np.where(temp == 255)
                mask_matched = mask[match].flatten()
                img_matched = img[match].flatten()

                density = mask_matched.sum() / (255.0 * len(mask_matched))
                # if self.debug:
                #     rows = 1
                #     cols = 2
                #     fig, (ax1, ax2) = plt.subplots(rows,cols, figsize=(5*cols, 5*rows))
                #     ax1.imshow(temp, cmap=plt.cm.bone)
                #     ax2.imshow(mask[y:y+h,x:x+w], cmap=plt.cm.bone)
                #     fig.suptitle(f'{area, x,y,w,h, density}')
                #     fig.tight_layout()
                #     fig.show()

                median = np.median(img_matched)
                # Check density and median of countour area in input image is below threshold limits
                if density > self.min_density and median < self.min_thresh and median < ret * self.thresh_limit_multiplier:
                    biggest_cs.append(c)

        # Only modify image if there is any area that matched, otherwise do not modify
        if len(biggest_cs) != 0:
            cv2.drawContours(mask_inpaint, biggest_cs, -1, (0,0,0), -1)
            
            if self.debug:
                for c in biggest_cs:
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(boxes, (x,y),(x+w,y+h), (0,255,0), 3)

            # Proceed with processes having background parts in white
            mask_inpaint = 255 - mask_inpaint

            # # apply morphology to remove isolated extraneous noise
            kernel = np.ones((self.inpaint_morph_kernel,self.inpaint_morph_kernel), np.uint8)
            mask_inpaint = cv2.morphologyEx(mask_inpaint, cv2.MORPH_OPEN, kernel)
            mask_inpaint = cv2.morphologyEx(mask_inpaint, cv2.MORPH_CLOSE, kernel)

            # Look for contours inside other contours, save them with the color of the parent contour
            cnts, hier = cv2.findContours(mask_inpaint, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)            
            fill_cnts = {}
            parent_bbox = {}
            for i, h in enumerate(hier[0]):
                # Internal contours are saved on fill_cnts to be painted on
                if h[-1] != -1:
                    # Determine parent color to fill child contour
                    x,y,w,h = cv2.boundingRect(cnts[h[-1]])
                    parent_color = np.median(mask_inpaint[y:y+h,x:x+w])

                    add_in_dict_list(fill_cnts, parent_color, cnts[i])
                # External contours are saved on parent_bbox to check later if are parents of other contours
                else:
                    # Determine parent color to fill child contour
                    x,y,w,h = cv2.boundingRect(cnts[i])
                    parent_color = 255 - np.median(mask_inpaint[y:y+h,x:x+w])

                    add_in_dict_list(parent_bbox, parent_color, (x,y,w,h))

            # Paint internal contours with its parent color
            for parent_color, cnts_list in fill_cnts.items():
                cv2.drawContours(mask_inpaint, cnts_list, -1, (parent_color,parent_color,parent_color), -1)

            # Proceed with processes having bone parts in white
            mask_inpaint = 255 - mask_inpaint

            # Look for contours inside other contours, save them with the color of the parent contour
            cnts, hier = cv2.findContours(mask_inpaint, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)            
            fill_cnts = {}
            for i, h in enumerate(hier[0]):
                bbox = cv2.boundingRect(cnts[i])
                # Internal contours are saved on fill_cnts to be painted on
                if h[-1] != -1:
                    # Determine parent color to fill child contour
                    x,y,w,h = cv2.boundingRect(cnts[h[-1]])
                    parent_color = np.median(mask_inpaint[y:y+h,x:x+w])

                    add_in_dict_list(fill_cnts, parent_color, cnts[i])
                # External contours are checked if are inside a previous parent bounding box, if so added to be painted on
                else:
                    for parent_color, p_bbox_list in parent_bbox.items():
                        for p_bbox in p_bbox_list:
                            if check_a_in_b(bbox, p_bbox):
                                add_in_dict_list(fill_cnts, parent_color, cnts[i])

            # Paint internal contours with its parent color
            for parent_color, cnts_list in fill_cnts.items():
                cv2.drawContours(mask_inpaint, cnts_list, -1, (parent_color,parent_color,parent_color), -1)

            # Apply mask on input image which results on image without background
            result = cv2.bitwise_and(result, result, mask = mask_inpaint)
        
        if self.debug:
            print()
            rows = 2
            cols = 2
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(rows,cols, figsize=(5*cols, 5*rows))
            ax1.imshow(img, cmap=plt.cm.bone)
            ax2.imshow(result, cmap=plt.cm.bone)
            ax3.imshow(mask, cmap=plt.cm.bone)
            ax4.imshow(mask_inpaint)
            fig.suptitle(f'area {(self.min_area, self.max_area)} - th_limit ({self.thresh_limit_multiplier:2f}, {ret * self.thresh_limit_multiplier:.0f}) - kernels (M,IM) ({self.morph_kernel},{self.inpaint_morph_kernel})')
            fig.tight_layout()
            fig.show()

        if self.np_output:
            return result
        else:
            value = self.PIL_cls.create(result)
            # value = Image.fromarray(
            #     img
            # )
            return value