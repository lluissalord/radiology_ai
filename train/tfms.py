from fastai.data.all import IntToFloatTensor
from fastai.vision.learner import *
from fastai.vision.augment import *
from fastai.vision.core import PILImageBW, PILImage
from fastai.vision.data import *

from preprocessing.transforms import *

# from preprocessing.dicom import *
from preprocessing.misc import *


def get_item_tfms(run_params):

    item_tfms = []

    # if run_params['HIST_CLIPPING']:
    #     item_tfms.append(XRayPreprocess(PIL_cls=PILImageBW, cut_min=run_params['HIST_CLIPPING_CUT_MIN'], cut_max=run_params['HIST_CLIPPING_CUT_MAX'], np_input=len(item_tfms) > 0, np_output=True))

    if run_params["KNEE_LOCALIZER"]:
        item_tfms.append(
            KneeLocalizer(
                run_params["KNEE_SVM_MODEL_PATH"],
                PIL_cls=PILImageBW,
                resize=run_params["RESIZE"],
                np_input=len(item_tfms) > 0,
                np_output=True,
            )
        )
    else:
        item_tfms.append(
            Resize(
                run_params["RESIZE"], method=ResizeMethod.Pad, pad_mode=PadMode.Zeros
            )
        )

    if run_params["BACKGROUND_PREPROCESS"]:
        item_tfms.append(
            BackgroundPreprocess(
                PIL_cls=PILImageBW, np_input=len(item_tfms) > 0, np_output=True
            )
        )

    # item_tfms.append(RandomResizedCrop(RANDOM_RESIZE_CROP))

    # Histogram scaling DICOM on the fly

    if run_params["CLAHE_SCALED"]:
        item_tfms.append(
            CLAHE_Transform(
                PIL_cls=PILImageBW,
                grayscale=not run_params["SELF_SUPERVISED"],
                np_input=len(item_tfms) > 0,
                np_output=False,
            )
        )
    elif run_params["HIST_SCALED"]:
        if run_params["HIST_SCALED_SELF"]:
            bins = None
        else:
            # bins = init_bins(fnames=L(list(final_df['Original'].values)), n_samples=100)
            all_valid_raw_preprocess = pd.concat(
                [pd.Series(unlabel_all_df.index), label_df["Raw_preprocess"]]
            )
            bins = init_bins(
                fnames=L(list(all_valid_raw_preprocess.values)),
                n_samples=100,
                isDCM=False,
            )
        # item_tfms.append(HistScaled(bins))
        item_tfms.append(HistScaled_all(bins))

    return item_tfms


def get_batch_tfms(run_params):
    label_tfms = [
        IntToFloatTensor(div=2 ** 16 - 1),
        *aug_transforms(
            pad_mode=PadMode.Zeros,
            mult=1.0,
            do_flip=True,
            flip_vert=False,
            max_rotate=90.0,
            min_zoom=0.9,
            max_zoom=1.2,
            max_lighting=0.4,
            max_warp=0.4,
            p_affine=0.9,
            p_lighting=0.9,
            mode="bilinear",
            align_corners=True,
        ),
        RandomResizedCropGPU(
            run_params["RANDOM_RESIZE_CROP"], min_scale=run_params["RANDOM_MIN_SCALE"]
        ),
        # Normalize() # Issue with CPU vs GPU interaction
    ]

    unlabel_tfms = [[IntToFloatTensor(div=2 ** 16 - 1)]]
    if run_params["SSL"] == run_params["SSL_FIX_MATCH"]:

        weak_transform = [
            IntToFloatTensor(div=1),
            RandomResizedCropGPU(
                run_params["RANDOM_RESIZE_CROP"],
                min_scale=run_params["RANDOM_MIN_SCALE"],
            ),
            Flip(),
            # Normalize()
        ]
        unlabel_tfms.append(weak_transform)

        strong_transform = [
            IntToFloatTensor(div=1),
            RandomResizedCropGPU(
                run_params["RANDOM_RESIZE_CROP"],
                min_scale=run_params["RANDOM_MIN_SCALE"],
            ),
            Flip(),
            Rotate(180),
            Brightness(),
            Contrast(),
            RandomErasing(),
            # Normalize()
        ]
        unlabel_tfms.append(strong_transform)
    elif run_params["SSL"] == run_params["SSL_MIX_MATCH"]:

        unlabel_transform = [
            IntToFloatTensor(div=2 ** 16 - 1),
            RandomResizedCropGPU(
                run_params["RANDOM_RESIZE_CROP"],
                min_scale=run_params["RANDOM_MIN_SCALE"],
            ),
            Flip(),
            Rotate(180),
            Brightness(),
            Contrast(),
            # Normalize()
        ]
        unlabel_tfms.append(unlabel_transform)

    return label_tfms, unlabel_tfms
