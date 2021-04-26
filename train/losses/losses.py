import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import torch

from fastai.losses import CrossEntropyLossFlat


def get_class_weights(classes, y, run_params):
    n_out = len(classes)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights /= class_weights.sum()

    # Correct the class weights in case of using AllLabelsInBatchDL
    class_weights = correct_weights(class_weights, n_out, run_params)

    class_weights = torch.as_tensor(class_weights).float()
    if torch.cuda.is_available():
        class_weights = class_weights.cuda()

    return class_weights


def correct_weights(class_weights, n_out, run_params):
    # Correct the class weights in case of using AllLabelsInBatchDL
    if run_params["ALL_LABELS_IN_BATCH"]:
        coef = run_params["MIN_SAMPLES_PER_LABEL"] * n_out / run_params["BATCH_SIZE"]
        class_weights *= 1 - coef
        class_weights += np.ones_like(class_weights) * coef / len(class_weights)

    return class_weights


def define_losses(run_params, loss_params, train_df, unlabel_dls):
    if run_params["SSL"] == run_params["SSL_FIX_MATCH"]:
        from semisupervised.fixmatch.losses import FixMatchLoss as SSLLoss
    elif run_params["SSL"] == run_params["SSL_MIX_MATCH"]:
        from semisupervised.mixmatch.losses import MixMatchLoss as SSLLoss
    from train.losses.APL_losses import FocalLossFlat

    classes = train_df["Target"].unique()
    n_out = len(classes)

    if run_params["CLASS_WEIGHT"]:
        class_weights = get_class_weights(
            classes, y=train_df["Target"], run_params=run_params
        )
    else:
        class_weights = None

    if run_params["SSL"]:
        if loss_params["use_SCL"]:
            loss_params["frequencies"] = torch.Tensor(train_df["Target"].value_counts())

        loss_func = SSLLoss(
            unlabel_dl=unlabel_dls[0], n_out=n_out, weight=class_weights, **loss_params
        )
    else:
        loss_func = FocalLossFlat(gamma=2, weight=class_weights)
        loss_func = CrossEntropyLossFlat(weight=class_weights)
        # loss_func = None

    return loss_func
