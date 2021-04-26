""" Model related utilities """

import torch
import numpy as np

from fastai.callback.core import Callback
from fastai.vision.learner import *
from fastcore.foundation import L
from fastai.torch_core import params

try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    print(
        "EfficientNet models cannot be used because `efficientnet_pytorch` package is not installed"
    )


def efficientNet_split(m):
    return L(
        torch.nn.ModuleList([m._swish, m._bn0, m._conv_stem, m._blocks]),
        torch.nn.ModuleList([m._swish, m._bn1, m._conv_head]),
    ).map(params)


def create_model(
    model_arq, n_out, model=None, pretrained=True, n_in=1, ema=False, bn_final=False
):
    if model is None:
        if type(model_arq) is str and model_arq.startswith("efficientnet"):
            model = EfficientNet.from_pretrained(
                model_arq, num_classes=n_out, include_top=True, in_channels=n_in
            )
        else:
            model = create_cnn_model(
                model_arq,
                n_out=n_out,
                cut=None,
                pretrained=pretrained,
                n_in=n_in,
                bn_final=bn_final,
            )

    if torch.cuda.is_available():
        model = model.cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


# TODO: This creates issue with a huge difference between training and validation
# seems like the logits are one or two order of magnitude higher during validation
class EMAModel(Callback):
    """Evaluate model during validation with a EMA model
    Based on https://raw.githubusercontent.com/valencebond/FixMatch_pytorch/master/models/ema.py
    """

    def __init__(self, alpha=0.999):
        self.alpha = alpha

    def after_create(self):
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]

    def after_batch(self):
        if not self.training:
            return
        self.update_params()

    def before_validate(self):
        self.apply_shadow()

    def after_validate(self):
        self.restore()

    def update_params(self):
        decay = self.alpha
        state = self.model.state_dict()  # current params
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name] + (1 - decay) * state[name]
            )

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {k: v.clone().detach() for k, v in self.model.state_dict().items()}


def get_training_model(run_params, loss_params, train_df, n_in=1, model_self_sup=None):
    classes = train_df["Target"].unique()
    n_out = len(classes)

    if run_params["USE_SAVED_MODEL"]:
        body = create_model(
            run_params["MODEL"], n_out, pretrained=True, n_in=n_in, bn_final=True
        )

        load_model(
            file=run_params["PRETRAINED_MODEL_SAVE_NAME"],
            model=body,
            opt=None,
            with_opt=False,
            device=torch.cuda.current_device(),
            strict=False,
        )
        body = body[0]

        nf = num_features_model(nn.Sequential(*body.children())) * 2
        head = create_head(
            nf, n_out, concat_pool=True, bn_final=True, ps=run_params["HEAD_DROPOUT_P"]
        )

        model = nn.Sequential(body, head)
        apply_init(model[1], nn.init.kaiming_normal_)

    elif run_params["SELF_SUPERVISED"]:
        if model_self_sup is None:
            raise ValueError("No Self Supervised model provided")

        concat_pool = True
        for i, layer_block in enumerate(model_self_sup.children()):
            if i == 1:
                for layer in layer_block.children():
                    for j, layer_ in enumerate(layer.children()):
                        if j == 3:
                            nf = layer_.out_features
        # nf = num_features_model(nn.Sequential(*model_self_sup.children())) * (2 if concat_pool else 1)
        # head = create_head(nf, n_out, lin_ftrs=[512], ps=0.5, concat_pool=concat_pool, bn_final=True)

        # Seems there is somekind of issue and nf only can be 2048
        nf = 2048
        layers = [
            nn.Dropout(p=run_params["HEAD_DROPOUT_P"]),
            nn.Linear(nf, n_out),
            nn.BatchNorm1d(n_out, momentum=0.01),
        ]
        head = nn.Sequential(*layers)
        model = nn.Sequential(model_self_sup, head)
    else:
        model = create_model(
            run_params["MODEL"], n_out, pretrained=True, n_in=n_in, bn_final=True
        )

    # TODO: Should be corrected depending on the ALL_LABELS_IN_BATCH or WEIGTHED_SAMPLER?
    # Initialize last BatchNorm bias with values reflecting the current probabilities with Softmax
    with torch.no_grad():
        try:
            vals = torch.as_tensor(
                [
                    np.log(p)
                    for p in train_df["Target"].value_counts(normalize=True).values
                ]
            )
            for name, param in model[-1][-1].named_parameters():
                if "bias" in name:
                    param.copy_(vals)
        except TypeError:
            model._fc.bias.copy_(vals)

    if torch.cuda.is_available():
        model = model.cuda()

    if run_params["SSL"]:
        if run_params["SSL"] == run_params["SSL_MIX_MATCH"]:
            loss_params["model"] = model

        # EMAModel needs to be fixed before being used again
        # cbs.append(EMAModel(alpha=run_params['EMA_DECAY']))

    # Get splitter for model param groups
    meta = model_meta.get(run_params["MODEL"], {"cut": -1, "split": default_split})
    splitter = (
        efficientNet_split
        if type(run_params["MODEL"]) is str
        and run_params["MODEL"].startswith("efficientnet")
        else meta["split"]
    )

    return model, splitter
