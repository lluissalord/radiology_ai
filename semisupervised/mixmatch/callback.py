""" Callback class for MixMatch algorith """

import torch

from fastai.callback.core import Callback
from fastai.callback.mixup import MixUp

from semisupervised.utils import interleave, de_interleave, categorical_to_one_hot


class MixMatchCallback(Callback):
    """MixMatch required preprocess before each batch"""

    run_before = MixUp

    def __init__(self, unlabel_dl, transform_dl, T):
        self.unlabel_dl = unlabel_dl
        self.transform_dl = transform_dl
        self.T = T

    def before_batch(self):

        # Only process if is trainig
        if not self.training:
            return

        # Extract a batch of unlabel images and repeat twice the transformation (different results due to randomness)
        raw_inputs_u = self.unlabel_dl.one_batch()[0]
        if torch.cuda.is_available:
            inputs_u = self.transform_dl.after_batch(
                raw_inputs_u.clone().detach().cuda()
            )
            inputs_u2 = self.transform_dl.after_batch(
                raw_inputs_u.clone().detach().cuda()
            )
        else:
            inputs_u = self.transform_dl.after_batch(raw_inputs_u.clone().detach())
            inputs_u2 = self.transform_dl.after_batch(raw_inputs_u.clone().detach())

        # Compute guessed labels of unlabel samples
        with torch.no_grad():
            outputs_u = self.model(inputs_u)
            outputs_u2 = self.model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p ** (1 / self.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # Make suree the input has the right structure
        input_x = self.learn.xb
        if type(input_x) is tuple:
            input_x = input_x[0]

        # Make suree the target has the right structure
        targets_x = self.learn.yb
        if type(targets_x) is tuple:
            targets_x = targets_x[0]

        if len(targets_x.size()) == 1:
            targets_x = categorical_to_one_hot(targets_x, targets_u.size()[-1])

        # Set together label and unlabel data
        self.learn.xb = torch.cat(
            [input_x, inputs_u, inputs_u2], dim=0
        )  # .unsqueeze(0)
        self.learn.yb = (
            torch.cat([targets_x, targets_u, targets_u], dim=0),  # .unsqueeze(0)
        )

        # Interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        self.learn.xb = interleave(self.learn.xb, self.learn.dls[0].bs)

    def after_loss(self):
        # Once loss has been calculated then it is only required to use the supervised targets/preds

        # Only process if is trainig
        if not self.training:
            return

        # Make suree the target has the right structure
        isTuple = False
        if type(self.learn.yb) is tuple:
            isTuple = True
            self.learn.yb = self.learn.yb[0]

        # Transform into categorical data and select only supervised targets
        self.learn.yb = torch.argmax(self.learn.yb[: self.learn.dls[0].bs], dim=1)

        # Select only supervised preds
        self.learn.pred = de_interleave(self.learn.pred, self.learn.dls[0].bs)[
            : self.learn.dls[0].bs
        ]

        # Format again as it should
        if isTuple:
            self.learn.yb = (self.learn.yb,)
