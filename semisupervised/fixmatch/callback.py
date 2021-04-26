""" Callback class for FixMatch algorith """

import torch

from fastai.callback.core import Callback

from semisupervised.utils import interleave


class FixMatchCallback(Callback):
    """FixMatch required preprocess before each batch"""

    def __init__(self, unlabel_dl, weak_transform_dl, strong_transform_dl):
        self.unlabel_dl = unlabel_dl
        self.weak_transform_dl = weak_transform_dl
        self.strong_transform_dl = strong_transform_dl

    def before_batch(self):

        # Only process if is trainig
        if not self.training:
            return

        # Extract a batch of unlabel images and repeat twice the transformation (different results due to randomness)
        raw_inputs_u = self.unlabel_dl.one_batch()[0]
        inputs_u_weak = self.weak_transform_dl.after_batch(
            raw_inputs_u.clone().detach()
        )
        inputs_u_strong = self.strong_transform_dl.after_batch(
            raw_inputs_u.clone().detach()
        )

        # Make suree the input has the right structure
        input_x = self.learn.xb
        if type(input_x) is tuple:
            input_x = input_x[0]

        # Make suree the target has the right structure
        target_x = self.learn.yb
        if type(target_x) is tuple:
            target_x = target_x[0]

        # Set targets to 0 for unlabel data
        targets_u_weak = torch.zeros(inputs_u_weak.size()[:1] + target_x.size()[1:])
        targets_u_strong = torch.zeros(inputs_u_strong.size()[:1] + target_x.size()[1:])
        if torch.cuda.is_available:
            targets_u_weak = targets_u_weak.cuda()
            targets_u_strong = targets_u_strong.cuda()

        # Set together label and unlabel data
        self.learn.xb = torch.cat(
            [input_x, inputs_u_weak, inputs_u_strong], dim=0
        ).unsqueeze(0)
        self.learn.yb = (
            torch.cat([target_x, targets_u_weak, targets_u_strong], dim=0)
            .unsqueeze(0)
            .long()
        )

        # Interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        self.learn.xb = interleave(self.learn.xb, self.learn.dls[0].bs)
