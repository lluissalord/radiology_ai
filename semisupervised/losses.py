""" Loss classes required for Semi-Supervised Learning """

# from abc import abstractmethod

import torch
import numpy as np

from semisupervised.utils import de_interleave

from fastai.basics import BaseLoss
from fastai.callback.core import Callback


class SuppressedConsistencyLoss(object):
    """Use Supressed Consistency Loss from [Class-Imbalanced Semi-Supervised Learning](https://arxiv.org/pdf/2002.06815.pdf) by Hyun et al.
    Intuition that we should suppress the consistency regularization of minor classes in class-imbalanced Semi-Supervised Learning
    """

    def __init__(self, frequencies, beta=0.5):
        weights = beta ** (1 - frequencies.float() / torch.max(frequencies))
        weights = weights.unsqueeze(1)
        self.emb = torch.nn.Embedding(len(frequencies), 1, _weight=weights)
        self.emb.weight.requires_grad = False

        if torch.cuda.is_available():
            self.emb = self.emb.cuda()

    def __call__(self, targets):
        return self.emb(targets)


class SemiLoss(object):
    """Define structure and process that it is required for a Semi-supervised loss"""

    def __init__(
        self,
        bs,
        lambda_u,
        n_out,
        Lx_criterion,
        Lu_criterion,
        use_SCL=True,
        frequencies=None,
        beta=0.5,
        axis=-1,
    ):
        self.axis = axis
        self.bs = bs
        self.lambda_u = lambda_u
        self.n_out = n_out
        self.Lx_criterion = Lx_criterion
        self.Lu_criterion = Lu_criterion
        self.losses = {}

        if use_SCL:
            if frequencies is None:
                raise ValueError(
                    "In order to use SCL frequencies of targets should be provided"
                )
            self.SCL = SuppressedConsistencyLoss(frequencies=frequencies, beta=beta)
        else:
            self.SCL = None

    def __call__(self, logits, targets, reduction="mean"):  # , batch_size, epoch):

        isTraining = len(logits) > self.bs

        # put interleaved samples back
        if isTraining:
            logits = de_interleave(logits, self.bs)

        # Clip values of logits
        logits = torch.clamp(
            logits, min=np.finfo(np.float16).min + 1, max=np.finfo(np.float16).max - 1
        )

        # Transform label if it has been flatten
        # if len(targets.size()) == 1:
        #     targets = targets.view(len(targets) // self.n_out, self.n_out)

        # Calculate supervised loss (Lx), unsupervised loss (Lu) and w scheduling (unsupervised multiplier)
        Lx = self.Lx_criterion(logits, targets, reduction=reduction)
        Lu = self.Lu_criterion(logits, targets, reduction=reduction)
        # w = self.w_scheduling(self.epoch)

        if torch.cuda.is_available:
            Lx = Lx.cuda()
            Lu = Lu.cuda()

        if isTraining and self.SCL is not None:
            with torch.no_grad():
                SCL_coefs = self.SCL(torch.argmax(logits[self.bs :], dim=-1))
            Lu = SCL_coefs * Lu

        # Calculation of total loss
        if reduction == "mean":
            loss = Lx + (self.lambda_u * Lu).mean()
            # self.losses = {'loss': loss.mean(), 'Lx': Lx.mean(), 'Lu': Lu.mean(), 'w': self.lambda_u}
        elif reduction == "sum":
            loss = Lx + (self.lambda_u * Lu).sum()
            # self.losses = {'loss': loss.sum(), 'Lx': Lx.sum(), 'Lu': Lu.sum(), 'w': self.lambda_u}
        else:
            loss = Lx + self.lambda_u * Lu
            # self.losses = {'loss': loss, 'Lx': Lx, 'Lu': Lu, 'w': self.lambda_u}

        return loss

    # @abstractmethod
    # def Lx_criterion(self, logits, targets):
    #     pass

    # @abstractmethod
    # def Lu_criterion(self, logits, targets):
    #     pass

    # @abstractmethod
    # def w_scheduling(self, epoch):
    #     pass


class SemiLossBase(BaseLoss):
    def __init__(self, beta=0.98, max_len_losses=500, **kwargs):
        super().__init__(**kwargs)
        self.smooth_losses = {}
        self.losses = {}
        self.beta = beta
        self.max_len_losses = max_len_losses

    def log_loss(self, name, val):
        if name in self.smooth_losses:
            self.smooth_losses[name] = (
                1 - self.beta
            ) * val + self.beta * self.smooth_losses[name]
            self.losses[name].append(val)

            if len(self.losses[name]) > self.max_len_losses:
                del self.losses[name][0]
        else:
            self.smooth_losses[name] = val
            self.losses[name] = [val]

    def Lx(self):
        return self.smooth_losses["Lx"] if "Lx" in self.smooth_losses else None

    def Lu(self):
        return self.smooth_losses["Lu"] if "Lu" in self.smooth_losses else None

    # def total_loss(self):
    #     return self.smooth_losses['loss'] if 'loss' in self.smooth_losses else None

    def w(self):
        return self.smooth_losses["w"] if "w" in self.smooth_losses else None
