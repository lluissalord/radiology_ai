""" Loss class for MixMatch algorith """

import torch
import torch.nn.functional as F
import numpy as np

from semisupervised.losses import SemiLoss, SemiLossBase
from semisupervised.utils import categorical_to_one_hot


class MixMatchLoss(SemiLossBase):
    """Loss for MixMatch process"""

    y_int = False

    def __init__(
        self,
        unlabel_dl,
        model,
        n_out,
        bs,
        lambda_u,
        weight=None,
        *args,
        axis=-1,
        reduction="mean",
        **kwargs
    ):
        super().__init__(
            loss_cls=SemiLoss,
            bs=bs,
            lambda_u=lambda_u,
            n_out=n_out,
            Lx_criterion=self.Lx_criterion,
            Lu_criterion=self.Lu_criterion,
            flatten=False,
            floatify=True,
            *args,
            axis=axis,
            **kwargs
        )
        self.axis = axis
        self.n_out = n_out
        self.unlabel_dl = unlabel_dl
        self.unlabel_dl_iter = iter(unlabel_dl)
        self.model = model
        self.bs = bs
        self.lambda_u = lambda_u
        self.losses = {}
        self.reduction = reduction
        self.weight = weight
        if weight is None:
            self.weight = torch.as_tensor(1.0).float()

            if torch.cuda.is_available:
                self.weight = self.weight.cuda()

        self.log_loss("w", self.lambda_u)

    def Lx_criterion(self, logits, targets, reduction="mean"):
        """Supervised loss criterion"""

        logits_x = logits[: self.bs]
        targets_x = targets[: self.bs]

        if len(targets_x.size()) == 1:
            targets_x = categorical_to_one_hot(targets_x, self.n_out)

        Lx = -torch.sum(self.weight * F.log_softmax(logits_x, dim=1) * targets_x, dim=1)
        if self.reduction == "mean":
            Lx = Lx.mean()
        elif self.reduction == "sum":
            Lx = Lx.sum()

        isTraining = len(logits) > self.bs

        # Only log loss if is on training
        if isTraining:
            self.log_loss("Lx", Lx.clone().detach())

        return Lx

    def Lu_criterion(self, logits, targets, reduction="mean"):
        """Unsupervised loss criterion"""

        logits_u = logits[self.bs :]
        targets_u = targets[self.bs :]

        if len(targets_u.size()) == 1:
            targets_u = categorical_to_one_hot(targets_u, self.n_out)

        # Return zero if no logits are provided (when not training)
        if len(logits_u):
            probs_u = torch.softmax(logits_u, dim=1)

            # return torch.mean((probs_u - targets_u)**2, dim=1)
            Lu = (probs_u - targets_u) ** 2
        else:
            Lu = torch.zeros(1)

        isTraining = len(logits) > self.bs

        # Only log loss if is on training
        if isTraining:
            self.log_loss("Lu", Lu.clone().detach().mean())

    def w_scheduling(self, epoch):
        """Scheduling of w paramater (unsupervised loss multiplier)"""

        # self.log_loss('w', self.lambda_u * linear_rampup(epoch))

        return self.lambda_u * linear_rampup(epoch)

    def decodes(self, x):
        return x.argmax(dim=self.axis)

    def activation(self, x):
        return F.softmax(x, dim=self.axis)


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
