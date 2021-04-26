""" Robust loss functions for training accurate deep neural networks (DNNs) in the presence of noisy (incorrect) labels.

Extracted [Normalized Loss Functions for Deep Learning with Noisy Labels](https://arxiv.org/abs/2006.13554) paper implemented on https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
```
@inproceedings{ma2020normalized,
  title={Normalized Loss Functions for Deep Learning with Noisy Labels},
  author={Ma, Xingjun and Huang, Hanxun and Wang, Yisen and Romano, Simone and Erfani, Sarah and Bailey, James},
  booktitle={ICML},
  year={2020}
}
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fastai.losses import BaseLoss

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda")
else:
    device = torch.device("cpu")


class FastaiLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(FastaiLoss, self).__init__()
        self.num_classes = num_classes

    def decodes(self, x):
        return x.argmax(dim=-1)

    # def decodes(self, x):
    #     dec = x.argmax(dim=-1)
    #     if len(dec.size()) == 1:
    #         dec = torch.zeros(len(dec), self.num_classes).scatter_(1, dec.cpu().view(-1,1).long(), 1)
    #     return dec

    def activation(self, x):
        return F.softmax(x, dim=-1)


class SCELoss(FastaiLoss):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__(num_classes=num_classes)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = (
            torch.nn.functional.one_hot(labels.long(), self.num_classes)
            .float()
            .to(self.device)
        )
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class ReverseCrossEntropy(FastaiLoss):
    def __init__(self, num_classes, scale=1.0):
        super(ReverseCrossEntropy, self).__init__(num_classes=num_classes)
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = (
            torch.nn.functional.one_hot(labels.long(), self.num_classes)
            .float()
            .to(self.device)
        )
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        return self.scale * rce.mean()


class NormalizedReverseCrossEntropy(FastaiLoss):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedReverseCrossEntropy, self).__init__(num_classes=num_classes)
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = (
            torch.nn.functional.one_hot(labels.long(), self.num_classes)
            .float()
            .to(self.device)
        )
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        normalizor = 1 / 4 * (self.num_classes - 1)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        return self.scale * normalizor * rce.mean()


class NormalizedCrossEntropy(FastaiLoss):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__(num_classes=num_classes)
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = (
            torch.nn.functional.one_hot(labels.long(), self.num_classes)
            .float()
            .to(self.device)
        )
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * nce.mean()


class GeneralizedCrossEntropy(FastaiLoss):
    def __init__(self, num_classes, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__(num_classes=num_classes)
        self.device = device
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = (
            torch.nn.functional.one_hot(labels.long(), self.num_classes)
            .float()
            .to(self.device)
        )
        gce = (1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()


class NormalizedGeneralizedCrossEntropy(FastaiLoss):
    def __init__(self, num_classes, scale=1.0, q=0.7):
        super(NormalizedGeneralizedCrossEntropy, self).__init__(num_classes=num_classes)
        self.device = device
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = (
            torch.nn.functional.one_hot(labels.long(), self.num_classes)
            .float()
            .to(self.device)
        )
        numerators = 1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        ngce = numerators / denominators
        return self.scale * ngce.mean()


class MeanAbsoluteError(FastaiLoss):
    def __init__(self, num_classes, scale=1.0):
        super(MeanAbsoluteError, self).__init__(num_classes=num_classes)
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = (
            torch.nn.functional.one_hot(labels.long(), self.num_classes)
            .float()
            .to(self.device)
        )
        mae = 1.0 - torch.sum(label_one_hot * pred, dim=1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        return self.scale * mae.mean()


class NormalizedMeanAbsoluteError(FastaiLoss):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedMeanAbsoluteError, self).__init__(num_classes=num_classes)
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = (
            torch.nn.functional.one_hot(labels.long(), self.num_classes)
            .float()
            .to(self.device)
        )
        normalizor = 1 / (2 * (self.num_classes - 1))
        mae = 1.0 - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * normalizor * mae.mean()


class NCEandRCE(FastaiLoss):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandRCE, self).__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)


class NCEandMAE(FastaiLoss):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandMAE, self).__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.mae(pred, labels)


class GCEandMAE(FastaiLoss):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandMAE, self).__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.mae(pred, labels)


class GCEandRCE(FastaiLoss):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandRCE, self).__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.rce(pred, labels)


class GCEandNCE(FastaiLoss):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandNCE, self).__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.nce = NormalizedCrossEntropy(num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.nce(pred, labels)


class NGCEandNCE(FastaiLoss):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandNCE, self).__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(
            scale=alpha, q=q, num_classes=num_classes
        )
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.nce(pred, labels)


class NGCEandMAE(FastaiLoss):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandMAE, self).__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(
            scale=alpha, q=q, num_classes=num_classes
        )
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.mae(pred, labels)


class NGCEandRCE(FastaiLoss):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandRCE, self).__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(
            scale=alpha, q=q, num_classes=num_classes
        )
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.rce(pred, labels)


class MAEandRCE(FastaiLoss):
    def __init__(self, alpha, beta, num_classes):
        super(MAEandRCE, self).__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.mae = MeanAbsoluteError(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.mae(pred, labels) + self.rce(pred, labels)


class NLNL(FastaiLoss):
    def __init__(self, train_loader, num_classes, ln_neg=1):
        super(NLNL, self).__init__(num_classes=num_classes)
        self.device = device
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.0
        if not hasattr(train_loader.dataset, "targets"):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (
                    torch.from_numpy(np.array(train_loader.dataset.targets)) == i
                ).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (
            labels.unsqueeze(-1).repeat(1, self.ln_neg)
            + torch.LongTensor(len(labels), self.ln_neg)
            .to(self.device)
            .random_(1, self.num_classes)
        ) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes - 1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(
            labels
        ) * self.ln_neg

        s_neg = torch.log(torch.clamp(1.0 - F.softmax(pred, 1), min=1e-5, max=1.0))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size()).to(self.device)
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(
            s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)
        ) * float((labels_neg >= 0).sum())
        loss = (loss + loss_neg) / (
            float((labels >= 0).sum()) + float((labels_neg[:, 0] >= 0).sum())
        )
        return loss


class FocalLoss(nn.CrossEntropyLoss):
    """ Focal loss for classification tasks on imbalanced datasets """

    def __init__(self, gamma, weight=None, ignore_index=-100, reduction="mean"):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction="none")
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return (
            torch.mean(loss)
            if self.reduction == "mean"
            else torch.sum(loss)
            if self.reduction == "sum"
            else loss
        )


class FocalLossFlat(BaseLoss):
    "Same as `FocalLoss`, but flattens input and target."
    y_int = True

    def __init__(
        self, weight=None, ignore_index=-100, reduction="mean", *args, axis=-1, **kwargs
    ):
        super().__init__(
            FocalLoss,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            *args,
            axis=axis,
            **kwargs
        )
        self.axis = axis

    def decodes(self, x):
        return x.argmax(dim=self.axis)

    def activation(self, x):
        return F.softmax(x, dim=self.axis)


class NormalizedFocalLoss(FastaiLoss):
    def __init__(
        self, scale=1.0, gamma=0, num_classes=10, alpha=None, size_average=True
    ):
        super(NormalizedFocalLoss, self).__init__(num_classes=num_classes)
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class NFLandNCE(FastaiLoss):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandNCE, self).__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes
        )
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.nce(pred, labels)


class NFLandMAE(FastaiLoss):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandMAE, self).__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes
        )
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.mae(pred, labels)


class NFLandRCE(FastaiLoss):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandRCE, self).__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes
        )
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)


class DMILoss(FastaiLoss):
    def __init__(self, num_classes):
        super(DMILoss, self).__init__(num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, output, target):
        outputs = F.softmax(output, dim=1)
        targets = target.reshape(target.size(0), 1).cpu()
        y_onehot = torch.FloatTensor(target.size(0), self.num_classes).zero_()
        y_onehot.scatter_(1, targets, 1)
        y_onehot = y_onehot.transpose(0, 1).cuda()
        mat = y_onehot @ outputs
        return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)
