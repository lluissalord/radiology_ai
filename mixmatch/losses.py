import torch
import torch.nn.functional as F
import numpy as np

from fastai.basics import BaseLoss
# from abc import abstractmethod

from mixmatch.utils import de_interleave

class SemiLoss(object):
    def __init__(self, bs, lambda_u, n_out, Lx_criterion, Lu_criterion, axis=-1):
        self.axis = axis
        self.bs = bs
        self.lambda_u = lambda_u
        self.n_out = n_out
        self.Lx_criterion = Lx_criterion
        self.Lu_criterion = Lu_criterion
        self.losses = {}

    def __call__(self, logits, targets):#, batch_size, epoch):

        # put interleaved samples back
        # logits = de_interleave(logits)

        # Transform label if it has been flatten
        if len(targets.size()) == 1:
            targets = targets.view(len(targets) // n_out, n_out)

        Lx = self.Lx_criterion(logits, targets)
        Lu = self.Lu_criterion(logits, targets)
        # w = self.w_scheduling(self.epoch)
        
        loss = Lx + self.lambda_u * Lu
        # self.losses = {'loss': loss, 'Lx': Lx, 'Lu': Lu, 'w': w}
        self.losses = {'loss': loss, 'Lx': Lx, 'Lu': Lu}

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

class MixMatchLoss(BaseLoss):
    def __init__(self, unlabel_dl, model, n_out, bs, lambda_u, weight=1, *args, axis=-1, **kwargs):
        super().__init__(loss_cls=SemiLoss, bs=bs, lambda_u=lambda_u, n_out=n_out, Lx_criterion=self.Lx_criterion, Lu_criterion=self.Lu_criterion, flatten=False, floatify=True, *args, axis=axis, **kwargs)
        self.axis = axis
        self.n_out = n_out
        self.unlabel_dl = unlabel_dl
        self.unlabel_dl_iter = iter(unlabel_dl)
        self.model = model
        self.bs = bs
        self.lambda_u = lambda_u
        self.losses = {}
        self.weight = weight.float()

    def Lx_criterion(self, logits, targets):
        logits_x = logits[:self.bs]
        targets_x = targets[:self.bs]
        
        return -torch.mean(torch.sum(self.weight * F.log_softmax(logits_x, dim=1) * targets_x, dim=1))

    def Lu_criterion(self, logits, targets):
        logits_u = logits[self.bs:]
        targets_u = targets[self.bs:]
        
        if len(logits_u):
            probs_u = torch.softmax(logits_u, dim=1)
            
            return torch.mean((probs_u - targets_u)**2)
        else:
            return 0

    def w_scheduling(self, epoch):
        return self.lambda_u * linear_rampup(epoch)

    # def decodes(self, x):    return x.argmax(dim=self.axis)
    def decodes(self, x):
        dec = x.argmax(dim=self.axis)
        if len(dec.size()) == 1:
            dec = torch.zeros(len(dec), self.n_out).scatter_(1, dec.cpu().view(-1,1).long(), 1)
        return dec

    def activation(self, x): return F.softmax(x, dim=self.axis)

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)