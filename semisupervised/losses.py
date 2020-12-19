# from abc import abstractmethod

import torch

from semisupervised.utils import de_interleave


class SuppressedConsistencyLoss(object):

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
    """ Define structure and process that it is required for a Semi-supervised loss """

    def __init__(self, bs, lambda_u, n_out, Lx_criterion, Lu_criterion, use_SCL=True, frequencies=None, beta=0.5, axis=-1):
        self.axis = axis
        self.bs = bs
        self.lambda_u = lambda_u
        self.n_out = n_out
        self.Lx_criterion = Lx_criterion
        self.Lu_criterion = Lu_criterion
        self.losses = {}

        if use_SCL:
            if frequencies is None:
                raise ValueError('In order to use SCL frequencies of targets should be provided')
            self.SCL = SuppressedConsistencyLoss(frequencies=frequencies, beta=beta)
        else:
            self.SCL = None

    def __call__(self, logits, targets, reduction='mean'):#, batch_size, epoch):

        # put interleaved samples back
        logits = de_interleave(logits)

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

        if self.SCL is not None:
            Lu = self.SCL(torch.argmax(logits, dim=-1)) * Lu

        # Calculation of total loss
        if reduction == 'mean':
            loss = Lx + (self.lambda_u * Lu).mean()
            # self.losses = {'loss': loss.mean(), 'Lx': Lx.mean(), 'Lu': Lu.mean(), 'w': w}
        elif reduction=='sum':
            loss = Lx + (self.lambda_u * Lu).sum()
            # self.losses = {'loss': loss.sum(), 'Lx': Lx.sum(), 'Lu': Lu.sum(), 'w': w}
        else:
            loss = Lx + self.lambda_u * Lu
            # self.losses = {'loss': loss, 'Lx': Lx, 'Lu': Lu, 'w': w}
 
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