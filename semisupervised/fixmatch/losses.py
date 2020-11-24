import torch
import torch.nn.functional as F
import numpy as np

from fastai.basics import BaseLoss

from semisupervised.losses import SemiLoss

class FixMatchLoss(BaseLoss):
    """ Loss for FixMatch process """

    def __init__(self, unlabel_dl, n_out, bs, mu, lambda_u, label_threshold=0.95, weight=None, *args, axis=-1, reduction='mean', **kwargs):
        super().__init__(loss_cls=SemiLoss, bs=bs, lambda_u=lambda_u, n_out=n_out, Lx_criterion=self.Lx_criterion, Lu_criterion=self.Lu_criterion, flatten=False, floatify=True, *args, axis=axis, **kwargs)
        self.axis = axis
        self.n_out = n_out
        self.bs = bs
        self.mu = mu
        self.lambda_u = lambda_u
        self.label_threshold = label_threshold
        self.losses = {}
        self.weight = weight
        self.reduction = reduction

    def Lx_criterion(self, logits, targets, reduction='mean'):
        """ Supervised loss criterion """

        logits_x = logits[:self.bs]
        targets_x = targets[:self.bs]
        
        if len(targets_x.size()) == 2:
            targets_x = torch.argmax(targets_x, axis=1)

        if torch.cuda.is_available():
            targets_x = targets_x.cuda()

        Lx_criterion_func = torch.nn.CrossEntropyLoss(weight=self.weight, reduction=self.reduction)
        if torch.cuda.is_available:
            Lx_criterion_func = Lx_criterion_func.cuda()

        return Lx_criterion_func(logits_x, targets_x.long())

    def Lu_criterion(self, logits, targets, reduction='mean'):
        """ Unsupervised loss criterion """
        
        # Return zero if no logits are provided (when not training)
        if len(logits[self.bs:]):
            
            # Split between logits of weak and strong transformation, according to batch size and mu
            logits_u_w, logits_u_s = torch.split(logits[self.bs:], self.bs * self.mu)

            # Calculate the guessed labels
            with torch.no_grad():
                probs = torch.softmax(logits_u_w, dim=1)
                scores, lbs_u_guess = torch.max(probs, dim=1)
                mask = scores.ge(self.label_threshold).float()
            
            Lu_criterion_func = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='none')
            if torch.cuda.is_available:
                Lu_criterion_func = Lu_criterion_func.cuda()

            return (Lu_criterion_func(logits_u_s, lbs_u_guess) * mask)
        else:
            return torch.zeros(1)

    def w_scheduling(self, epoch):
        """ Scheduling of w paramater (unsupervised loss multiplier) """

        return self.lambda_u

    def decodes(self, x):    return x.argmax(dim=1)

    def activation(self, x): return F.softmax(x, dim=self.axis)