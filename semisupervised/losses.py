# from abc import abstractmethod

from semisupervised.utils import de_interleave

class SemiLoss(object):
    """ Define structure and process that it is required for a Semi-supervised loss """

    def __init__(self, bs, lambda_u, n_out, Lx_criterion, Lu_criterion, axis=-1):
        self.axis = axis
        self.bs = bs
        self.lambda_u = lambda_u
        self.n_out = n_out
        self.Lx_criterion = Lx_criterion
        self.Lu_criterion = Lu_criterion
        self.losses = {}

    def __call__(self, logits, targets, reduction='mean'):#, batch_size, epoch):

        # put interleaved samples back
        logits = de_interleave(logits)

        # Transform label if it has been flatten
        if len(targets.size()) == 1:
            targets = targets.view(len(targets) // self.n_out, self.n_out)

        # Calculate supervised loss (Lx), unsupervised loss (Lu) and w scheduling (unsupervised multiplier)
        Lx = self.Lx_criterion(logits, targets, reduction=reduction)
        Lu = self.Lu_criterion(logits, targets, reduction=reduction)
        # w = self.w_scheduling(self.epoch)
        
        # Calculation of total loss
        loss = Lx + self.lambda_u * Lu

        if reduction == 'mean':
            # self.losses = {'loss': loss.mean(), 'Lx': Lx.mean(), 'Lu': Lu.mean(), 'w': w}
            return loss.mean()
        elif reduction=='sum':
            # self.losses = {'loss': loss.sum(), 'Lx': Lx.sum(), 'Lu': Lu.sum(), 'w': w}
            return loss.sum()
        else:
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