import torch

from fastai.callback.core import Callback
from fastai.callback.mixup import MixUp

from mixmatch.utils import interleave

class MixMatchCallback(Callback):

    run_after = MixUp
    def __init__(self, unlabel_dl, transform_dl, T):
        self.unlabel_dl = unlabel_dl
        self.transform_dl = transform_dl
        self.T = T

    def before_batch(self):
        if not self.training: return
        
        raw_inputs_u = self.unlabel_dl.one_batch()[0]
        inputs_u = self.transform_dl.after_batch(raw_inputs_u.detach())
        inputs_u2 = self.transform_dl.after_batch(raw_inputs_u.detach())

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = self.model(inputs_u)
            outputs_u2 = self.model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/self.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
        
        input_x = self.learn.xb
        if type(input_x) is tuple:
            input_x = input_x[0]

        target_x = self.learn.yb
        if type(target_x) is tuple:
            target_x = target_x[0]
    
        self.learn.xb = torch.cat([input_x, inputs_u, inputs_u2], dim=0).unsqueeze(0)
        self.learn.yb = torch.cat([target_x, targets_u, targets_u], dim=0).unsqueeze(0)

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        self.learn.xb = interleave(self.learn.xb)