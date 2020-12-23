""" Model related utilities """

import torch

from fastai.callback.core import Callback
from fastai.vision.learner import create_cnn_model

try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    print('EfficientNet models cannot be used because `efficientnet_pytorch` package is not installed')


def create_model(model_arq, n_out, model=None, pretrained=True, n_in=1, ema=False, bn_final=False):
    if model is None:
      if type(model_arq) is str and model_arq[:12].lower == 'efficientnet':
        model = EfficientNet.from_pretrained(model_arq, num_classes=n_out, include_top=True, in_channels=n_in)
      else:
        model = create_cnn_model(model_arq, n_out=n_out, cut=None, pretrained=pretrained, n_in=n_in, bn_final=bn_final)
    
    if torch.cuda.is_available():
      model = model.cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


class EMAModel(Callback):
    """ Evaluate model during validation with a EMA model
    Based on https://raw.githubusercontent.com/valencebond/FixMatch_pytorch/master/models/ema.py
    """
    def __init__(self, alpha=0.999):
        self.alpha = alpha
        
    def after_create(self):
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]

    def after_batch(self):
        if not self.training: return
        self.update_params()

    def before_validate(self):
        self.apply_shadow()

    def after_validate(self):
        self.restore()

    def update_params(self):
        decay = self.alpha
        state = self.model.state_dict()  # current params
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name] + (1 - decay) * state[name]
            )

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }
