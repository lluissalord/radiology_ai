from fastai.callback.core import Callback

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
