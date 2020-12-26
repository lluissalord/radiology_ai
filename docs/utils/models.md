Module radiology_ai.utils.models
================================
Model related utilities

Functions
---------

    
`create_model(model_arq, n_out, model=None, pretrained=True, n_in=1, ema=False, bn_final=False)`
:   

Classes
-------

`EMAModel(alpha=0.999)`
:   Evaluate model during validation with a EMA model
    Based on https://raw.githubusercontent.com/valencebond/FixMatch_pytorch/master/models/ema.py

    ### Ancestors (in MRO)

    * fastai.callback.core.Callback
    * fastcore.foundation.GetAttr

    ### Methods

    `after_batch(self)`
    :

    `after_create(self)`
    :

    `after_validate(self)`
    :

    `apply_shadow(self)`
    :

    `before_validate(self)`
    :

    `get_model_state(self)`
    :

    `restore(self)`
    :

    `update_params(self)`
    :