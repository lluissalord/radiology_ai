Module radiology_ai.semisupervised.mixmatch.callback
====================================================
Callback class for MixMatch algorith

Classes
-------

`MixMatchCallback(unlabel_dl, transform_dl, T)`
:   MixMatch required preprocess before each batch

    ### Ancestors (in MRO)

    * fastai.callback.core.Callback
    * fastcore.foundation.GetAttr

    ### Class variables

    `run_before`
    :   Basic class handling tweaks of the training loop by changing a `Learner` in various events

    ### Methods

    `after_loss(self)`
    :

    `before_batch(self)`
    :