Module radiology_ai.semisupervised.mixmatch.losses
==================================================
Loss class for MixMatch algorith

Functions
---------

    
`linear_rampup(current, rampup_length)`
:   

Classes
-------

`MixMatchLoss(unlabel_dl, model, n_out, bs, lambda_u, weight=None, *args, axis=-1, reduction='mean', **kwargs)`
:   Loss for MixMatch process

    ### Ancestors (in MRO)

    * fastai.losses.BaseLoss

    ### Class variables

    `y_int`
    :

    ### Methods

    `Lu_criterion(self, logits, targets, reduction='mean')`
    :   Unsupervised loss criterion

    `Lx_criterion(self, logits, targets, reduction='mean')`
    :   Supervised loss criterion

    `activation(self, x)`
    :   Do nothing (method)

    `decodes(self, x)`
    :   Do nothing (method)

    `w_scheduling(self, epoch)`
    :   Scheduling of w paramater (unsupervised loss multiplier)