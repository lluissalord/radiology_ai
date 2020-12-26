Module radiology_ai.semisupervised.fixmatch.losses
==================================================
Loss class for FixMatch algorith

Classes
-------

`FixMatchLoss(unlabel_dl, n_out, bs, mu, lambda_u, label_threshold=0.95, weight=None, *args, axis=-1, reduction='mean', **kwargs)`
:   Loss for FixMatch process

    ### Ancestors (in MRO)

    * fastai.losses.BaseLoss

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