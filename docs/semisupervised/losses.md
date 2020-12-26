Module radiology_ai.semisupervised.losses
=========================================
Loss classes required for Semi-Supervised Learning

Classes
-------

`SemiLoss(bs, lambda_u, n_out, Lx_criterion, Lu_criterion, use_SCL=True, frequencies=None, beta=0.5, axis=-1)`
:   Define structure and process that it is required for a Semi-supervised loss

`SuppressedConsistencyLoss(frequencies, beta=0.5)`
:   Use Supressed Consistency Loss from [Class-Imbalanced Semi-Supervised Learning](https://arxiv.org/pdf/2002.06815.pdf) by Hyun et al. 
    Intuition that we should suppress the consistency regularization of minor classes in class-imbalanced Semi-Supervised Learning