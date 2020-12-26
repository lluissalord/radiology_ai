Module radiology_ai.APL_losses
==============================
Extracted from https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
```
@inproceedings{ma2020normalized,
  title={Normalized Loss Functions for Deep Learning with Noisy Labels},
  author={Ma, Xingjun and Huang, Hanxun and Wang, Yisen and Romano, Simone and Erfani, Sarah and Bailey, James},
  booktitle={ICML},
  year={2020}
}
```

Classes
-------

`DMILoss(num_classes)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, output, target) ‑> Callable[..., Any]`
    :

`FastaiLoss(num_classes)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * radiology_ai.APL_losses.DMILoss
    * radiology_ai.APL_losses.FocalLoss
    * radiology_ai.APL_losses.GCEandMAE
    * radiology_ai.APL_losses.GCEandNCE
    * radiology_ai.APL_losses.GCEandRCE
    * radiology_ai.APL_losses.GeneralizedCrossEntropy
    * radiology_ai.APL_losses.MAEandRCE
    * radiology_ai.APL_losses.MeanAbsoluteError
    * radiology_ai.APL_losses.NCEandMAE
    * radiology_ai.APL_losses.NCEandRCE
    * radiology_ai.APL_losses.NFLandMAE
    * radiology_ai.APL_losses.NFLandNCE
    * radiology_ai.APL_losses.NFLandRCE
    * radiology_ai.APL_losses.NGCEandMAE
    * radiology_ai.APL_losses.NGCEandNCE
    * radiology_ai.APL_losses.NGCEandRCE
    * radiology_ai.APL_losses.NLNL
    * radiology_ai.APL_losses.NormalizedCrossEntropy
    * radiology_ai.APL_losses.NormalizedFocalLoss
    * radiology_ai.APL_losses.NormalizedGeneralizedCrossEntropy
    * radiology_ai.APL_losses.NormalizedMeanAbsoluteError
    * radiology_ai.APL_losses.NormalizedReverseCrossEntropy
    * radiology_ai.APL_losses.ReverseCrossEntropy
    * radiology_ai.APL_losses.SCELoss

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `activation(self, x)`
    :

    `decodes(self, x)`
    :

    `forward(self, *input: Any) ‑> NoneType`
    :

`FocalLoss(gamma=0, alpha=None, size_average=True)`
:   https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, input, target) ‑> Callable[..., Any]`
    :

`GCEandMAE(alpha, beta, num_classes, q=0.7)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`GCEandNCE(alpha, beta, num_classes, q=0.7)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`GCEandRCE(alpha, beta, num_classes, q=0.7)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`GeneralizedCrossEntropy(num_classes, q=0.7)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`MAEandRCE(alpha, beta, num_classes)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`MeanAbsoluteError(num_classes, scale=1.0)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`NCEandMAE(alpha, beta, num_classes)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`NCEandRCE(alpha, beta, num_classes)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`NFLandMAE(alpha, beta, num_classes, gamma=0.5)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`NFLandNCE(alpha, beta, num_classes, gamma=0.5)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`NFLandRCE(alpha, beta, num_classes, gamma=0.5)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`NGCEandMAE(alpha, beta, num_classes, q=0.7)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`NGCEandNCE(alpha, beta, num_classes, q=0.7)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`NGCEandRCE(alpha, beta, num_classes, q=0.7)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`NLNL(train_loader, num_classes, ln_neg=1)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`NormalizedCrossEntropy(num_classes, scale=1.0)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`NormalizedFocalLoss(scale=1.0, gamma=0, num_classes=10, alpha=None, size_average=True)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, input, target) ‑> Callable[..., Any]`
    :

`NormalizedGeneralizedCrossEntropy(num_classes, scale=1.0, q=0.7)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`NormalizedMeanAbsoluteError(num_classes, scale=1.0)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`NormalizedReverseCrossEntropy(num_classes, scale=1.0)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`ReverseCrossEntropy(num_classes, scale=1.0)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :

`SCELoss(alpha, beta, num_classes=10)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * radiology_ai.APL_losses.FastaiLoss
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, pred, labels) ‑> Callable[..., Any]`
    :