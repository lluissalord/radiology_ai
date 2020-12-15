# Radiology AI

## Installation

Add conda-forge channel:
`conda config --add channels conda-forge`

Create environment:
`conda create -n radiology_ai python=3.6 --yes -q`
`conda activate radiology_ai`

Install environment:
`conda install -n radiology_ai -c fastai -c pytorch pytorch=1.6.0 fastai fastcore=1.0.9 pydicom gdcm kornia scikit-image scikit-learn pandas numpy ipykernel ipywidgets xlrd tensorboard openpyxl pytorch-lightning --yes -q`

Install EfficientNet for Pytorch and Pytorch Ligthning Bolts
`pip install efficientnet_pytorch opencv-python`

Install Pytorch Lignthning Bolts (due to an issue not yet solved on release better to use this than pip install pytorch-lightning)
`pip install git+https://github.com/PytorchLightning/pytorch-lightning-bolts.git@master --upgrade`

## Troubleshooting

1. AttributeError: '_FakeLoader' object has no attribute 'persistent_workers'
`pip install --upgrade fastai`

2. Progress bar issues using VSCode
`conda install "ipython>=6.0.0"`