# Radiology AI

## Installation

Add conda-forge channel:
`conda config --add channels conda-forge`

Create environment:
`conda create -n radiology_ai python=3.6 -y`

Install environment:
`conda install -n radiology_ai -c fastai -c pytorch pytorch=1.6.0 fastai fastcore=1.0.9 pydicom gdcm kornia scikit-image scikit-learn pandas numpy ipykernel xlrd tensorboard -y`


## Troubleshooting

1. AttributeError: '_FakeLoader' object has no attribute 'persistent_workers'
`pip install --upgrade fastai`