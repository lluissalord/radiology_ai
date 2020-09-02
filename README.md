# Radiology AI

## Installation
Create and install environment
`conda create -n radiology_ai python=3.8 scikit-learn pandas numpy ipykernel`

Install fastai:
`conda install -n radiology_ai -c fastai -c pytorch -c anaconda fastai gh anaconda`

In order to use fastai for medical imaging, which here is the case:
`conda install -n radiology_ai pydicom kornia`