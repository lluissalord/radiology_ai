# Radiology AI

## Installation

Add conda-forge channel:
`conda config --add channels conda-forge`

Create environment:
`conda create -n radiology_ai python=3.8`

Install environment:
`conda create -n radiology_ai -c fastai -c pytorch fastai pydicom kornia scikit-learn pandas numpy ipykernel`