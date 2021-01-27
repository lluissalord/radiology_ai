# Radiology AI

This is an OpenSource project collaborating with a healthcare mutual insurance company (data provider). The aim of this project is to build a pipeline to classify knee frontal radiography images according to Schatzker classification (from 0 to 6) from DICOM files. However, as first approach it used a binary classification of fractured knee (Schatzker classification different than 0).

Medical imaging projects as this one have several features that should be taken into account:

1. Lack of datasets/samples for specific task

2. Annotation requires expertise on traumatology field, hence it is limited

3. Data is highly imbalanced towards non-fractured knees

4. Medical images require special treatments

In order to tackle these issues we implemented the following methods on [PyTorch](https://pytorch.org/) using [Fastai](https://docs.fast.ai/) platform:

- [x] **Transfer Learning** on several steps (first ImageNet and then using DICOM metadata for pretraining model)

- [x] **Self-Supervised Learning** ([`pytorch-lightning`](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html) package) for state-of-the-art transfer learning

- [x] **Semi-Supervised Learning** ([MixMatch](https://arxiv.org/abs/1905.02249) and [FixMatch](https://arxiv.org/abs/2001.07685) own implementations) to be able to train model without full dataset annotated

- [x] **DataLoader sampler** which allows to have a minimum of samples of all the labels in each batch

- [x] Use of specific losses to be robust to **imbalance** datasets

- [x] Preprocesses to **transform DICOM files** into images independently of its metadata

- [x] **Adaptative equalization of histogram** to normalize bimodal data

- [x] [**Knee localization**](https://github.com/MIPT-Oulu/KneeLocalizer) to center-crop images on relevant part

- [x] Model interpretation with GradCAM or Layer Relevance Propagation

- [ ] Active Learning to select most interesting files to annotate

- [ ] Organize experiments for Hyperparameters search (MLFlow or Ax-BoTorch)

- [ ] Use channels for different views of the radiography

---
## Installation

Add conda-forge channel:
`conda config --add channels conda-forge`

Create environment:
`conda create -n radiology_ai python=3.6 --yes -q`
`conda activate radiology_ai`

Install environment:
`conda install -n radiology_ai -c fastai -c pytorch pytorch=1.6.0 fastai fastcore=1.0.9 pydicom gdcm=2.8.4 kornia=0.3.0 scikit-image scikit-learn pandas ipykernel ipywidgets tensorboard openpyxl --yes -q`

Install EfficientNet for Pytorch and Pytorch Ligthning Bolts:
`pip install efficientnet_pytorch opencv-python`

Install Pytorch Lignthning Bolts (due to an issue not yet solved on release better to use this than pip install pytorch-lightning):
~~`pip install git+https://github.com/PytorchLightning/pytorch-lightning-bolts.git@master --upgrade`~~
`pip install https://github.com/lluissalord/pytorch-lightning-bolts/archive/issue_436_simCLR_IndexError.zip --upgrade`

Install fastai2_extensions (renamed as amalgam) to use gradCAM visualizations:
`pip install git+https://github.com/Synopsis/amalgam.git`


---

## API Documentation

API Documentation on [lluissalord.github.io/radiology_ai/](https://lluissalord.github.io/radiology_ai/)

---

## Troubleshooting

1. AttributeError: '_FakeLoader' object has no attribute 'persistent_workers'
`pip install --upgrade fastai`

2. Progress bar issues using VSCode
`conda install "ipython>=6.0.0"`