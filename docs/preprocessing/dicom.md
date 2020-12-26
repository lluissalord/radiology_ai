Module radiology_ai.preprocessing.dicom
=======================================
Preprocessings used for DICOM treatments

Classes
-------

`DCMPreprocessDataset(fnames, resize=None, padding_to_square=True, bins=None)`
:   Dataset class for DCM preprocessing

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset

    ### Methods

    `get_dcm(self, idx)`
    :   Get the DCM file

    `get_dcm_by_name(self, img_name)`
    :

    `get_fname(self, idx)`
    :   Get the filenames

    `init_bins(self, n_samples=None)`
    :   Initialize bins to equally distribute the histogram of the dataset

    `save(self, dst_folder, extension='png', overwrite=True, keep=False, clean_folder=False)`
    :   Saves all preprocessed files converting them into image files

`HistScaled_Dicom(bins=None)`
:   Transformation of Histogram Scaling compatible with DataLoaders, allowing Histogram Scaling on the fly

    ### Ancestors (in MRO)

    * fastcore.transform.Transform

    ### Methods

    `decodes(...)`
    :   Dictionary-like object; `__getitem__` matches keys of types using `issubclass`

    `encodes(...)`
    :   Dictionary-like object; `__getitem__` matches keys of types using `issubclass`

    `setups(...)`
    :   Dictionary-like object; `__getitem__` matches keys of types using `issubclass`

`PILDicom_scaled()`
:   Generalization of PILDicom class making sure it is properly scaled taking into account DICOM metadata

    ### Ancestors (in MRO)

    * fastai.medical.imaging.PILDicom
    * fastai.vision.core.PILBase
    * PIL.Image.Image

    ### Static methods

    `create(fn: (<class 'pathlib.Path'>, <class 'str'>, <class 'bytes'>), mode=None) ‑> NoneType`
    :   Open a `DICOM file` from path `fn` or bytes `fn` and load it as a `PIL Image`