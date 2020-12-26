Module radiology_ai.preprocessing.transforms
============================================
Fastai Transformations to be used on training

Classes
-------

`CLAHE_Transform(PIL_cls, clipLimit=2.0, tileGridSize=(8, 8), grayscale=True, np_input=False, np_output=False)`
:   Implement CLAHE transformation for Adaptative Histogram Equalization

    ### Ancestors (in MRO)

    * fastcore.transform.Transform

    ### Methods

    `decodes(...)`
    :   Dictionary-like object; `__getitem__` matches keys of types using `issubclass`

    `encodes(...)`
    :   Dictionary-like object; `__getitem__` matches keys of types using `issubclass`

    `setups(...)`
    :   Dictionary-like object; `__getitem__` matches keys of types using `issubclass`

`HistScaled(bins=None)`
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

`KneeLocalizer(svm_model_path, PIL_cls, resize=None, np_input=False, np_output=False, debug=False)`
:   Transformation which finds out where is the knee and cut out the rest of the image
    Based on code from https://github.com/MIPT-Oulu/KneeLocalizer
    
    ```
    @inproceedings{tiulpin2017novel,
        title={A novel method for automatic localization of joint area on knee plain radiographs},
        author={Tiulpin, Aleksei and Thevenot, Jerome and Rahtu, Esa and Saarakkala, Simo},
        booktitle={Scandinavian Conference on Image Analysis},
        pages={290--301},
        year={2017},
        organization={Springer}
    }
    ```

    ### Ancestors (in MRO)

    * fastcore.transform.Transform

    ### Methods

    `decodes(...)`
    :   Dictionary-like object; `__getitem__` matches keys of types using `issubclass`

    `encodes(...)`
    :   Dictionary-like object; `__getitem__` matches keys of types using `issubclass`

    `get_joint_x_proposals(self, img, av_points=2, av_derv_points=11, margin=0.25, step=10)`
    :   Return X-coordinates of the bone approximate locations

    `get_joint_y_proposals(self, img, av_points=2, av_derv_points=11, margin=0.25, step=10)`
    :   Return Y-coordinates of the joint approximate locations.

    `setups(...)`
    :   Dictionary-like object; `__getitem__` matches keys of types using `issubclass`

    `smooth_line(self, line, av_points)`
    :

`XRayPreprocess(PIL_cls, cut_min=5.0, cut_max=99.0, only_non_zero=True, scale=True, np_input=False, np_output=False)`
:   Preprocess the X-ray image using histogram clipping and global contrast normalization.

    ### Ancestors (in MRO)

    * fastcore.transform.Transform

    ### Methods

    `decodes(...)`
    :   Dictionary-like object; `__getitem__` matches keys of types using `issubclass`

    `encodes(...)`
    :   Dictionary-like object; `__getitem__` matches keys of types using `issubclass`

    `setups(...)`
    :   Dictionary-like object; `__getitem__` matches keys of types using `issubclass`