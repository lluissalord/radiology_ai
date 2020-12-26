Module radiology_ai.preprocessing.misc
======================================
Miscellaneous preprocessings

Functions
---------

    
`dcm_scale(dcm)`
:   Transform from raw pixel data to scaled one and inversing (if the case)

    
`dcmread_scale(fn)`
:   Transform from path of raw pixel data to scaled one and inversing (if the case)

    
`init_bins(fnames, n_samples=None, isDCM=True)`
:   Initialize bins to equally distribute the histogram of the dataset