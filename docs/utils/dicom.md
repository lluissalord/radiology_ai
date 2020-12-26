Module radiology_ai.utils.dicom
===============================
DICOM utilities for checking/filtering depending on metadata

Functions
---------

    
`check_DICOM(dcm, check_DICOM_dict=None, debug=False)`
:   Check the DICOM file if it is has the feature required

    
`default_check_DICOM_dict()`
:   Get default values for check DICOM dictionary

    
`df_check_DICOM(df, check_DICOM_dict)`
:   Filter DataFrame on the rows that match the filter specified on `check_DICOM_dict`

    
`filter_fnames(fnames, metadata_raw_path, check_DICOM_dict={'Modality': ['CR', 'DR', 'DX']})`
:   Filter all the filenames which are fulfill the metadata conditions passed on `check_DICOM_dict`

    
`get_each_case_samples(df, all_keys, max_samples_per_case=5)`
:   Extract all the different cases on `all_keys` and creating a DataFrame with the number set for all the cases

    
`sample_df_check_DICOM(df, check_DICOM_dict, max_samples_per_case=5)`
:   Get a random sample of the filtered DataFrame determined by `check_DICOM_dict` appearing all the cases there