import os
import pandas as pd

from tqdm.auto import tqdm

def filter_fnames(
    fnames,
    metadata_raw_path,
    check_DICOM_dict={
        'Modality': ['CR', 'DR', 'DX'],
    }
):
    """ Filter all the filenames which are fulfill the metadata conditions passed on `check_DICOM_dict` """

    # Load metadata 
    metadata_df = pd.read_csv(metadata_raw_path)

    # Filter by the ones taht fulfill the conditions
    metadata_df = df_check_DICOM(metadata_df, check_DICOM_dict)

    # Create a DataFrame to compare filenames
    check_df = pd.DataFrame(metadata_df.fname.str.split('/').str[-1], index=metadata_df.fname.str.split('/').str[-1])

    # Loop over all the filenames
    filter_fnames = []
    for fname in tqdm(fnames):
        try:
            filename, ext = os.path.splitext(fname.name)
        except AttributeError:
            fname = Path(fname)
            filename, ext = os.path.splitext(fname.name)

        # Take into account the ones which seems to have extension ".PACSXXX" is not an extension
        if ext.startswith('.PACS'):
            filename = fname.name

        # Check if is in the list and add it
        try:
            check_df.loc[filename, :]
            filter_fnames.append(str(fname).replace(os.sep, '/'))
        except KeyError:
            continue
    
    return filter_fnames


def default_check_DICOM_dict():
    """ Get default values for check DICOM dictionary """

    check_DICOM_dict = {
        'SeriesDescription': ['RODILLA AP', 'RODILLAS AP'],
        'BodyPartExamined': ['LOWER LIMB', 'KNEE']
    }

    return check_DICOM_dict


def df_check_DICOM(df, check_DICOM_dict):
    """ Filter DataFrame on the rows that match the filter specified on `check_DICOM_dict` """
    match = True
    for key, value in check_DICOM_dict.items():
        # Lambda function checking if DICOM file fulfills the condition
        if key == 'function':
            match = (match) & df.apply(value, axis=1)
        else:
            match = (match) & df[key].isin(value)
    
    return df[match]


def sample_df_check_DICOM(df, check_DICOM_dict, max_samples_per_case=5):
    """ Get a random sample of the filtered DataFrame determined by `check_DICOM_dict` appearing all the cases there """

    df_match = df_check_DICOM(df, check_DICOM_dict)
    
    all_keys = list(check_DICOM_dict.keys())

    return get_each_case_samples(df_match, all_keys, max_samples_per_case)


def get_each_case_samples(df, all_keys, max_samples_per_case=5):
    """ Extract all the different cases on `all_keys` and creating a DataFrame with the number set for all the cases """
    cases_df = df[all_keys].drop_duplicates()
    
    concat_list = []
    for i in range(len(cases_df.index)):
        concat_list.append(
            df[
                (
                    df[all_keys] == cases_df.iloc[i][all_keys]
                ).all(axis=1)
            ].sample(max_samples_per_case, replace=True)
        )

    return pd.concat(concat_list)


def check_DICOM(dcm, check_DICOM_dict=None, debug=False):
    """ Check the DICOM file if it is has the feature required """

    if check_DICOM_dict is None:
        check_DICOM_dict = default_check_DICOM_dict()

    check = True
    for key, value in check_DICOM_dict.items():
        # Lambda function checking if DICOM file fulfills the condition
        if key == 'function':
            check = value(dcm)
            if not check:
                break
        elif dcm.get(key) not in value:
            check = False
            if debug:
                print(f'{key}: {dcm.get(key)} on Accession Number: {dcm.AccessionNumber}')
            break
    return check