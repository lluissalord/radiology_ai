Module radiology_ai.utils.organize
==================================
Functions required for organizing files for proper annotation

Functions
---------

    
`add_new_relation(relation_df, src_path, src_filename, new_filename)`
:   Add a new relation on the relation DataFrame

    
`check_data_in_template(template_file, sep=None)`
:   Check if there is data different than null in each template

    
`check_generic_path(path)`
:   Check if path exists even if folder name modified adding suffix

    
`check_metadata_label(raw_path, metadata_labels, label='ap')`
:   Check if the path matches with the desired label on metadata_labels

    
`check_relation(relation_df, check_path=True, check_raw=True)`
:   Check that the relation on the relation DataFrame is preserved

    
`concat_templates(src_folder, excel=True, csv_sep=';')`
:   Concatenate all the template into a DataFrame

    
`expand_list(l, n)`
:   Expand a list `l` to repeat its elements till reaching length of `n`

    
`generate_template(dst_folder, groups, subgroup_length, filename_prefix='IMG_', excel=True, csv_sep=';', able_overwrite=False)`
:   Generates template files for each group of DICOM files using the filename as ID

    
`get_final_dst(dst_folder, filepaths, groups, subgroup_length, start_num_subgrups=None)`
:   Relates source filepaths with destination paths depending on shuffle groups and subgroup length

    
`get_last_id(relation_df, prefix='IMG_')`
:   Get the ID of the last filename from current relation of files

    
`modify_template(dst_folder, modify_func, groups, subgroup_length, excel=True, csv_sep=';')`
:   Modify all the templates on dst_folder applying `modify_func` on each DataFrame

    
`move_blocks(parent_folder, new_folder, blocks, relation_filepath, template_extension='xlsx')`
:   Move blocks of files to the new folder (person name)

    
`move_file(src_filepath, filename, dst_folder, force_extension=None, copy=True, return_filepath=True)`
:   Move or copy file to the destination folder with folder name

    
`move_relation(relation_filepath, copy=True, to_raw=True)`
:   Move or copy files to/from raw destination from/to final destination based on relation file

    
`open_name_relation_file(filepath, sep=',')`
:   Extract DataFrame from file containing the relation between original and new files

    
`organize_folders(src_folder, dst_folder, relation_filepath, reset=False, groups=None, subgroup_length=None, filename_prefix='IMG_', force_extension=None, copy=True, metadata_labels=None, check_DICOM_dict=None, debug=False)`
:   Organize folders and files to set all the desired DICOM files into the correct folder

    
`read_template(template_file, sep=None, dtype=None)`
:   Read template idenpendently of the extension

    
`rename_patient(dicom_files)`
:   Modify metadata regarding Patient's Name and Patient's ID to set them as the filename

    
`save_name_relation_file(relation_df, filepath, sep=',')`
:   Save file containing the relation between original and new files

    
`shuffle_group_folders(folders, groups, subgroup_length=None, start_num_subgrups=None)`
:   Generate dictionaries of the shuffled groups and subgroups related to folders

    
`sort_template_file(df, filename_prefix)`
:   Sort DataFrame based on template filename

    
`update_block_relation(relation_df, parent_folder, block, new_folder, sep='/')`
:   Replace the old folder names by the new folder only to the paths where the block appears