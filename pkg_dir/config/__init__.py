## INITIALIZATION MODULE - CONFIGURATION


## Imports
from .config import (

    package_dir,
    creds_file_path,
    utc_tz,
    mexico_tz,

    ## Dataset files paths
    dataset_name,
    dataset_local_files,
    dataset_dir,
    training_dataset,
    test_dataset,

    ## Pickles files path
    pickles_dir_path,
    pipeline_pkl_extract_local_dir,
    pipeline_pkl_extract_name,
    pipeline_pkl_transform_name,
    pipeline_pkl_transform_local_dir,
    pipeline_pkl_feateng_name,
    pipeline_pkl_feateng_local_dir,
    pipeline_pkl_modtrain_name,
    pipeline_pkl_modtrain_local_dir,
    pipeline_pkl_modsel_name,
    pipeline_pkl_modsel_local_dir,
    pipeline_pkl_bifair_name,
    pipeline_pkl_bifair_local_dir,
    pipeline_pkl_predict_name,
    pipeline_pkl_predict_local_dir,

    ## AWS S3 paths
    base_bucket_name,
    pipeline_pkl_extract_aws_key,
    pipeline_pkl_transform_aws_key,
    pipeline_pkl_feateng_aws_key,
    pipeline_pkl_modtrain_aws_key,

)





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"

