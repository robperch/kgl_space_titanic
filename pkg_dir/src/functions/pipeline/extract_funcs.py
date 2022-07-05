## MODULE TO EXTRACT DATA FROM SOURCE





"----------------------------------------------------------------------------------------------------------------------"
############################################# Imports ##################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--- Standard library imports ---"
import zipfile
import pickle
import os

"--- Third party imports ---"
import kaggle
import pandas as pd

"--- Local application imports ---"
from pkg_dir.config import *
from pkg_dir.src.utils import (

    create_directory_if_nonexistent,
    upload_file_to_s3,

)





"----------------------------------------------------------------------------------------------------------------------"
############### Extract general functions ##############################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--------------- Unitary functions ---------------"

## Downloading data from Kaggle if it's not present in the project's dir
def download_data_if_none():
    """
    Downloading data from Kaggle if it's not present in the project's dir

    :return:
    """


    ## Checking if the data file is in local directory
    if dataset_name in os.listdir(package_dir + '/data' + '/dataset'):
        print("Dataset already present locally... skipping download...")

    else:
        print("Dataset not present locally... downloading from source")

        ## Downloading dataset with kaggle's api
        kaggle.api.competition_download_files(
            'spaceship-titanic',
            path=dataset_dir,
        )

        ## Unzipping dir
        zip_file = dataset_dir + '/' + dataset_name + '.zip'
        dump_dir = os.path.join(dataset_dir, dataset_name)

        os.mkdir(dump_dir)

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dump_dir)


    return



## Saving locally train and test dataset as df-pickle
def save_extract_local_df_pkl():
    """
    Saving locally train and test dataset as df-pickle

    :return None:
    """


    ## Ensuring that the directory where the pickle will be saved exists
    create_directory_if_nonexistent(pipeline_pkl_extract_local_dir)

    ## Reading csv file as a df
    for file in os.listdir(dataset_local_files):

        ## Condition to skip unnecessary file
        if file != 'sample_submission.csv':

            ## Data prefix used in pickle name
            if 'train' in file:
                data_prefix = '_Xy_'
            elif 'test' in file:
                data_prefix = '_X_'

            ## Reading csv file as pandas dataframe
            dfx = pd.read_csv(os.path.join(dataset_local_files, file))

            ## Saving df as pickle and storing it locally
            pickle.dump(
                dfx,
                open(
                    os.path.join(pipeline_pkl_extract_local_dir, pipeline_pkl_extract_name) + data_prefix + file.split(sep='.')[0] + '.pkl',
                    'wb'
                )
            )


    return



## Saving local extract pickles in AWS S3
def save_extract_pkl_s3():
    """
    Saving local extract pickles in AWS S3

    :return None:
    """


    ## Iterating over every pickle to store it in s3
    for pkl in os.listdir(pipeline_pkl_extract_local_dir):

        ## Function parameters
        file_path = os.path.join(pipeline_pkl_extract_local_dir, pkl)
        object_name = os.path.join(pipeline_pkl_extract_aws_key, pkl)

        ## Function execution
        upload_file_to_s3(file_path, base_bucket_name, object_name)


    return



"--------------- Compounded functions ---------------"


## Extract pipeline function
def extract_pipeline_func():
    """
    Extract pipeline function

    :return None:
    """


    ## Downloading data from Kaggle if it's not present in the project's dir
    download_data_if_none()

    ## Saving locally train and test dataset as df-pickle
    save_extract_local_df_pkl()

    ## Saving local extract pickles in AWS S3
    save_extract_pkl_s3()


    return





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
