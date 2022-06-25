## MODULE TO TRANSFORM DATA





"----------------------------------------------------------------------------------------------------------------------"
############################################# Imports ##################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--- Standard library imports ---"
import pickle
import os

"--- Third party imports ---"

"--- Local application imports ---"
from pkg_dir.config import *
from pkg_dir.src.utils import *
from pkg_dir.src.parameters import *





"----------------------------------------------------------------------------------------------------------------------"
############### Downloading dataset from Kaggle ########################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--------------- Unitary functions ---------------"

## Setting the id feature as the index
def set_id_feature_as_index(dfx):
    """
    Setting the id feature as the index

    :param dfx: (pd.DataFrame) data after the wrangling process
    :return dfx: (pd.DataFrame) data with the id feature set as index
    """


    ## Id feature
    id_feature = [
        col
        for col in titanicsp_data_schema
        if 'id_feature' in titanicsp_data_schema[col]
    ][0]

    ## Set id feature as index
    dfx.set_index(id_feature, inplace=True)


    return dfx



## Saving the transform pickles locally
def save_transform_local_df_pkl(obj_key, dfx):
    """
    Saving the transform pickles locally

    :param obj_key: (string) identifier to differentiate if it's the training or testing dataset
    :param dfx: (pd.DataFrame) dataframe with either the test or train info
    :return None:
    """


    ## Differentiate between the transform datasets (train or test)
    if obj_key == 'train':

        ## Saving features and labels separately

        ### Labels dataframe
        df_labels = dfx.loc[:, 'label'].copy()

        ### Saving labels dataframe as a pickle locally
        pkl_path = os.path.join(
            pipeline_pkl_transform_local_dir,
            pipeline_pkl_transform_name
        ) + '_' + obj_key + '_y.pkl'
        pickle.dump(df_labels, open(pkl_path, 'wb'))

        ### Features dataframe
        df_features = dfx.drop('label', axis=1)

        ### Saving features dataframe as a pickle locally
        pkl_path = os.path.join(
            pipeline_pkl_transform_local_dir,
            pipeline_pkl_transform_name
        ) + '_' + obj_key + '_x.pkl'
        pickle.dump(df_features, open(pkl_path, 'wb'))


    elif obj_key == 'test':

        ## Saving test dataframe locally
        pkl_path = os.path.join(
            pipeline_pkl_transform_local_dir,
            pipeline_pkl_transform_name
        ) + '_' + obj_key + '_x.pkl'
        pickle.dump(dfx, open(pkl_path, 'wb'))


    return



## Saving the transform pickles locally
def save_transform_aws_df_pkl(obj_key, dfx):
    """
    Saving the transform pickles locally

    :param obj_key: (string) identifier to differentiate if it's the training or testing dataset
    :param dfx: (pd.DataFrame) dataframe with either the test or train info
    :return None:
    """


    ## Differentiate between the transform datasets (train or test)
    if obj_key == 'train':

        ## Saving features

        ### Function parameters

        file_path = os.path.join(
            pipeline_pkl_transform_local_dir,
            pipeline_pkl_transform_name
        ) + '_' + obj_key + '_x.pkl'

        object_name = os.path.join(
            pipeline_pkl_transform_aws_key,
            pipeline_pkl_transform_name
        ) + '_' + obj_key + '_x.pkl'


        ### Uploading file

        upload_file_to_s3(file_path, base_bucket_name, object_name)


        ## Saving labels

        ### Function parameters

        file_path = os.path.join(
            pipeline_pkl_transform_local_dir,
            pipeline_pkl_transform_name
        ) + '_' + obj_key + '_y.pkl'

        object_name = os.path.join(
            pipeline_pkl_transform_aws_key,
            pipeline_pkl_transform_name
        ) + '_' + obj_key + '_y.pkl'


        ### Uploading file

        upload_file_to_s3(file_path, base_bucket_name, object_name)



    elif obj_key == 'test':

        ## Function parameters

        file_path = os.path.join(
            pipeline_pkl_transform_local_dir,
            pipeline_pkl_transform_name
        ) + '_' + obj_key + '_x.pkl'

        object_name = os.path.join(
            pipeline_pkl_transform_aws_key,
            pipeline_pkl_transform_name
        ) + '_' + obj_key + '_x.pkl'


        ## Uploading file

        upload_file_to_s3(file_path, base_bucket_name, object_name)


    return



"--------------- Compounded functions ---------------"

## Transform pipeline function
def transform_pipeline_func():
    """
    Transform pipeline function

    :return None:
    """


    ## Listing the objects obtained after de 'extract' step of the pipeline and saved locally
    extract_objects = os.listdir(pipeline_pkl_extract_local_dir)


    ## Iterating over every extract object and applying the wrangling functions
    for extract_obj in extract_objects:

        ## Setting key to identify the object
        obj_key = discern_between_train_and_test(extract_obj)

        ## Reading the object's content
        dfx = read_s3_obj_to_variable(base_bucket_name, pipeline_pkl_extract_aws_key, extract_obj)

        ## Apply data wrangling functions based on a predefined dataschema
        dfx = data_wrangling_schema_functions(dfx, titanicsp_data_schema)

        ## Setting the id feature as the index
        dfx = set_id_feature_as_index(dfx)

        ## Creating directory for local pickles
        create_directory_if_nonexistent(pipeline_pkl_transform_local_dir)

        ## Saving the transform pickles locally
        save_transform_local_df_pkl(obj_key, dfx)

        ## Saving object a AWS bucket
        save_transform_aws_df_pkl(obj_key, dfx)


    return





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
