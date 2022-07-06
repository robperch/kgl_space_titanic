## MODULE TO APPLY FEATURE ENGINEERING TO DATA





"----------------------------------------------------------------------------------------------------------------------"
############################################# Imports ##################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--- Standard library imports ---"
import pickle
import os

"--- Third party imports ---"
import numpy as np

"--- Local application imports ---"
from pkg_dir.config import *
from pkg_dir.src.utils import *
from pkg_dir.src.parameters import *





"----------------------------------------------------------------------------------------------------------------------"
############### Feature engineering pipeline functions #################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--------------- Unitary functions ---------------"

## Adding new features to dataset
def adding_new_features(dfx):
    """
    Adding new features to dataset

    :param dfx: (pd.DataFrame) data that will be enhanced with new features
    :return dfx: (pd.DataFrame) data with new features added
    """


    ## Inserting column with total amount of spent money in luxury amenities

    ### Expenses columns
    exp_cols = [
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck',
    ]

    ### Inserting new column
    dfx.insert(
        dfx.columns.tolist().index('VRDeck') + 1,
        'ExpensesSum',
        dfx.loc[:, exp_cols].sum(axis=1)
    )

    ### Data schema of the new feature
    new_feat_data_schema = {
        'relevant': True,
        'clean_col_name': 'ExpensesSum',
        'data_type': 'float',
        'feature_type': 'numerical',
        'model_relevant': True,
        'note': 'feature added from base features',
    }

    ### Updating and saving data schema with new feature
    update_save_data_schema(
        titanicsp_full_data_schema,
        'ExpensesSum',
        new_feat_data_schema,
        local_json_path,
        json_name
    )


    return dfx



## Dropping features irrelevant for the model
def dropping_irrelevant_model_features(dfx):
    """
    Dropping features irrelevant for the model

    :param dfx: (pd.DataFrame) dataframe with all features, including the recently added
    :return dfx: (pd.DataFrame) dataframe only with the features that are relevant for the model
    """


    ## List of features that will be fed to the model
    model_features = [
        feat
        for feat in titanicsp_full_data_schema
        if
        'model_relevant' in titanicsp_full_data_schema[feat]
        and
        titanicsp_full_data_schema[feat]['model_relevant']
        and
        feat in dfx.columns
    ]

    ## Leaving only features relevant for the model
    dfx = dfx.loc[:, model_features].copy()


    return dfx



## Imputing feature values based on pre-defined rules
def imput_feature_values(dfx):
    """
    Imputing feature values based on pre-defined rules

    :param dfx: (pd.DataFrame) df with features containing missing values
    :return dfx: (pd.DataFrame) df with missing values imputed
    """


    ## Segmenting features by type to process them through pipeline
    feat_imputation_dict = features_list_dict(dfx, titanicsp_full_data_schema, 'imputation_strategy')

    ## Applying standard imputation
    dfx = apply_imputations(dfx, feat_imputation_dict, np.nan, 0)


    return dfx



## Building list of tuples to feed the data processing pipeline
def processing_data_through_pipeline(dfx):
    """
    Building list of tuples to feed the data processing pipeline

    :param dfx: (pd.DataFrame) df with raw and added features before being adjusted for the model
    :return data_ppl_tuples: (list) list containing the tuples needed for sklearn's ColumnTransformer
    """


    ## Segmenting features by type to process them through pipeline
    feat_type_dict = features_list_dict(dfx, titanicsp_full_data_schema, 'feature_type')

    ## Building list of tuples to feed the data processing pipeline
    data_ppl_tuples = [
        ('categorical', categorical_ppl, feat_type_dict['categorical']),
        ('numerical', numerical_ppl, feat_type_dict['numerical']),
    ]

    ### Applying pipeline based on provided tuples
    dfx = apply_data_ppl_with_tuples(dfx, data_ppl_tuples)


    return dfx



## Applying feature engineering functions
def feature_engineering(dfx):
    """
    Applying feature engineering functions

    :param dfx: (pd.DataFrame) df with raw and added features before being adjusted for the model
    :return dfx: (pd.DataFrame) df with features adjusted for the model
    """


    ## Inputting nan values
    dfx = imput_feature_values(dfx)


    ## Building list of tuples to feed the data processing pipeline
    dfx = processing_data_through_pipeline(dfx)


    return dfx



## Saving module results
def save_feateng_results(dataset_dict):
    """
    Saving module results

    :param dataset_dict: (dictionary) dict containing all the dataset objects (e.g. train_x, train_y, test_x, test_y)
    :return None:
    """


    ## Creating directory for local pickles if not existent
    create_directory_if_nonexistent(pipeline_pkl_feateng_local_dir)

    ## Saving locally the dataset objects as pickles
    save_dataset_objects_locally(
        dataset_dict,
        pipeline_pkl_feateng_local_dir,
        pipeline_pkl_feateng_name,
    )

    ## Saving in the cloud the dataset objects that were locally saved as pickles
    save_dataset_objects_in_cloud(
        dataset_dict,
        pipeline_pkl_feateng_local_dir,
        cloud_provider,
        base_bucket_name,
        pipeline_pkl_feateng_aws_key,
        pipeline_pkl_feateng_name,
    )


    return



"--------------- Compounded functions ---------------"

## Feature engineering pipeline function
def feateng_pipeline_func():
    """
    Feature engineering pipeline function

    :return None:
    """


    ## Saving dataset objects in a dictionary data structure
    dataset_dict = dataset_objects_dict(pipeline_pkl_transform_local_dir)

    ## Leaving only data objects that contain features, not labels
    transform_features_datasets = [
        tr_obj
        for tr_obj in dataset_dict
        if 'X_' in tr_obj
    ]


    ## Iterating over every extract object and applying the wrangling functions
    for transform_obj in transform_features_datasets:

        ## Saving the dataset object in variable
        dfx = dataset_dict[transform_obj]

        ## Adding new features
        dfx = adding_new_features(dfx)

        ## Dropping features irrelevant for the model
        dfx = dropping_irrelevant_model_features(dfx)

        ## Applying feature engineering functions
        dfx = feature_engineering(dfx)

        ## Updating the dataset object stored in the dataset dictionary with result obtained
        dataset_dict.update({transform_obj: dfx})


    ## Saving module results
    save_feateng_results(dataset_dict)


    return





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
