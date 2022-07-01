## MODULE TO APPLY FEATURE ENGINEERING TO DATA





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



## Applying feature engineering functions
def feature_engineering(dfx):
    """
    Applying feature engineering functions

    :param dfx: (pd.DataFrame) df with raw and added features before being adjusted for the model
    :return dfx: (pd.DataFrame) df with features adjusted for the model
    """


    ## Segmenting features by type to process them through pipeline

    ### Categorical features
    categorical_feats = [
        feat
        for feat in titanicsp_full_data_schema
        if
        'feature_type' in titanicsp_full_data_schema[feat]
        and
        titanicsp_full_data_schema[feat]['feature_type'] == 'categorical'
        and
        feat in dfx.columns
    ]

    ### numerical features
    numerical_feats = [
        feat
        for feat in titanicsp_full_data_schema
        if
        'feature_type' in titanicsp_full_data_schema[feat]
        and
        titanicsp_full_data_schema[feat]['feature_type'] == 'numerical'
        and
        feat in dfx.columns
    ]


    ## Applying the data processing pipeline to the features

    ### Building list of tuples to feed the data processing pipeline
    data_ppl_tuples = [
        ('categorical', categorical_ppl, categorical_feats),
        ('numerical', numerical_ppl, numerical_feats),
    ]

    ### Applying pipeline based on provided tuples
    dfx = apply_data_ppl_with_tuples(dfx, data_ppl_tuples)


    return dfx



"--------------- Compounded functions ---------------"

## Feature engineering pipeline function
def feateng_pipeline_func():
    """
    Feature engineering pipeline function

    :return None:
    """


    ## Listing the objects obtained after de 'extract' step of the pipeline and saved locally
    transform_objects = os.listdir(pipeline_pkl_transform_local_dir)

    ## Leaving only data objects that contain features, not labels
    transform_objects = [
        tr_obj
        for tr_obj in transform_objects
        if '_x.' in tr_obj
    ]


    ## Iterating over every extract object and applying the wrangling functions
    for transform_obj in transform_objects:

        ## Reading the object's content from the locally saved pickle
        with open(pipeline_pkl_transform_local_dir + transform_obj, 'rb') as obj:
            dfx = pickle.load(obj)

        ## Adding new features
        dfx = adding_new_features(dfx)

        ## Dropping features irrelevant for the model
        dfx = dropping_irrelevant_model_features(dfx)

        ## Applying feature engineering functions
        dfx = feature_engineering(dfx)

        ## Saving dataschema as json after feature engineering

        ## Saving results locally as pickles

        ## Saving results in AWS S3 as pickles


    return





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
