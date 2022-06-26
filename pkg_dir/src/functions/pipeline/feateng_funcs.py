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


    return dfx



## Dropping features irrelevant for the model
def dropping_irrelevant_model_features(dfx):
    """
    Dropping features irrelevant for the model

    :param dfx: (pd.DataFrame) dataframe with all features, including the recently added
    :return dfx: (pd.DataFrame) dataframe only with the features that are relevant for the model
    """


    ##


    return





"--------------- Compounded functions ---------------"

## Feature engineering pipeline function
def feateng_pipeline_func():
    """
    Feature engineering pipeline function

    :return None:
    """


    ## Listing the objects obtained after de 'extract' step of the pipeline and saved locally
    transform_objects = os.listdir(pipeline_pkl_transform_local_dir)


    ## Iterating over every extract object and applying the wrangling functions
    for transform_obj in transform_objects:

        ## Setting key to identify the object
        obj_key = discern_between_train_and_test(transform_objects)

        ## Reading the object's content from the locally saved pickle
        with open(pipeline_pkl_extract_local_dir + transform_obj) as obj:
            dfx = pickle.load(obj)

        ## Adding new features
        dfx = adding_new_features(dfx)

        ## Dropping features irrelevant for the model
        dfx = dropping_irrelevant_model_features(dfx)

        ## Applying feature engineering functions

        ## Saving dataschema as json after feature engineering

        ## Saving results locally as pickles

        ## Saving results in AWS S3 as pickles


    return





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
