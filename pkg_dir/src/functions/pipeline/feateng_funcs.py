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

## Xxx




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
        with open(pipeline_pkl_extract_local_dir + transform_obj) as object:
            dfx = pickle.load(object)


    return





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
