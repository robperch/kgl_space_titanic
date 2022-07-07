## MODULE TO SELECT THE BEST MODEL FOR THE PREDICTION





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
############### Model selection pipeline functions #####################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--------------- Unitary functions ---------------"

##



"--------------- Compounded functions ---------------"

## Models evaluation and selection pipeline function
def modevalsel_pipeline_func():
    """
    Models evaluation and selection pipeline function

    :return:
    """

    ## Saving dataset objects from the 'feateng' step of the pipeline in a dictionary data structure
    dataset_dict = dataset_objects_dict(pipeline_pkl_modtrain_local_dir)

    ## Magic loop: iterating over various models and hyper-parameters to find best parameters

    # ## Saving module results
    # save_modtrain_results(dataset_dict, models_magic_loop)


    return






"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
