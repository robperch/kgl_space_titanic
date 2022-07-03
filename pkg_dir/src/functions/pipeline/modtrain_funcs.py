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
############### Models training pipeline functions #####################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--------------- Unitary functions ---------------"

##



"--------------- Compounded functions ---------------"

## Models training pipeline function
def modtrain_pipeline_func():
    """
    Models training pipeline function

    :return:
    """

    ## Saving dataset objects from the 'feateng' step of the pipeline in a dictionary data structure
    dataset_dir = dataset_objects_dict(pipeline_pkl_feateng_local_dir)

    ## Magic loop: iterating over various models and hyper-parameters to find best parameters
    models_magic_loop = models_training_magic_loop(
        predict_models_dict,
        dataset_dir['train_x'],
        dataset_dir['train_y'],
        predict_model_eval_metric
    )


    return






"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
