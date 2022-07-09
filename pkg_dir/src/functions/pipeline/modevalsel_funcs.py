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

## Loading the trained models stored
def load_trained_models():
    """
    Loading the trained models stored

    :return modtrain_res: (dictionary) dict containing the best estimator and score for a specific type of model
    """


    pkl_obj = 'modtrain_model_ml.pkl'

    with open(pipeline_pkl_modtrain_local_dir + pkl_obj, 'rb') as obj:
        modtrain_res = pickle.load(obj)


    return modtrain_res



"--------------- Compounded functions ---------------"

## Models evaluation and selection pipeline function
def modevalsel_pipeline_func():
    """
    Models evaluation and selection pipeline function

    :return:
    """

    ## Saving dataset objects from the 'modtrain' step of the pipeline in a dictionary data structure
    dataset_dict = dataset_objects_dict(pipeline_pkl_modtrain_local_dir)

    ## Loading the trained models stored
    modtrain_res = load_trained_models()

    ## Add the trained models' predictions for the validation labels
    dataset_dict = add_validation_predictions_per_model(dataset_dict, modtrain_res)

    ## Generate a summary performance metrics table for every model with the validation dataset
    metrics_table = validation_models_performance_table(dataset_dict, modtrain_res, model_eval_metrics)

    ## Generate performance visualizations for every model with the validation dataset

    # ## Saving module results
    # save_modtrain_results(dataset_dict, models_magic_loop)


    return






"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
