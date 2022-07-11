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



## Saving table with the evaluation metrics results
def save_eval_metrics_results(metrics_table):
    """
    Saving table with the evaluation metrics results

    :param metrics_table: (pd.Dataframe) table with a summary of the trained models performance with the validation dataset
    :return None:
    """


    ## Path where the pickle will be stored locally
    pkl_path = os.path.join(
        pipeline_pkl_modevalsel_local_dir,
        pipeline_pkl_modevalsel_name,
    ) + '_metrics.pkl'

    ## Saving the object locally as pickle
    pickle.dump(
        metrics_table,
        open(pkl_path, 'wb')
    )

    ## Path where the pickle object will be stored on AWS' S3
    obj_name = os.path.join(
        pipeline_pkl_modevalsel_aws_key,
        pipeline_pkl_modevalsel_name,
    ) + '_metrics.pkl'

    ## Saving object in AWS S3
    upload_file_to_s3(pkl_path, base_bucket_name, object_name=obj_name)


    return



## Saving module results
def save_modevalsel_results(dataset_dict, metrics_table):
    """
    Saving module results

    :param dataset_dict: (dictionary) dict containing all the dataset objects (e.g. train_x, train_y, test_x, test_y)
    :param metrics_table: (pd.DataFrame or similar) table with a summary of the evaluation metrics made on the validation dataset
    :return None:
    """


    ## Creating directory for local pickles if not existent
    create_directory_if_nonexistent(pipeline_pkl_modevalsel_local_dir)

    ## Saving locally the dataset objects as pickles
    save_dataset_objects_locally(
        dataset_dict,
        pipeline_pkl_modevalsel_local_dir,
        pipeline_pkl_modevalsel_name
    )

    ## Saving in the cloud the dataset objects that were locally saved as pickles
    save_dataset_objects_in_cloud(
        dataset_dict,
        pipeline_pkl_modevalsel_local_dir,
        cloud_provider,
        base_bucket_name,
        pipeline_pkl_modevalsel_aws_key,
        pipeline_pkl_modevalsel_name
    )

    ## Saving table with the evaluation metrics results
    save_eval_metrics_results(metrics_table)


    return



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
    ## TBD

    ## Saving module results
    save_modevalsel_results(dataset_dict, metrics_table)


    return






"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
