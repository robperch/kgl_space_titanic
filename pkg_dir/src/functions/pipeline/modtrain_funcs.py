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

## Saving the results obtained from the model's magic loop as a pickle
def save_magic_loop_results(models_magic_loop):
    """
    Saving the results obtained from the model's magic loop as a pickle

    :param models_magic_loop: (dictionary) dict containing the best model per type based on the specified hyper-parameters
    :return None:
    """


    ## Saving the magic loop results

    ### Path where the pickle will be stored locally
    pkl_path = os.path.join(
        pipeline_pkl_modtrain_local_dir,
        pipeline_pkl_modtrain_name,
    ) + '_model_ml.pkl'

    ### Saving the object locally as pickle
    pickle.dump(
        models_magic_loop,
        open(pkl_path, 'wb')
    )

    ### Path where the pickle object will be stored on AWS' S3
    obj_name = os.path.join(
        pipeline_pkl_modtrain_aws_key,
        pipeline_pkl_modtrain_name,
    ) + '_model_ml.pkl'

    ### Saving object in AWS S3
    upload_file_to_s3(pkl_path, base_bucket_name, object_name=obj_name)


    ## Saving the dataset objects


    return



## Saving module results
def save_modtrain_results(dataset_dict, models_magic_loop):
    """
    Saving module results

    :param dataset_dict: (dictionary) dict containing all the dataset objects (e.g. train_x, train_y, test_x, test_y)
    :param models_magic_loop: (dictionary) dict containing the best model per type based on the specified hyper-parameters
    :return None:
    """


    ## Creating directory for local pickles if not existent
    create_directory_if_nonexistent(pipeline_pkl_modtrain_local_dir)

    ## Saving locally the dataset objects as pickles
    save_dataset_objects_locally(
        dataset_dict,
        pipeline_pkl_modtrain_local_dir,
        pipeline_pkl_modtrain_name
    )

    ## Saving in the cloud the dataset objects that were locally saved as pickles
    save_dataset_objects_in_cloud(
        dataset_dict,
        pipeline_pkl_modtrain_local_dir,
        cloud_provider,
        base_bucket_name,
        pipeline_pkl_modtrain_aws_key,
        pipeline_pkl_modtrain_name
    )

    ## Saving module results
    save_magic_loop_results(models_magic_loop)


    return



"--------------- Compounded functions ---------------"

## Models training pipeline function
def modtrain_pipeline_func():
    """
    Models training pipeline function

    :return:
    """

    ## Saving dataset objects from the 'feateng' step of the pipeline in a dictionary data structure
    dataset_dict = dataset_objects_dict(pipeline_pkl_feateng_local_dir)

    ## Magic loop: iterating over various models and hyper-parameters to find best parameters
    models_magic_loop = models_training_magic_loop(
        predict_models_dict,
        dataset_dict['X_train'],
        dataset_dict['y_train'],
        predict_model_eval_metric
    )

    ## Saving module results
    save_modtrain_results(dataset_dict, models_magic_loop)


    return





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
