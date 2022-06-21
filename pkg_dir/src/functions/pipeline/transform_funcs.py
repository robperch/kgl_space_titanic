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



"--------------- Compounded functions ---------------"

## Transform pipeline function
def transform_pipeline_func(bucket_name, extract_bucket_key):
    """
    Extract pipeline function

    :param bucket_name: (string) name of the bucket where the objects obtained from the 'extract' step are
    :param extract_bucket_key: (string) key to locate the objects obtained from the 'extract' step
    :return df_trans_train: (pd.DataFrame) df with the training data after the 'transform' step of the pipeline
    :return df_trans_test: (pd.DataFrame) df with the test data after the 'transform' step of the pipeline
    """


    ## Listing the objects obtained after the 'extract' step of the pipeline
    extract_objects = list_objects_in_bucket_key(bucket_name, extract_bucket_key)


    ## Iterating over every extract object and applying the wrangling functions

    ### Initializing dictionary where the results will be temporarily saved
    res_dict = {}

    ### Loop to obtain results and save them in dictionary
    for extract_obj in extract_objects:

        ## Setting key to identify the object
        if 'train' in extract_obj:
            obj_key = 'train'
        elif 'test' in extract_obj:
            obj_key = 'test'
        else:
            raise NameError('No keyword was identified in the object')

        ## Reading the object's content
        obj_var = read_s3_obj_to_variable(bucket_name, extract_bucket_key, extract_obj)

        ## Apply data wrangling functions based on a predefined dataschema
        res_dict[obj_key] = data_wrangling_schema_functions(obj_var, titanicsp_data_schema)

        ## Saving object locally as pickle
        pkl_path = os.path.join(pipeline_pkl_extract_local_dir, pipeline_pkl_transform_name) + '_' + obj_key + '.pkl'
        pickle.dump(res_dict[obj_key], open(pkl_path, 'wb'))

        ## Saving object in a s3 path
        upload_file_to_s3(pkl_path, base_bucket_name, pipeline_pkl_transform_name + '_' + obj_key + '.pkl')


    ## Saving the results stored in the dictionary as variables
    df_trans_train = res_dict['train']
    df_trans_test = res_dict['test']

    return df_trans_train, df_trans_test





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
