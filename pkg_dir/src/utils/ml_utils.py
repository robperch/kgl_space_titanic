## MODULE WITH UTIL FUNCTIONS - ML PROJECTS





"----------------------------------------------------------------------------------------------------------------------"
############################################# Imports ##################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--- Standard library imports ---"


"--- Third party imports ---"
from sklearn.compose import ColumnTransformer

"--- Local application imports ---"
from pkg_dir.src.utils import *






"----------------------------------------------------------------------------------------------------------------------"
############### Generic functions ######################################################################################
"----------------------------------------------------------------------------------------------------------------------"


## Discerning between train and test datasets
def discern_between_train_and_test(obj):
    """
    Discerning between train and test datasets

    :param obj: (string) name of the extract object obtained from AWS S3
    :return obj_key: (string) keyword to indentify between train and test
    """


    ## Finding keyword in the object's name
    if 'train' in obj:
        obj_key = 'train'
    elif 'test' in obj:
        obj_key = 'test'
    else:
        raise NameError('No keyword was identified in the object')


    return obj_key



## Updating and saving data schema with new feature
def update_save_data_schema(data_schema, feature_name, new_feature_data_schema, dump_path, json_name):
    """
    Updating and saving data schema with new feature

    :param data_schema: (dict) full data schema containing base and new features
    :param feature_name: (string) name of the new feature
    :param new_feature_data_schema: data schema attributes of the new feature
    :param dump_path: (string) path where the json will be stored on local machine
    :param json_name: (string) name of the json file that will be stored
    :return:
    """


    ## Adding new feature to data schema
    data_schema[feature_name] = new_feature_data_schema

    ## Saving data schema locally as a json file
    dump_dir_as_json(data_schema, dump_path, json_name)


    return



## Applying the data processing pipeline to the features based on provided tuples
def apply_data_ppl_with_tuples(features, tuples):
    """
    Applying the data processing pipeline to the features based on provided tuples

    :param features: (pd.DataFrame) features to be processed
    :param tuples: (list) list containing the tuples needed for sklearn's ColumnTransformer
    :return processed_features: (np.array) array with features after being processed through the data pipeline
    """


    ## Defining the pipeline
    pipeline = ColumnTransformer(tuples)

    ## Executing the transformation
    processed_features = pipeline.fit_transform(features)


    return processed_features






"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
