## MODULE WITH UTIL FUNCTIONS - ML PROJECTS





"----------------------------------------------------------------------------------------------------------------------"
############################################# Imports ##################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--- Standard library imports ---"
import os
import pickle

"--- Third party imports ---"
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

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



## Saving dataset objects in a dictionary data structure
def dataset_objects_dict(dataset_objs_path):
    """
    Saving dataset objects in a dictionary data structure

    :param dataset_objs_path: (string) path where the dataset objects are stored locally as pickles
    :return dataset_dir: (dictionary) dict containing all the dataset objects (e.g. train_x, train_y, test_x, test_y)
    """


    ## List of locally saved pickles
    objects = os.listdir(dataset_objs_path)

    ## Dictionary where the results will be stored
    dataset_dir = {}

    ## Iterating over every dataset object, reading it, and saving it in the dictionary
    for obj in objects:

        ## Key to identify the object
        if 'train_x' in obj:
            key = 'train_x'
        elif 'train_y' in obj:
            key = 'train_y'
        elif 'test_x' in obj:
            key = 'test_x'
        elif 'test_y' in obj:
            key = 'test_y'
        else:
            continue

        ## Reading the object's contents
        with open(dataset_objs_path + obj, 'rb') as obj_content:
            dfx = pickle.load(obj)

        ## Saving the object's contents in a dictionary
        dataset_dir[key] = dfx


    return dataset_dir



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



## Magic loop: iterating over various models and hyper-parameters to find best parameters
def models_training_magic_loop(models_dict, train_x, train_y, eval_metric):
    """
    Magic loop: iterating over various models and hyper-parameters to find best parameters

    :param models_dict: (dictionary) dict containing different ML models and hyper-parameters
    :param train_x: (np.array) training data features
    :param train_y: (np.array) training data labels
    :return models_magic_loop: (dictionary) dict containing the best model per type based on the specified hyper-parameters
    """


    ## Results dictionary
    models_magic_loop = {}

    ## Magic loop
    for mod in models_dict:

        ## Scikit learn model
        model = models_dict[mod]['model']

        ## Grid search to find best model hyper-parameters
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=models_dict[mod]['param_grid'],
            scoring=eval_metric,
            return_train_score=True,
            n_jobs=-1,
        )

        ## Executing defined grid search
        grid_search.fit(train_x, train_y)

        ## Best model found
        models_magic_loop[mod] = {
            'best_estimator': grid_search.best_estimator_,
            'best_estimator_score': grid_search.best_score_,
        }


    return models_magic_loop





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
