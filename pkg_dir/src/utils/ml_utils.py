## MODULE WITH UTIL FUNCTIONS - ML PROJECTS





"----------------------------------------------------------------------------------------------------------------------"
############################################# Imports ##################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--- Standard library imports ---"
import os
import pickle

"--- Third party imports ---"
from sklearn.impute import SimpleImputer
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
            dfx = pickle.load(obj_content)

        ## Saving the object's contents in a dictionary
        dataset_dir[key] = dfx


    return dataset_dir



## Creating dictionary with list of features grouped by a defined attribute
def features_list_dict(dfx,  data_schema, group_attribute):
    """
    Creating dictionary with list of features grouped by a defined attribute

    :param dfx: (pd.DataFrame) df with all features
    :param data_schema: (dictionary) data schema containing all features
    :param group_attribute: (string) attribute used to group the lists of features
    :return feat_list_dict: (dictionary) dictionary with the type of feature as key and the list of features as value
    """


    ## Generating set with all different categories in data schema
    categories_set = {
        data_schema[feat][group_attribute]
        for feat in data_schema
        if
            group_attribute in data_schema[feat]
            and
            feat in dfx.columns
    }

    ## Initializing dictionary with results
    feat_list_dict = {}

    ## Iterating over categories and features to populate the dictionary's contents
    for cat in categories_set:
        feat_list_dict[cat] = [
            feat
            for feat in data_schema
            if
                group_attribute in data_schema[feat]
                and
                data_schema[feat][group_attribute] == cat
                and
                feat in dfx.columns
        ]


    return feat_list_dict



## Updating and saving data schema with new feature
def update_save_data_schema(data_schema, feature_name, new_feature_data_schema, dump_path, json_name):
    """
    Updating and saving data schema with new feature

    :param data_schema: (dict) full data schema containing base and new features
    :param feature_name: (string) name of the new feature
    :param new_feature_data_schema: (dictionary) data schema attributes of the new feature
    :param dump_path: (string) path where the json will be stored on local machine
    :param json_name: (string) name of the json file that will be stored
    :return:
    """


    ## Adding new feature to data schema
    data_schema[feature_name] = new_feature_data_schema

    ## Saving data schema locally as a json file
    dump_dir_as_json(data_schema, dump_path, json_name)


    return



## Apply imputation of feature's missing values
def apply_imputations(features, feat_imputation_dict, missing_values, fill_value):
    """
    Apply imputation of feature's missing values

    :param features: (pd.DataFrame) features with the missing values that will be imputed
    :param feat_imputation_dict: (dictionary) dict with the imputation strategy as key and the list of features for that key as values
    :param missing_values: (int, float, str, np.nan, None or pandas.NA) the placeholder for the missing values. All occurrences of missing_values will be imputed. For pandas’ dataframes with nullable integer dtypes with missing values, missing_values can be set to either np.nan or pd.NA.
    :param fill_value: (str or numerical value) when strategy == “constant”, fill_value is used to replace all occurrences of missing_values. If left to the default, fill_value will be 0 when imputing numerical data and “missing_value” for strings or object data types.
    :return features: (pd.DataFrame or similar) features with missing values imputed
    """


    ## Eliminating from the dictionary the features that have a 'custom' imputation strategy
    if 'custom' in {imp_stg for imp_stg in feat_imputation_dict}:
        feat_imputation_dict.pop('custom')

    ## Applying imputations
    for imp_stg in feat_imputation_dict:

        ## Defining imputer
        imputer = SimpleImputer(
            missing_values=missing_values,
            strategy=imp_stg,
            fill_value=fill_value,
        )

        ## Applying imputation
        features.loc[:, feat_imputation_dict[imp_stg]] = imputer.fit_transform(features.loc[:, feat_imputation_dict[imp_stg]])


    return features



## Applying the data processing pipeline to the features based on provided tuples
def apply_data_ppl_with_tuples(features, tuples):
    """
    Applying the data processing pipeline to the features based on provided tuples

    :param features: (pd.DataFrame or similar) features to be processed
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
