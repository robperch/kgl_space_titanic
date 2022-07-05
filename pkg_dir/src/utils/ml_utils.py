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
from sklearn.model_selection import train_test_split

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
    :return dataset_dict: (dictionary) dict containing all the dataset objects (e.g. train_x, train_y, test_x, test_y)
    """


    ## List of locally saved pickles
    objects = os.listdir(dataset_objs_path)

    ## Dictionary where the results will be stored
    dataset_dict = {}

    ## Iterating over every dataset object, reading it, and saving it in the dictionary
    for obj in objects:

        ## Key to identify the object

        ### Training data
        if 'X_train' in obj:
            key = 'X_train'
        elif 'y_train' in obj:
            key = 'y_train'

        ### Validation data
        elif 'X_val' in obj:
            key = 'X_val'
        elif 'y_val' in obj:
            key = 'y_val'

        ### Test data
        elif 'X_test' in obj:
            key = 'X_test'
        elif 'y_test' in obj:
            key = 'y_test'

        ### Unidentified data
        else:
            continue

        ## Reading the object's contents
        with open(dataset_objs_path + obj, 'rb') as obj_content:
            dfx = pickle.load(obj_content)

        ## Saving the object's contents in a dictionary
        dataset_dict[key] = dfx


    return dataset_dict



## Saving locally the dataset objects as pickles
def save_dataset_objects_locally(dataset_dict, local_path, object_prefix):
    """
    Saving locally the dataset objects as pickles

    :param dataset_dict: (dictionary) dict containing all the dataset objects (e.g. train_x, train_y, test_x, test_y)
    :param local_path: (string) local path where the pickles will be saved
    :param object_prefix: (string) text preceding the object key in the dict (usually the pipeline section of the process)
    :return None:
    """


    ## Iterating over every dataset object and saving it as a pickle
    for ds_key in dataset_dict:

        df_obj = dataset_dict[ds_key]

        pkl_path = os.path.join(
            local_path,
            object_prefix,
        ) + '_' + ds_key + '.pkl'

        pickle.dump(df_obj, open(pkl_path, 'wb'))


    return



## Saving in the cloud the dataset objects that were locally saved as pickles
def save_dataset_objects_in_cloud(dataset_dict, local_path, cloud_provider, base_path, cloud_path, object_prefix):
    """
    Saving in the cloud the dataset objects that were locally saved as pickles

    :param dataset_dict: (dictionary) dict containing all the dataset objects (e.g. train_x, train_y, test_x, test_y)
    :param local_path: (string) local path where the pickles will be saved
    :param cloud_provider: (string) name of the cloud provider that will be used to store the results (options: 'aws', 'gcp', 'azure')
    :param base_path: (string) name of the base path (e.g. bucket name) where the results will be stored
    :param cloud_path: (string) path on top of the base path where the object will be located in the cloud
    :param object_prefix: (string) text preceding the object key in the dict (usually the pipeline section of the process)
    :return None:
    """


    ## Selecting the cloud provider
    if cloud_provider == 'aws':

        ## Iterating over every dataset object and saving it as a pickle
        for ds_key in dataset_dict:

            local_pkl_path = os.path.join(
                local_path,
                object_prefix,
            ) + '_' + ds_key + '.pkl'

            object_name = os.path.join(
                cloud_path,
                object_prefix,
            ) + '_' + ds_key + '.pkl'

            upload_file_to_s3(local_pkl_path, base_path, object_name)

    else:
        raise Exception('Cloud provider not identified')


    return



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





"----------------------------------------------------------------------------------------------------------------------"
############### Data transformation functions ##########################################################################
"----------------------------------------------------------------------------------------------------------------------"

## Splitting data in train and test (or validate)
def split_data_train_test(X_train, y_train, test_size, train_size=None, random_state=123):
    """
    ## Splitting data in train and test (or validate)

    :param data: (pd.Dataframe or similar) Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    :param test_size: (float or int) If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
    :param train_size: (float or int) If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.
    :param random_state: (int) Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
    :return X_train: (pd.Dataframe or similar) training data features
    :return y_train: (pd.Dataframe or similar) training data labels
    :return X_test: (pd.Dataframe or similar) test (or validation) data features
    :return y_test: (pd.Dataframe or similar) test (or validation) data labels (or validate)
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        train_size=None,
        random_state=random_state,
    )


    return X_train, X_test, y_train, y_test





"----------------------------------------------------------------------------------------------------------------------"
############### Feature engineering functions ##########################################################################
"----------------------------------------------------------------------------------------------------------------------"


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





"----------------------------------------------------------------------------------------------------------------------"
############### Models training functions ##############################################################################
"----------------------------------------------------------------------------------------------------------------------"


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
