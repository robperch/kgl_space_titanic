## MODULE WITH GENERAL PARAMETERS FOR THE PROJECT





"----------------------------------------------------------------------------------------------------------------------"
############################################# Imports ##################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--- Standard library imports ---"


"--- Third party imports ---"


"--- Local application imports ---"





"----------------------------------------------------------------------------------------------------------------------"
############### Data schema ############################################################################################
"----------------------------------------------------------------------------------------------------------------------"


## Data schema to handle data
titanicsp_base_data_schema = {

    'PassengerId': {
        'relevant': True,
        'clean_col_name': 'PassengerId',
        'data_type': 'str',
        'feature_type': 'categorical',
        'model_relevant': False,
        'id_feature': True,
    },

    'HomePlanet': {
        'relevant': True,
        'clean_col_name': 'HomePlanet',
        'data_type': 'str',
        'feature_type': 'categorical',
        'model_relevant': True,
        'imputation_strategy': 'most_frequent',
    },

    'CryoSleep': {
        'relevant': True,
        'clean_col_name': 'CryoSleep',
        'data_type': 'bool',
        'value_map': {
            'False': False,
            'True': True,
        },
        'feature_type': 'categorical',
        'model_relevant': True,
        'imputation_strategy': 'most_frequent',
    },

    'Cabin': {
        'relevant': False,
        'clean_col_name': 'Cabin',
        'data_type': 'str',
        'feature_type': 'categorical',
        'model_relevant': False,
    },

    'Destination': {
        'relevant': True,
        'clean_col_name': 'Destination',
        'data_type': 'str',
        'feature_type': 'categorical',
        'model_relevant': True,
        'imputation_strategy': 'most_frequent',
    },

    'Age': {
        'relevant': True,
        'clean_col_name': 'Age',
        'data_type': 'int',
        'feature_type': 'numerical',
        'model_relevant': True,
        'imputation_strategy': 'median',
    },

    'VIP': {
        'relevant': True,
        'clean_col_name': 'VIP',
        'data_type': 'bool',
        'value_map': {
            'False': False,
            'True': True,
        },
        'feature_type': 'categorical',
        'model_relevant': True,
        'imputation_strategy': 'most_frequent',
    },

    'RoomService': {
        'relevant': True,
        'clean_col_name': 'RoomService',
        'data_type': 'float',
        'feature_type': 'numerical',
        'model_relevant': False,
        'imputation_strategy': 'constant',
    },

    'FoodCourt': {
        'relevant': True,
        'clean_col_name': 'FoodCourt',
        'data_type': 'float',
        'feature_type': 'numerical',
        'model_relevant': False,
        'imputation_strategy': 'constant',
    },

    'ShoppingMall': {
        'relevant': True,
        'clean_col_name': 'ShoppingMall',
        'data_type': 'float',
        'feature_type': 'numerical',
        'model_relevant': False,
        'imputation_strategy': 'constant',
    },

    'Spa': {
        'relevant': True,
        'clean_col_name': 'Spa',
        'data_type': 'float',
        'feature_type': 'numerical',
        'model_relevant': False,
        'imputation_strategy': 'constant',
    },

    'VRDeck': {
        'relevant': True,
        'clean_col_name': 'VRDeck',
        'data_type': 'float',
        'feature_type': 'numerical',
        'model_relevant': False,
        'imputation_strategy': 'constant',
    },

    'Name': {
        'relevant': False,
        'clean_col_name': 'Name',
        'data_type': 'str',
        'feature_type': 'categorical',
        'model_relevant': False,
    },

    'Transported': {
        'relevant': True,
        'clean_col_name': 'label',
        'data_type': 'boolean',
        'value_map': {
            'False': False,
            'True': True,
        },
        'predict_label': True,
    },

}



## Data schema to handle data
titanicsp_full_data_schema = titanicsp_base_data_schema



## Parameters for local json containing updated data schema
local_json_path = "pkg_dir/src/parameters/"
json_name = "data_schema"





"----------------------------------------------------------------------------------------------------------------------"
############### XXX ####################################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--------------- XXX ---------------"





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
