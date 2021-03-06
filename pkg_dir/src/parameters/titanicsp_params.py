## PYTHON TEMPLATE FILE





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
titanicsp_data_schema = {

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
    },

    'CryoSleep': {
        'relevant': True,
        'clean_col_name': 'CryoSleep',
        'data_type': 'bool',
        'value_map': {
            'False': False,
            'True': True,
        },
        'feature_type': 'boolean',
        'model_relevant': True,
    },

    'Cabin': {
        'relevant': True,
        'clean_col_name': 'Cabin',
        'data_type': 'str',
        'feature_type': 'categorical',
        'model_relevant': True,
    },

    'Destination': {
        'relevant': True,
        'clean_col_name': 'Destination',
        'data_type': 'str',
        'feature_type': 'categorical',
        'model_relevant': True,
    },

    'Age': {
        'relevant': True,
        'clean_col_name': 'Age',
        'data_type': 'int',
        'feature_type': 'numerical',
        'model_relevant': True,
    },

    'VIP': {
        'relevant': True,
        'clean_col_name': 'VIP',
        'data_type': 'bool',
        'value_map': {
            'False': False,
            'True': True,
        },
        'feature_type': 'boolean',
        'model_relevant': True,
    },

    'RoomService': {
        'relevant': True,
        'clean_col_name': 'RoomService',
        'data_type': 'float',
        'feature_type': 'numerical',
        'model_relevant': True,
    },

    'FoodCourt': {
        'relevant': True,
        'clean_col_name': 'FoodCourt',
        'data_type': 'float',
        'feature_type': 'numerical',
        'model_relevant': True,
    },

    'ShoppingMall': {
        'relevant': True,
        'clean_col_name': 'ShoppingMall',
        'data_type': 'float',
        'feature_type': 'numerical',
        'model_relevant': True,
    },

    'Spa': {
        'relevant': True,
        'clean_col_name': 'Spa',
        'data_type': 'float',
        'feature_type': 'numerical',
        'model_relevant': True,
    },

    'VRDeck': {
        'relevant': True,
        'clean_col_name': 'VRDeck',
        'data_type': 'float',
        'feature_type': 'numerical',
        'model_relevant': True,
    },

    'Name': {
        'relevant': True,
        'clean_col_name': 'Name',
        'data_type': 'str',
        'feature_type': 'categorical',
        'model_relevant': True,
    },

    'Transported': {
        'relevant': True,
        'clean_col_name': 'Transported',
        'data_type': 'str',
        'predict_label': True,
    },

}





"----------------------------------------------------------------------------------------------------------------------"
############### XXX ####################################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--------------- XXX ---------------"





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
