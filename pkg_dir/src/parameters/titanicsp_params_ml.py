## MODULE WITH ML PARAMETERS FOR THE PROJECT





"----------------------------------------------------------------------------------------------------------------------"
############################################# Imports ##################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--- Standard library imports ---"


"--- Third party imports ---"
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

"--- Local application imports ---"






"----------------------------------------------------------------------------------------------------------------------"
############### Scikit-Learn parameters ################################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--------------- Data transformation ---------------"

test_split_size = 0.2
random_state_split = 123



"--------------- Column Transformer pipeline ---------------"

## Categorical pipeline
categorical_ppl = Pipeline(
    [
        ('hotencode', OneHotEncoder())
    ]
)

## Numerical pipeline
numerical_ppl = Pipeline(
    [
        ('std_scaler', StandardScaler())
    ]
)



"--------------- Predicting ML models ---------------"

predict_models_dict = {

    'random_forest': {
        'short_name': 'rand_f',
        'model': RandomForestClassifier(
            max_features=10,
            n_estimators=10,
            max_leaf_nodes=50,
            oob_score=True,
            n_jobs=-1,
            random_state=1111
        ),
        'param_grid': {
            'n_estimators': [5, 7],
            'min_samples_leaf': [10],
            'criterion': ['gini']
        }
    },

    'decision_tree': {
        'short_name': 'dec_t',
        'model': DecisionTreeClassifier(
            random_state=2222
        ),
        'param_grid': {
            'max_depth': [10, 15],
            'min_samples_leaf': [5]
        }
    },

}

predict_model_eval_metric = 'accuracy'





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
