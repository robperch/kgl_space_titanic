## MODULE WITH ML PARAMETERS FOR THE PROJECT





"----------------------------------------------------------------------------------------------------------------------"
############################################# Imports ##################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--- Standard library imports ---"
from functools import partial

"--- Third party imports ---"
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

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



"--------------- Prediction ML models ---------------"

predict_models_dict = {

    'random_forest': {
        'alias': 'randf',
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
        },
        'class_thresh': 0.4,
    },

    'decision_tree': {
        'alias': 'dect',
        'model': DecisionTreeClassifier(
            random_state=2222
        ),
        'param_grid': {
            'max_depth': [10, 15],
            'min_samples_leaf': [5]
        },
        'class_thresh': 0.5,
    },

}

predict_model_eval_metric = 'accuracy'

positive_label = True



"--------------- Model evaluation metrics ---------------"

model_eval_metrics = {

    'accuracy_score':
        {
            'alias': 'acc',
            'params': 'label/predict',
            'method': partial(metrics.accuracy_score),
        },

    'balanced_accuracy_score':
        {
            'alias': 'bacc',
            'params': 'label/predict',
            'method': partial(metrics.balanced_accuracy_score),
        },

    'average_precision_score':
        {
            'alias': 'avgprec',
            'params': 'label/proba',
            'method': partial(metrics.average_precision_score),
        },

    'f1_score':
        {
            'alias': 'f1',
            'params': 'label/predict',
            'method': partial(metrics.f1_score),
        },

}





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
