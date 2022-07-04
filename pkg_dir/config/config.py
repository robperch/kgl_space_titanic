## CONFIGURATION FILE TO MANAGE PROJECT PATHS





"----------------------------------------------------------------------------------------------------------------------"
############################################# Imports ##################################################################
"----------------------------------------------------------------------------------------------------------------------"


## Standard library imports
import os

## Third party imports
from pytz import timezone

## Local application imports





"----------------------------------------------------------------------------------------------------------------------"
############################## Project path ############################################################################
"----------------------------------------------------------------------------------------------------------------------"


## Package directory
package_dir = os.path.dirname(os.path.dirname(__file__))

## Local credentials
creds_file_path = os.path.join(package_dir, "config", "local", "credentials.yaml")



"----------------------------------------------------------------------------------------------------------------------"
############################## Data files paths ########################################################################
"----------------------------------------------------------------------------------------------------------------------"


"-------------- Dataset files base path --------------"

## Dataset name
dataset_name = 'spaceship-titanic'

## Dataset dir
dataset_dir = os.path.join(package_dir, 'data', 'dataset')

## Dataset base file location
dataset_local_files = os.path.join(dataset_dir, dataset_name)

## Path to training data
training_dataset = os.path.join(dataset_local_files, 'train.csv')

## Path to test data
test_dataset = os.path.join(dataset_local_files, 'test.csv')


"-------------- Pickles base path --------------"

## Pickles base location
pickles_dir_path = os.path.join(package_dir, 'data', 'pickles')


## Pipeline pickles location

### Base pipeline pickles dir path
pipeline_pickles_dir = os.path.join(pickles_dir_path, 'pipeline') + '/'

### Extract pickles
pipeline_pkl_extract_name = 'extract'
pipeline_pkl_extract_local_dir = os.path.join(pipeline_pickles_dir, pipeline_pkl_extract_name) + '/'

### Transform pickles
pipeline_pkl_transform_name = 'trans'
pipeline_pkl_transform_local_dir = os.path.join(pipeline_pickles_dir, pipeline_pkl_transform_name) + '/'

### Feature engineering pickles
pipeline_pkl_feateng_name = 'feateng'
pipeline_pkl_feateng_local_dir = os.path.join(pipeline_pickles_dir, pipeline_pkl_feateng_name) + '/'

### Model training pickles
pipeline_pkl_modtrain_name = 'modtrain'
pipeline_pkl_modtrain_local_dir = os.path.join(pipeline_pickles_dir, pipeline_pkl_modtrain_name) + '/'

### Model selection pickles
pipeline_pkl_modsel_name = 'modsel'
pipeline_pkl_modsel_local_dir = os.path.join(pipeline_pickles_dir, pipeline_pkl_modsel_name) + '/'

### Bias and fairness pickles
pipeline_pkl_bifair_name = 'bifair'
pipeline_pkl_bifair_local_dir = os.path.join(pipeline_pickles_dir, pipeline_pkl_bifair_name) + '/'

### Prediction pickles
pipeline_pkl_predict_name = 'predict'
pipeline_pkl_predict_local_dir = os.path.join(pipeline_pickles_dir, pipeline_pkl_predict_name) + '/'





"----------------------------------------------------------------------------------------------------------------------"
############################## AWS parameters ##########################################################################
"----------------------------------------------------------------------------------------------------------------------"


## S3 parameters

base_bucket_name = 'titanic-spaceship-aws-bucket'

### Pipeline pickles
pipeline_pkl_extract_aws_key = os.path.join('pipeline_pkls', pipeline_pkl_extract_name)
pipeline_pkl_transform_aws_key = os.path.join('pipeline_pkls', pipeline_pkl_transform_name)
pipeline_pkl_feateng_aws_key = os.path.join('pipeline_pkls', pipeline_pkl_feateng_name)
pipeline_pkl_modtrain_aws_key = os.path.join('pipeline_pkls', pipeline_pkl_modtrain_name)





"----------------------------------------------------------------------------------------------------------------------"
############################## Useful parameters #######################################################################
"----------------------------------------------------------------------------------------------------------------------"


## Translation of keywords from english to spanish
word_translation = {

    "months": {

        'April': "Abril",
        'August': "Agosto",
        'December': "Diciembre",
        'February': "Febrero",
        'January': "Enero",
        'July': "Julio",
        'June': "Junio",
        'March': "Marzo",
        'May': "Mayo",
        'November': "Noviembre",
        'October': "Octubre",
        'September': "Septiembre",

    },

}





"----------------------------------------------------------------------------------------------------------------------"
############################## Time zone parameters ####################################################################
"----------------------------------------------------------------------------------------------------------------------"


## Relevant time zones
utc_tz = timezone('UTC')
mexico_tz = timezone('Mexico/General')





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
