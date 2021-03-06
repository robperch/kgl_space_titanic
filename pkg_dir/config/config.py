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
dataset_files = os.path.join(dataset_dir, dataset_name)

## Path to training data
training_dataset = os.path.join(dataset_files, 'train.csv')

## Path to test data
test_dataset = os.path.join(dataset_files, 'test.csv')


"-------------- Pickles base path --------------"

## Pickles base location
pickles_dir_path = os.path.join(package_dir, 'data', 'pickles')


## Pipeline pickles location

### Base pipeline pickles dir path
pipeline_pickles_dir = os.path.join(pickles_dir_path, 'pipeline') + '/'

### Extract pickles
pipeline_pkl_extract_local_dir = os.path.join(pipeline_pickles_dir, 'extract') + '/'
pipeline_pkl_extract_name = 'extract'





"----------------------------------------------------------------------------------------------------------------------"
############################## AWS parameters ##########################################################################
"----------------------------------------------------------------------------------------------------------------------"


## S3 parameters

base_bucket_name = 'titanic-spaceship-aws-bucket'

### Pipeline pickles
aws_pipeline_pkl_extract = os.path.join(base_bucket_name, 'pipeline_pkls')





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
