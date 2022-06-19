## MODULE TO EXTRACT DATA FROM SOURCE





"----------------------------------------------------------------------------------------------------------------------"
############################################# Imports ##################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--- Standard library imports ---"
import zipfile

"--- Third party imports ---"
import kaggle

"--- Local application imports ---"
from pkg_dir.config.config import *





"----------------------------------------------------------------------------------------------------------------------"
############### Extract general functions ##############################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--------------- Unitary functions ---------------"

## Downloading data from Kaggle if it's not present in the project's dir
def download_data_if_none():
    """
    Downloading data from Kaggle if it's not present in the project's dir

    :return:
    """


    ## Checking if the data file is in local directory
    if dataset_name in os.listdir(package_dir + '/data' + '/dataset'):
        print("Dataset already present locally... skipping download...")

    else:
        print("Dataset not present locally... downloading from source")

        ## Downloading dataset with kaggle's api
        kaggle.api.competition_download_files(
            'spaceship-titanic',
            path=dataset_dir,
        )

        ## Unzipping dir
        zip_file = dataset_dir + '/' + dataset_name + '.zip'
        dump_dir = os.path.join(dataset_dir, dataset_name)

        os.mkdir(dump_dir)

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dump_dir)


    return



## Saving train and test dataset locally as pickles



"--------------- Compounded functions ---------------"


## Extract pipeline function
def extract_pipeline_func():
    """
    Extract pipeline function

    :return None:
    """


    ## Downloading data from Kaggle if it's not present in the project's dir
    download_data_if_none()


    return





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
