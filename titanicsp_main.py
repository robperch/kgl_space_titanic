## MAIN MODULE TO COORDINATE THE EXECUTION OF THE PROJECT





"----------------------------------------------------------------------------------------------------------------------"
############################################# Imports ##################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--- Standard library imports ---"
import os


"--- Third party imports ---"


"--- Local application imports ---"
from pkg_dir.config import *
from pkg_dir.src.functions import *





"----------------------------------------------------------------------------------------------------------------------"
############### Main pipeline function definition ######################################################################
"----------------------------------------------------------------------------------------------------------------------"


## Main function to execute the complete ML pipeline
def pipeline_main_func():
    """
    Main function to execute the complete ML pipeline

    :return None:
    """


    ## Extract pipeline function
    # extract_pipeline_func()

    ## Transform pipeline function
    # transform_pipeline_func()

    ## Feature engineering pipeline function
    # feateng_pipeline_func()

    ## Models training pipeline function
    modtrain_pipeline_func()


    return





"----------------------------------------------------------------------------------------------------------------------"
############### Main pipeline function execution #######################################################################
"----------------------------------------------------------------------------------------------------------------------"


## Execution of main pipeline function
if __name__ == '__main__':
    pipeline_main_func()





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
