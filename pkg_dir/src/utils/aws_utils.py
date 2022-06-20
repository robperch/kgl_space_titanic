## MODULE WITH AWS UTILS





"----------------------------------------------------------------------------------------------------------------------"
############################################# Imports ##################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--- Standard library imports ---"
import os
import logging

"--- Third party imports ---"
import boto3
from botocore.exceptions import ClientError

"--- Local application imports ---"
from pkg_dir.src.utils import read_yaml
from pkg_dir.config import *





"----------------------------------------------------------------------------------------------------------------------"
############### AWS general functions ##################################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--------------- Unitary functions ---------------"

## Create session from locally stored credentials
def create_aws_session_from_local_yaml():
    """
    Create session from locally stored credentials

    :return aws_ses: (boto3.session.Session) aws session to interact with various aws services
    """


    ## Reading yaml file
    creds = read_yaml(creds_file_path)

    ## Creating session based on credentials
    aws_ses = boto3.Session(
        aws_access_key_id=creds['aws']['aws_access_key_id'],
        aws_secret_access_key=creds['aws']['aws_secret_access_key'],
    )


    return aws_ses






"----------------------------------------------------------------------------------------------------------------------"
############### S3 functions ###########################################################################################
"----------------------------------------------------------------------------------------------------------------------"


"--------------- Unitary functions ---------------"

## Setting up s3 client
def create_s3_client():
    """
    Setting up s3 client

    :return s3_client:
    """


    ## Creating session
    aws_ses = create_aws_session_from_local_yaml()

    ## Create client
    s3_client = aws_ses.client('s3')


    return s3_client



## Uploading file to s3 bucket
def upload_file_to_s3(file_path, bucket, object_name=None):
    """
    Upload a file to an S3 bucket

    :param file_path: (string) path to the file that will be uploaded
    :param bucket: (string) name of the bucket where the file will be uploaded
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """


    ## If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_path)

    ## Setting up s3 client
    s3_client = create_s3_client()


    ## Upload the file

    try:
        response = s3_client.upload_file(file_path, bucket, object_name)

    except ClientError as e:
        logging.error(e)
        print("Error uploading file to AWS bucket")
        return False

    print("Successfully uploaded file to AWS ({})".format(object_name))


    return True




"--------------- Compounded functions ---------------"





"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
############################################# END OF FILE ##############################################################
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
