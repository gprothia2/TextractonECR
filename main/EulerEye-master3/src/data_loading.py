"""
data_loading.py
----------------
Holds read and write functions between SageMaker and S3 bucket
@author Zhu, Wenzhen (wenzhu@amazon.com)
@date   02/26/2020
"""

import boto3, botocore, logging
from PIL import Image
import numpy as np


def upload_to_s3(file_name, bucket, obj_name):
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file(file_name, bucket, obj_name)
    except botocore.exceptions.ClientError as e:
        logging.error(e)
        return False
    return True


def fetch_all_files(bucket_name, suffix, level=None, keyword=None):
    """
    Fetch all files with a specific suffix at a given level from S3 bucket
    :param bucket_name : str
        Name of the bucket
    :param suffix : str
        Suffix of files to fetch
        example: 'jpg', 'csv', 'xls', or '.jpg', '.csv', '.xls'
    :param level : int
        Specify which level's files
    :return: all files with given suffix in a proper level

    Example:
    To fetch all .txt files:
    >>> fetch_all_files(bucket_name, suffix='txt')
    To fetch all .txt files in level 3:
    >>> fetch_all_files(bucket_name, suffix='txt', level=3)
    To fetch all .txt files with 'ABBYY9' in the file name:
    >>> fetch_all_files(bucket_name, suffix='txt', keyword='ABBYY9')
    """
    s3 = boto3.resource("s3")
    s3_bucket = s3.Bucket(bucket_name)
    files = []
    for obj in s3_bucket.objects.all():
        key = obj.key
        if key.endswith(suffix):
            if keyword:
                if keyword in key:
                    files.append(key)
            else:
                files.append(key)

    if level:
        res = []
        for file in files:
            file2list = file.split("/")
            if len(file2list) == level + 1:
                res.append(file)
        return res
    else:
        return files


def read_text_from_s3(bucket_name, key):
    """
    Load text file from s3
    :param bucket_name : str
        name of s3 bucket
    :param key : str or list
        full path of a file
    :return: string or a list of string based on key type
    """
    s3 = boto3.resource("s3")
    s3_bucket = s3.Bucket(bucket_name)

    if type(key) == str:
        try:
            obj = s3_bucket.Object(key)
            file_stream = obj.get()["Body"].read().decode("utf-8")
            return file_stream
        except:
            print(
                "NoSuchKey: An error occurred (NoSuchKey) when calling the GetObject operation: "
                "The specified key does not exist."
            )

    elif type(key) == list:
        texts = []
        for f in key:
            obj = s3_bucket.Object(f)
            # TODO: Report this error to Guang about some non-ascii character in texts
            # Current fix:
            # https://stackoverflow.com/questions/22216076/unicodedecodeerror-utf8-codec-cant-decode-byte-0xa5-in-position-0-invalid-s
            file_stream = obj.get()["Body"].read().decode("utf-8", errors="replace")
            texts.append(file_stream)
        return texts
    else:
        raise ValueError("param key is unexpected")


def read_image_from_s3(bucket_name, key):
    """Load image file from s3.
    Parameters
    ----------
    bucket_name : string
        Bucket name
    key : string
        Path in s3

    Returns
    -------
    np array
        Image array
    """
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    obj = bucket.Object(key)
    try:
        response = obj.get()
        file_stream = response["Body"]
        im = Image.open(file_stream)
        return np.array(im)
    except:
        print("cannot get such key")
