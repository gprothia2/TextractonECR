import glob
import os, errno
import boto3
import json
import numpy as np

############################################################################
#                           FILE operation
############################################################################
# TODO: Separate file directory manipulation based on
#  1) S3 Bucket
#  2) local
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_file_name(path):
    [year, f_name] = path.split("/")[-2:]
    return year + "/" + f_name


def generate_file_name(file_name):
    prefix = "/".join(file_name.split("/")[:-4])
    [year, f_name] = file_name.split("/")[-2:]
    txt = f_name.replace("png", "txt")
    return prefix + "/result/ABBYY_cloud_sdk/" + year + "/" + txt


def make_directory(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_files(dir_name, extension):
    img_files = []
    for f in glob.iglob(dir_name + "**/*" + extension, recursive=True):
        img_files.append(f)
    return img_files


def get_valid_dir(dir_name):
    return glob.glob(dir_name + "*/")


def find_index(file_checkpoints, target_files):
    index = 0
    for f in target_files:
        index += 1
        if get_file_name(f) == file_checkpoints:
            return index

    raise ValueError("No index is found! Please check file checkpoints name")


def find_file(files, keyword):
    for f in files:
        if keyword in f:
            return f
    print("not found")


def remove_files(files, keyword):
    for f in files:
        if keyword in f:
            files.remove(f)
    return files


def take_last_k(file_name, k):
    return "/".join(file_name.split("/")[-1 * k :])


def generate_file_name_for_error(file_name):
    """
    :param file_name : str
        a png file: such as 'PNG/compress_1_over_2/1700-1899/063426457.png'
        or a txt file: such as 'NcomSampleData/GroundTruth/1700-1899/005633301.txt'
    :return: dict
        the corresponding textract result, abbyy result, and ground truth result as a dictionary
    Example:
    >>> generate_file_name_for_error('NcomSampleData/GroundTruth/1700-1899/005633301.txt')
    >>> {'textract': 'result/Amazon_Textract_compress_1_over_2/1700-1899/005633301.txt',
    >>>  'abbyy': 'NcomSampleData/Images/1700-1899/ABBYY9/1700-1899_005633301.txt',
    >>>  'ground_truth': 'NcomSampleData/GroundTruth/1700-1899/005633301.txt'}
    """
    last = take_last_k(file_name, 2)
    [year, f_name] = last.split("/")
    f_name = f_name.replace("png", "txt")
    key = year + "/" + f_name

    textract_file_name = "result/Amazon_Textract_compress_1_over_2/" + key
    abbyy_file_name = "NcomSampleData/Images/" + year + "/ABBYY9/" + year + "_" + f_name
    ground_truth_file_name = "NcomSampleData/GroundTruth/" + key

    return {
        "textract": textract_file_name,
        "abbyy": abbyy_file_name,
        "ground_truth": ground_truth_file_name,
    }


def fetch_all_files_in_s3_bucket(bucket_name, suffix, level=None, keyword=None):
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

    if level is not None:
        res = []
        for file in files:
            file2list = file.split("/")
            if len(file2list) == level + 1:
                res.append(file)
        return res
    else:
        return files


############################################################################
#                              FILE  IO
############################################################################
def initiate_file(file_name):
    file = open("%s" % (file_name), "a")
    file.seek(0)
    file.truncate()
    file.flush()
    return file


def write_file(result_file, lines):
    """
    save a nested listed of ocr output to file
    """
    file = initiate_file(result_file)
    for line in lines:
        if isinstance(line, list):
            file.write(" ".join(line) + "\n")
        else:
            file.write(line)
            file.write("\n")
    file.flush()

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
#     with open(json_path, 'w') as f:
#         # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
#         d = {k: float(v) for k, v in d.items()}
#         json.dump(d, f, indent=4)
    with open(json_path, 'w') as fout:
        json.dump(d, fout, cls=NpEncoder, indent=4)      
    
def load_ocr_json_output(file):
    """
    load ocr result in json format
    """
    with open(file, "r") as read_file:
        texts_w_coordinates_dict = json.load(read_file)
    return texts_w_coordinates_dict

def save_texts(texts, path):
    """
    save a list of texts to file
    """
    with open(path, "w") as f:
        for text in texts:
            f.write(text)
            f.write("\n")
